"""P×K batch sampler，混入同 Product Line 硬负样本，DDP 分片。

每个 batch 采 P 个类、每类 K 张图（P*K = batch_size）；以 hard_ratio 的概率
保证 batch 内至少 2 个类来自同一 product_line（同系列不同口味/变体互为硬负样本），
迫使度量损失学习变体间的细微颜色/文字差异边界。

hierarchy: {class_name: {"product_line": str, ...}}，由 app/build_hierarchy.py 生成。
所有 rank 用相同 seed 生成完整 batch 序列后按 rank 交错分片，保证 DDP 一致。
"""

import warnings
from typing import Optional

import numpy as np
from torch.utils.data.sampler import Sampler

from dinov3.distributed import get_rank, get_world_size


class HardNegativeBatchSampler(Sampler):
    def __init__(
        self,
        *,
        targets,
        class_names,
        hierarchy: dict,
        batch_size: int,
        samples_per_class: int = 4,
        hard_ratio: float = 0.5,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        self._targets = np.asarray(targets)
        self._batch_size = batch_size
        self._samples_per_class = samples_per_class
        self._classes_per_batch = max(2, batch_size // samples_per_class)
        self._hard_ratio = hard_ratio
        self._num_replicas = get_world_size() if num_replicas is None else num_replicas
        self._rank = get_rank() if rank is None else rank
        self._seed = seed
        self._epoch = 0

        # class_index -> sample indices
        self._class_to_indices = {}
        for idx, cls in enumerate(self._targets):
            self._class_to_indices.setdefault(int(cls), []).append(idx)
        self._all_classes = np.array(sorted(self._class_to_indices.keys()))

        # product_line -> 该类集合（仅保留含 >=2 个变体类的 product_line 作为硬负源）
        pl_to_classes = {}
        for cls_idx in self._all_classes:
            name = str(class_names[cls_idx])
            pl = hierarchy.get(name, {}).get("product_line", "")
            if pl:
                pl_to_classes.setdefault(pl, []).append(int(cls_idx))
        self._hard_pools = [v for v in pl_to_classes.values() if len(v) >= 2]
        if not self._hard_pools:
            warnings.warn(
                "HardNegativeBatchSampler: 没有含 >=2 个变体的 product_line，"
                "退化为普通 P×K 采样（检查 hierarchy.json）",
                stacklevel=1,
            )

    def _sample_class_indices(self, rng, cls_idx: int, k: int) -> np.ndarray:
        pool = self._class_to_indices[cls_idx]
        replace = len(pool) < k
        return rng.choice(pool, k, replace=replace)

    def _make_batch(self, rng) -> np.ndarray:
        p = self._classes_per_batch
        k = self._samples_per_class
        chosen = []
        if self._hard_pools and rng.random() < self._hard_ratio:
            # 从同一 product_line 取 2 个变体类作为硬负对，其余类随机
            pool = self._hard_pools[rng.integers(len(self._hard_pools))]
            hard = rng.choice(pool, 2, replace=False)
            chosen.extend(int(c) for c in hard)
            p -= 2
        if p > 0:
            easy = rng.choice(self._all_classes, p, replace=False)
            chosen.extend(int(c) for c in easy)
        batch = [self._sample_class_indices(rng, c, k) for c in chosen]
        return np.concatenate(batch)

    def _num_batches(self) -> int:
        # pad 到 num_replicas 的整数倍：保证所有 rank 迭代次数完全一致，
        # 否则总 batch 数为奇数时 rank0 多跑 1 个迭代，DDP 集合通信错位，
        # epoch 末 watchdog 超时 SIGABRT（与 DistributedSampler 的 padding 同理）
        n = (len(self._targets) + self._batch_size - 1) // self._batch_size
        r = self._num_replicas
        return ((n + r - 1) // r) * r

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._epoch)
        for i in range(self._num_batches()):
            if i % self._num_replicas == self._rank:
                yield self._make_batch(rng).tolist()
            else:
                # 保持各 rank RNG 消耗一致（batch 生成与是否分片无关）
                self._make_batch(rng)

    def __len__(self):
        return self._num_batches() // self._num_replicas

    def set_epoch(self, epoch: int):
        self._epoch = epoch
