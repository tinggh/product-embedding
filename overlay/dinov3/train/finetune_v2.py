"""
Description: 商品特征微调训练 v2（深解冻 + Sub-center ArcFace + 保色增强 + 局部-全局一致性）
Author: liuting, ting.liu@hanshow.com

与旧版 finetune.py 的关系：
- 旧版 finetune.py 保持不动，用于 E0 基线复现与历史 ckpt 测试；
- 本脚本为优化方案的训练入口，差异如下：
  * backbone 由 ProductEmbedder 封装（CLS + GeM 池化 + 投影，1024 维 L2 归一化嵌入）；
  * 解冻深度可控（--unfreeze_last），默认全解冻 + layer-wise lr decay（--llrd）；
  * 默认 Sub-center ArcFace（--num_subcenters）+ Center Loss（--center_lambda）；
  * 可选局部-全局一致性损失（--consistency_lambda > 0 时启用双视图）；
  * 可选同 Product Line 硬负样本 batch 采样（--hierarchy_json + --hard_ratio）；
  * 保色抗遮挡增强（--aug color_preserving，--hue 控制小幅色相扰动）；
  * AdamW + bf16 AMP + 梯度累积（opt.accum_steps），适配 4090-24G。
"""

import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import json
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from dinov3.loss.arcface_loss import ArcFaceLoss
from dinov3.loss.subcenter_arcface_loss import SubCenterArcFaceWithCenterLoss
from dinov3.loss.local_global_consistency import LocalGlobalConsistencyLoss
from dinov3.data.transforms import ResizeWithRatio
from dinov3.data.color_preserving_augs import (
    build_color_preserving_transform,
    build_eval_transform,
    DualViewTransform,
)
from dinov3.data.hard_negative_sampler import HardNegativeBatchSampler
from dinov3.models.embedder import build_product_embedder
from dinov3.data.datasets import RetailProduct

import argparse
import re

from app.log_module import logger

from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class opt:
    lr: float = 1e-4          # backbone 顶层 lr（浅层按 llrd 衰减）
    head_lr: float = 1e-3     # 池化/投影头与度量损失权重 lr
    llrd: float = 0.8         # layer-wise lr decay
    weight_decay: float = 5e-4
    print_freq: int = 100
    max_epoch: int = 100
    lr_step: int = 20
    save_interval: int = 10
    batchsize: int = 128      # per-GPU batch（4090-24G + ViT-L 全解冻 + bf16 的稳妥起点）
    accum_steps: int = 4      # 梯度累积，有效 batch = batchsize * world_size * accum_steps
    num_workers: int = 8


def setup_distributed():
    """Initialize the distributed environment."""
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    logger.info(
        f"Rank {rank} of {world_size}, local rank {local_rank} of {local_world_size}"
    )
    return rank, world_size, local_rank, local_world_size


def build_train_transform(args):
    """按 --aug 与 --consistency_lambda 构建训练变换（可能返回双视图）。"""
    if args.aug == "legacy":
        return transforms.Compose(
            [
                ResizeWithRatio(256),
                transforms.RandomCrop(224, padding=0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    if args.consistency_lambda > 0:
        return DualViewTransform(input_size=224, hue=args.hue)
    return build_color_preserving_transform(input_size=224, hue=args.hue)


def build_val_transform():
    return build_eval_transform(input_size=224)


def build_criterion(args, num_classes):
    if args.loss == "subcenter":
        return SubCenterArcFaceWithCenterLoss(
            1024,
            num_classes,
            num_subcenters=args.num_subcenters,
            margin=args.margin,
            scale=args.scale,
            center_lambda=args.center_lambda,
        )
    return ArcFaceLoss(1024, num_classes, margin=args.margin, scale=args.scale)


def load_model_weights(embedder, ckpt_path):
    """加载预训练/续训权重，兼容三种格式：

    1. ProductEmbedder 完整权重（含 proj./gem. 前缀 key）；
    2. 裸 backbone state_dict（官方 LVD 权重 / 旧 best.pth）；
    3. SSL teacher checkpoint（"teacher" 子字典，key 带 backbone. 前缀，含 dino_head 等）。
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint
    for key in ("model", "teacher"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state_dict = checkpoint[key]
            break
    state_dict = OrderedDict(
        {re.sub(r"^module\.", "", k): v for k, v in state_dict.items()}
    )
    # 丢弃 SSL 自蒸馏头
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not k.split(".", 1)[-1].startswith(("dino_head", "ibot_head"))
        and not k.startswith(("dino_head", "ibot_head"))
    }
    if any(k.startswith(("proj.", "gem.")) for k in state_dict):
        msg = embedder.load_state_dict(state_dict, strict=False)
    else:
        state_dict = OrderedDict(
            {re.sub(r"^backbone\.", "", k): v for k, v in state_dict.items()}
        )
        msg = embedder.backbone.load_state_dict(state_dict, strict=False)
    logger.info(
        "Pretrained weights found at {} and loaded with msg: {}".format(ckpt_path, msg)
    )


def eval_model(model, val_loader, criterion, device, rank=0):
    """
    Evaluate the model on validation dataset
    """
    model.eval()
    criterion.eval()
    total_loss = torch.tensor(0.0).to(device)
    correct = torch.tensor(0).to(device)
    total = torch.tensor(0).to(device)

    if rank == 0:
        logger.info(f"Starting evaluation with {len(val_loader)} batches")

    with torch.no_grad():
        for idx, (data_input, label) in enumerate(val_loader):
            data_input = data_input.to(device)
            label = label.to(device).long()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                embedding = model(data_input)
            loss, output = criterion(embedding.float(), label)

            total_loss += loss.item()

            pred = output.data.max(1)[1]
            correct += pred.eq(label.data).sum().item()
            total += label.size(0)
            if rank == 0 and idx % 10 == 0:
                logger.info(f"Evaluation progress: {idx+1}/{len(val_loader)} batches")

    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.barrier()

    avg_loss = total_loss.item() / total.item()
    accuracy = 100.0 * correct.item() / total.item()

    if rank == 0:
        logger.info(
            f"Evaluation completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

    return avg_loss, accuracy


def train_distributed(opt, args):
    rank, world_size, local_rank, local_world_size = setup_distributed()
    device = torch.device("cuda")
    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True
    extra_root = args.dataset_extra or args.dataset_root

    train_dataset = RetailProduct(
        split=RetailProduct.Split.TRAIN,
        root=args.dataset_root,
        extra=extra_root,
        transform=build_train_transform(args),
    )

    # batch 采样：提供 hierarchy.json 时启用同 Product Line 硬负样本混合
    batch_sampler = None
    train_sampler = None
    if args.hierarchy_json and args.hard_ratio > 0:
        with open(args.hierarchy_json) as f:
            hierarchy = json.load(f)
        class_ids = train_dataset._get_class_ids()
        class_names = train_dataset._get_class_names()
        names_by_id = {
            int(cid): str(name) for cid, name in zip(class_ids, class_names)
        }
        batch_sampler = HardNegativeBatchSampler(
            targets=train_dataset.get_targets(),
            class_names=names_by_id,
            hierarchy=hierarchy,
            batch_size=opt.batchsize,
            hard_ratio=args.hard_ratio,
            num_replicas=world_size,
            rank=rank,
        )
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    if batch_sampler is not None:
        # batch_sampler 与 batch_size/shuffle/sampler/drop_last 互斥，只能单独传
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=opt.num_workers,
            pin_memory=True,
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batchsize,
            shuffle=False,
            num_workers=opt.num_workers,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True,
        )

    class_ids = train_dataset._get_class_ids()
    num_classes = len(class_ids)
    criterion = build_criterion(args, num_classes)
    consistency_loss = (
        LocalGlobalConsistencyLoss() if args.consistency_lambda > 0 else None
    )

    val_dataset = RetailProduct(
        split=RetailProduct.Split.VAL,
        root=args.dataset_root,
        extra=extra_root,
        transform=build_val_transform(),
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    valdataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batchsize,
        shuffle=False,
        num_workers=opt.num_workers,
        sampler=val_sampler,
        pin_memory=True,
    )

    model = build_product_embedder(pooling=args.pooling, embed_dim=1024)

    if args.ckpt_path and not args.resume_path:
        load_model_weights(model, args.ckpt_path)

    # LLRD 深解冻：冻结范围之外的 block 按层衰减学习率
    param_groups = model.backbone_param_groups(
        base_lr=opt.lr, llrd=opt.llrd, unfreeze_last=args.unfreeze_last
    )
    param_groups.append({"params": model.head_parameters(), "lr": opt.head_lr})
    param_groups.append({"params": criterion.parameters(), "lr": opt.head_lr})

    model.to(device)
    criterion.to(device)
    # broadcast_buffers=False：双视图第二次前向会触发 buffer 广播（copy_ 原地修改），
    # 使第一次前向保存的 qkv.bias_mask version 失效导致 backward 报错；
    # bias_mask/rope periods 均为静态 buffer，无需广播。
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=opt.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    start_epoch = 0
    if args.resume_path:
        checkpoint = torch.load(args.resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        criterion.load_state_dict(checkpoint["head"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0)
        logger.info(f"Resumed from checkpoint at epoch {start_epoch}")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    dist.barrier()

    start = time.time()
    best_acc = 0
    for i in range(start_epoch, opt.max_epoch):
        if batch_sampler is not None:
            batch_sampler.set_epoch(i)
        else:
            train_sampler.set_epoch(i)
        model.train()
        criterion.train()
        for ii, data in enumerate(trainloader):
            if args.max_iters_per_epoch and ii >= args.max_iters_per_epoch:
                break  # 消融短跑：限制每 epoch 迭代数
            views, label = data
            label = label.to(device, non_blocking=True).long()

            # 双视图（局部-全局一致性）或单视图
            if isinstance(views, (list, tuple)) and len(views) == 2 and torch.is_tensor(views[0]):
                img_global = views[0].to(device, non_blocking=True)
                img_local = views[1].to(device, non_blocking=True)
            else:
                img_global = views.to(device, non_blocking=True)
                img_local = None

            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb_global = model(img_global)
                loss, output = criterion(emb_global.float(), label)
                if img_local is not None:
                    emb_local = model(img_local)
                    # 局部视图同样参与分类（部分→同 SKU），再加全局-局部一致性
                    loss_l, _ = criterion(emb_local.float(), label)
                    loss = loss + loss_l + args.consistency_lambda * consistency_loss(
                        emb_global.float(), emb_local.float()
                    )
                loss = loss / opt.accum_steps

            loss.backward()
            if (ii + 1) % opt.accum_steps == 0 or ii == len(trainloader) - 1:
                optimizer.step()
                optimizer.zero_grad()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0 and rank == 0:
                _, preds = torch.max(output, 1)
                acc = (preds == label).float().mean().item()
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                optlr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "{} train epoch {} iter {} {} iters/s lr {} loss {} acc {}".format(
                        time_str, i, iters, speed, optlr, loss.item() * opt.accum_steps, acc
                    )
                )

                start = time.time()

        if i % opt.save_interval == 0 or i == opt.max_epoch - 1:

            _, eval_acc = eval_model(model, valdataloader, criterion, device, rank)
            if rank == 0:
                logger.info(f"validataion -Epoch {i} - Accuracy: {eval_acc:.2f}%")
                save_path = osp.join(args.output_dir, "vitl_epoch_{}.pth".format(i))
                ckpt_dict = {
                    "model": model.state_dict(),
                    "head": criterion.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": i + 1,
                }
                if i == opt.max_epoch - 1:
                    ckpt_dict = {
                        "model": model.state_dict(),
                        "head": criterion.state_dict(),
                    }
                torch.save(ckpt_dict, save_path)

            if eval_acc > best_acc:
                best_acc = eval_acc
                save_best_path = osp.join(args.output_dir, "best.pth")
                ckpt_dict = {
                    "model": model.state_dict(),
                    "head": criterion.state_dict(),
                }
                torch.save(ckpt_dict, save_best_path)
                logger.info(
                    f"New best model saved at epoch {i} with accuracy: {eval_acc:.2f}%"
                )

        scheduler.step()
    dist.destroy_process_group()
    torch.cuda.empty_cache()


def load_checkpoint(model, criterion, checkpoint_path):
    """
    Load model and criterion from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model_state_dict = OrderedDict(
        {re.sub(r"^module\.", "", k): v for k, v in checkpoint["model"].items()}
    )
    model.load_state_dict(model_state_dict)

    if "head" in checkpoint:
        head_state_dict = OrderedDict(
            {re.sub(r"^module\.", "", k): v for k, v in checkpoint["head"].items()}
        )
        criterion.load_state_dict(head_state_dict)


def test_distributed(opt, args):
    rank, world_size, local_rank, local_world_size = setup_distributed()
    device = torch.device("cuda")
    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True
    extra_root = args.dataset_extra or args.dataset_root

    model = build_product_embedder(pooling=args.pooling, embed_dim=1024)

    train_dataset = RetailProduct(
        split=RetailProduct.Split.TRAIN,
        root=args.dataset_root,
        extra=extra_root,
        transform=build_val_transform(),
    )
    class_ids = train_dataset._get_class_ids()
    num_classes = len(class_ids)
    criterion = build_criterion(args, num_classes)

    val_dataset = RetailProduct(
        split=RetailProduct.Split.TEST,
        root=args.dataset_root,
        extra=extra_root,
        transform=build_val_transform(),
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    valdataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batchsize,
        shuffle=False,
        num_workers=opt.num_workers,
        sampler=val_sampler,
        pin_memory=True,
    )
    load_checkpoint(model, criterion, args.ckpt_path)
    model.to(device)
    criterion.to(device)

    eval_model(model, valdataloader, criterion, device, rank)


def parse_args():
    parser = argparse.ArgumentParser(description="model inference")
    parser.add_argument(
        "--dataset_root",
        default="/ya/Dataset/shelf/liuting/rec/tesco_skus_dataset",
        help="input file or directory",
    )
    parser.add_argument(
        "--ckpt_path",
        default="/ya/Code/liuting/modelscope/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        help="input pretrained ckpt file or test models",
    )
    parser.add_argument(
        "--resume_path",
        default="",
        help="resume training from checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        default="/ya/Code/liuting/runs/rec/dinov3_vitl_cls",
        help="output directory",
    )
    parser.add_argument("--train", action="store_true", help="whether train or test")
    # ---- 模型与解冻 ----
    parser.add_argument(
        "--pooling", default="cls+gem", choices=["cls", "gem", "cls+gem"],
        help="嵌入池化方式（cls = 旧行为）",
    )
    parser.add_argument(
        "--unfreeze_last", type=int, default=24,
        help="解冻最后 N 个 block（>=24 为全解冻，旧基线为 1）",
    )
    # ---- 损失 ----
    parser.add_argument(
        "--loss", default="subcenter", choices=["arcface", "subcenter"],
        help="度量损失（arcface = 旧行为）",
    )
    parser.add_argument("--num_subcenters", type=int, default=3, help="每类子中心数 K")
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=64.0)
    parser.add_argument("--center_lambda", type=float, default=0.5, help="Center Loss 权重")
    parser.add_argument(
        "--consistency_lambda", type=float, default=0.0,
        help="局部-全局一致性损失权重（>0 启用双视图训练）",
    )
    # ---- 增强 ----
    parser.add_argument(
        "--aug", default="color_preserving", choices=["legacy", "color_preserving"],
        help="训练增强管道（legacy = 旧行为）",
    )
    parser.add_argument(
        "--hue", type=float, default=0.02,
        help="保色增强的小幅色相扰动幅度（P1 失败调小，P4 失败调大）",
    )
    # ---- 硬负样本采样 ----
    parser.add_argument(
        "--hierarchy_json", default="",
        help="层级标签文件（app/build_hierarchy.py 生成），提供后启用硬负样本 batch 采样",
    )
    parser.add_argument("--hard_ratio", type=float, default=0.5, help="硬负样本 batch 比例")
    # ---- 训练超参 CLI 覆盖（默认 None 时用 opt dataclass 值） ----
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--batchsize", type=int, default=None, help="per-GPU batch")
    parser.add_argument("--accum_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument(
        "--dataset_extra", default="",
        help="npy 缓存目录（默认与 dataset_root 相同；数据集目录只读时指定）",
    )
    parser.add_argument(
        "--max_iters_per_epoch", type=int, default=0,
        help="每 epoch 最大迭代数（0=完整 epoch；消融短跑用，如 1500）",
    )
    args = parser.parse_args()
    return args


def apply_cli_overrides(opt, args):
    """CLI 覆盖 opt dataclass 的训练超参（冒烟/调参用）。"""
    for field in ("max_epoch", "batchsize", "accum_steps", "num_workers", "save_interval"):
        value = getattr(args, field, None)
        if value is not None:
            setattr(opt, field, value)
    return opt


if __name__ == "__main__":
    opt_instance = opt()
    args = parse_args()
    opt_instance = apply_cli_overrides(opt_instance, args)
    if args.train:
        train_distributed(opt_instance, args)
    else:
        test_distributed(opt_instance, args)
