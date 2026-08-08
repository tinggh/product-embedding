"""
保色抗遮挡训练增强管道（微调阶段专用）。

设计原则：小幅扰动色相、正常扰动亮度/光影/遮挡——
- 变体（口味）间的颜色差异是大色块、大色相差异，必须保持敏感 → 不做大幅 hue 抖动、
  不灰度化、不 solarize；
- 货架灯源色温变化是小幅整体色相偏移（约 ±5~10°），必须稳健 → 保留小幅 hue 抖动
  （默认 ±0.02 ≈ ±7°），把色温漂移标记为非本质变化；
- brightness/contrast 正常抖动，覆盖明暗差异；
- 阴影/暗角/遮挡只调制亮度或挖块，不改色相。

hue 幅度是关键超参：P1 探针（颜色变体可分）失败就调小，P4 探针（光照稳健）失败就调大。

仅依赖 torchvision / numpy，不引入 albumentations。
"""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from dinov3.data.transforms import ResizeWithRatio

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 小幅 hue 抖动：覆盖货架光源色温漂移，同时远小于变体间的色相差异
DEFAULT_HUE = 0.02


class RandomPlasmaShadow:
    """随机低频阴影（模拟货架挡板/纵深阴影），只调制亮度，不改色相。"""

    def __init__(self, intensity_range=(0.3, 0.7), p=0.3):
        self.intensity_range = intensity_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return img
        w, h = img.size
        # 低频随机梯度场（粗网格 + 双线性放大），避免逐像素噪声破坏纹理
        gw, gh = 4, 4
        field = np.random.rand(gh, gw).astype(np.float32)
        field = np.array(
            Image.fromarray((field * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        lo, hi = self.intensity_range
        strength = np.random.uniform(lo, hi)
        shadow = 1.0 - strength * field
        arr = np.asarray(img, dtype=np.float32) * shadow[..., None]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class RandomVignetting:
    """渐晕暗角（模拟密集货架缝隙处的暗角渐变），只调制亮度。"""

    def __init__(self, max_strength=0.4, p=0.2):
        self.max_strength = max_strength
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return img
        w, h = img.size
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2 * np.random.uniform(0.7, 1.3), h / 2 * np.random.uniform(0.7, 1.3)
        dist = np.sqrt(((x - cx) / w) ** 2 + ((y - cy) / h) ** 2)
        dist = dist / dist.max()
        strength = np.random.uniform(0.0, self.max_strength)
        mask = 1.0 - strength * dist.astype(np.float32)
        arr = np.asarray(img, dtype=np.float32) * mask[..., None]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class CoarseDropout:
    """多孔随机丢弃（模拟吊牌/挡条遮挡），作用在 tensor 上，填 0（=normalize 后的均值附近）。"""

    def __init__(self, num_holes_range=(1, 4), hole_size_range=(0.05, 0.2), p=0.3):
        self.num_holes_range = num_holes_range
        self.hole_size_range = hole_size_range  # 相对边长比例
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > self.p:
            return img
        _, h, w = img.shape
        n = np.random.randint(self.num_holes_range[0], self.num_holes_range[1] + 1)
        for _ in range(n):
            hh = int(h * np.random.uniform(*self.hole_size_range))
            ww = int(w * np.random.uniform(*self.hole_size_range))
            y0 = np.random.randint(0, max(1, h - hh))
            x0 = np.random.randint(0, max(1, w - ww))
            img[:, y0 : y0 + hh, x0 : x0 + ww] = 0.0
        return img


def _color_safe_jitter(hue: float):
    """保色颜色扰动：brightness/contrast ±0.25，saturation ≤0.1，hue 小幅（默认 ±0.02）。"""
    return transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=hue)],
        p=0.5,
    )


def build_color_preserving_transform(input_size=224, hue=DEFAULT_HUE):
    """微调训练增强：pad 方形保长宽比 → RandomResizedCrop 多尺度 + 保色颜色扰动 + 阴影/暗角/遮挡。

    前置 ResizeWithRatio 与生产预处理（pad 成方形）保持一致，避免训练/推理分布不一致。
    """
    return transforms.Compose(
        [
            ResizeWithRatio(input_size + 32),
            transforms.RandomResizedCrop(input_size, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(),
            _color_safe_jitter(hue),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))],
                p=0.2,
            ),
            RandomPlasmaShadow(intensity_range=(0.3, 0.7), p=0.3),
            RandomVignetting(max_strength=0.4, p=0.2),
            transforms.ToTensor(),
            CoarseDropout(num_holes_range=(1, 4), hole_size_range=(0.05, 0.2), p=0.3),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(input_size=224):
    """与生产推理一致的预处理：pad 成方形后 resize。"""
    return transforms.Compose(
        [
            ResizeWithRatio(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class DualViewTransform:
    """双视图变换：同一图产出全局视图 + 局部视图（局部-全局一致性损失用）。

    返回 (global_tensor, local_tensor)。局部视图用更激进的裁剪范围模拟
    「只看到商品一部分」的查询条件，颜色扰动与全局视图同一策略。
    """

    def __init__(self, input_size=224, local_scale=(0.15, 0.5), hue=DEFAULT_HUE):
        self.global_transform = build_color_preserving_transform(input_size, hue=hue)
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=local_scale),
                transforms.RandomHorizontalFlip(),
                _color_safe_jitter(hue),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, img: Image.Image):
        return self.global_transform(img), self.local_transform(img)
