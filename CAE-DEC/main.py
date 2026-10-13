# -*- coding: utf-8 -*-
"""
Floating Offshore Wind Siting: CAE-DEC Suitability Assessment
"""

import os
import gc
import time
import argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_bounds, Affine
from rasterio.crs import CRS

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.decomposition import PCA
from scipy.stats import wilcoxon
from scipy.ndimage import label as nd_label, binary_erosion

import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 全局常量
# ============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  cuDNN benchmark: on, TF32: on")

BATCH_SIZE = 32
N_INIT_KMEANS = 10
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-5

FINETUNE_LR = 5e-4
FINETUNE_PATIENCE = 20

NODATA_FLOAT = -9999.0
NODATA_INT = 0


# ============================================================
# 工具函数
# ============================================================
def _looks_like_lonlat(bounds):
    l, b, r, t = bounds
    if not (-180 <= l <= 180 and -180 <= r <= 180 and
            -90 <= b <= 90 and -90 <= t <= 90):
        return False
    if (r - l) > 30 or (t - b) > 30:
        return False
    if (r - l) <= 0 or (t - b) <= 0:
        return False
    return True


def _bounds_from_transform_raw(transform, shape):
    h, w = shape
    left, top = transform.c, transform.f
    right = left + transform.a * w
    bottom = top + transform.e * h
    return (min(left, right), min(top, bottom),
            max(left, right), max(top, bottom))


def _resolve_crs(src, calc_bounds, fname):
    meta_crs = src.crs
    if meta_crs is None:
        inferred = "EPSG:4326" if _looks_like_lonlat(calc_bounds) else "EPSG:32633"
        print(f"  [警告] {fname}: 元数据无 CRS, 根据 bounds 推断为 {inferred}")
        return inferred

    meta_str = meta_crs.to_string()
    a = abs(src.transform.a)
    c = abs(src.transform.c)
    f = abs(src.transform.f)

    looks_ll = (0 < a < 0.1) and (c < 180) and (f < 90)
    looks_utm = (a > 1) and (c > 10000)
    meta_is_ll = meta_crs.is_geographic

    if looks_ll and not meta_is_ll:
        print(f"  ⚠️  {fname}: 元数据 CRS={meta_str} 但 transform 数值是 "
              f"经纬度量级 (a={a:.6f}, c={c:.3f}, f={f:.3f})")
        print(f"      → 强制判定为 EPSG:4326")
        return "EPSG:4326"

    if looks_utm and meta_is_ll:
        print(f"  ⚠️  {fname}: 元数据 CRS={meta_str} 但 transform 数值是 "
              f"投影量级 (a={a:.2f}, c={c:.0f}, f={f:.0f})")
        print(f"      → 强制判定为 EPSG:32633")
        return "EPSG:32633"

    print(f"  [{fname}] 采用元数据 CRS = {meta_str} "
          f"(a={a:.4f}, c={c:.2f}, f={f:.2f})")
    return meta_str


def save_tif(array, profile, path, dtype="float32", nodata=None):
    p = profile.copy()
    if nodata is None:
        nodata = NODATA_FLOAT if np.issubdtype(
            np.dtype(dtype), np.floating) else NODATA_INT
    p.update(count=1, dtype=dtype, nodata=nodata, compress="lzw")
    arr = array.astype(dtype)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.where(np.isnan(arr), nodata, arr)
        arr = np.where(np.isinf(arr), nodata, arr)
    valid = arr != nodata
    if valid.any():
        vmin = float(arr[valid].min())
        vmax = float(arr[valid].max())
        vmean = float(arr[valid].mean())
    else:
        vmin, vmax, vmean = 0.0, 0.0, 0.0
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr, 1)
        dst.update_tags(STATISTICS_MINIMUM=str(vmin),
                        STATISTICS_MAXIMUM=str(vmax),
                        STATISTICS_MEAN=str(vmean))
    print(f"已保存: {path}  (nodata={nodata}, "
          f"valid={int(valid.sum()):,}/{arr.size:,}, "
          f"min={vmin:.4f}, max={vmax:.4f})")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def check_input_resolutions(tif_paths):
    print("--- 输入数据分辨率预检 ---")
    resolutions = []
    for p in tif_paths:
        fname = os.path.basename(p)
        try:
            with rasterio.open(p) as src:
                res_x = abs(src.transform.a)
                res_y = abs(src.transform.e)
                crs_obj = src.crs
                crs_str = crs_obj.to_string() if crs_obj else "None"
                a = abs(src.transform.a)
                c = abs(src.transform.c)
                f_val = abs(src.transform.f)
                looks_ll = (0 < a < 0.1) and (c < 180) and (f_val < 90)
                if looks_ll:
                    rx_m, ry_m = a * 111320.0, abs(src.transform.e) * 111320.0
                    note = f"(度 ≈ {rx_m:.0f} m)"
                else:
                    rx_m, ry_m = res_x, res_y
                    note = ""
                resolutions.append(max(abs(rx_m), abs(ry_m)))
                warn = ""
                if rx_m < 1 or ry_m < 1:
                    warn = "  ← [警告] 分辨率异常"
                print(f"  {fname}: ≈ {rx_m:.1f} × {ry_m:.1f} m, "
                      f"CRS={crs_str}, shape={src.shape} {note}{warn}")
        except Exception as e:
            print(f"  {fname}: 读取失败 {e}")
    if resolutions:
        print(f"  → 输入分辨率范围: {min(resolutions):.1f} ~ "
              f"{max(resolutions):.1f} m")
        return max(resolutions)
    return None


def _is_degenerate_bounds(bounds, min_span=1e-6):
    l, b, r, t = bounds
    return (r - l) <= min_span or (t - b) <= min_span


def _infer_transform_for_broken(shape, ref_bounds_utm):
    h, w = shape
    l, b, r, t = ref_bounds_utm
    xres = (r - l) / w
    yres = (t - b) / h
    return Affine(xres, 0, l, 0, -yres, t)


# ============================================================
# 1. 数据对齐
# ============================================================
def align_rasters(tif_paths, target_resolution=100, target_crs="EPSG:32633"):
    use_crs = CRS.from_string(target_crs)
    print("  --- 读取源数据 ---")
    sources = []
    for i, path in enumerate(tif_paths):
        fname = os.path.basename(path)
        with rasterio.open(path) as src:
            raw = src.read(1)
            transform = src.transform
            shape = src.shape
            nodata_val = src.nodata
            data = raw.astype(np.float32)
            calc_bounds = _bounds_from_transform_raw(transform, shape)
            real_crs = _resolve_crs(src, calc_bounds, fname)
            if nodata_val is not None and not (
                    isinstance(nodata_val, float) and np.isnan(nodata_val)):
                data[raw == nodata_val] = np.nan

            is_deg = _is_degenerate_bounds(calc_bounds)
            flag = "  ⚠️ transform 退化!" if is_deg else ""
            print(f"  [{i+1}/{len(tif_paths)}] {fname}, "
                  f"shape={shape}, CRS={real_crs}{flag}")
            sources.append({"name": fname, "data": data,
                            "transform": transform, "crs": real_crs,
                            "bounds": calc_bounds,
                            "shape": shape,
                            "is_degenerate": is_deg})

    valid = [s for s in sources if not s["is_degenerate"]]
    if not valid:
        raise ValueError("❌ 所有输入文件的 transform 均已损坏, "
                         "请先在 QGIS 中修复源文件后再运行")

    ref = max(valid,
              key=lambda s: (s["bounds"][2] - s["bounds"][0]) *
                            (s["bounds"][3] - s["bounds"][1]))
    if ref["name"] != sources[0]["name"]:
        print(f"\n  ℹ️  参考栅格切换为 {ref['name']} (面积最大)")

    ref_bounds_utm = (transform_bounds(ref["crs"], use_crs, *ref["bounds"])
                      if ref["crs"] != target_crs else ref["bounds"])
    left, bottom, right, top = ref_bounds_utm
    ref_width = int((right - left) / target_resolution)
    ref_height = int((top - bottom) / target_resolution)
    if ref_width <= 0 or ref_height <= 0:
        raise ValueError(f"目标栅格尺寸无效: {ref_width} × {ref_height}")

    n_pixels = ref_width * ref_height
    mem_gb = n_pixels * len(tif_paths) * 4 / (1024 ** 3)
    print(f"\n  参考栅格: {ref['name']}")
    print(f"  目标网格: {ref_width} × {ref_height} "
          f"({n_pixels/1e6:.1f} 百万像元)")
    print(f"  预计对齐结果内存占用: {mem_gb:.2f} GB")

    ref_transform = from_bounds(left, bottom, right, top,
                                ref_width, ref_height)

    n_fixed = 0
    for s in sources:
        if not s["is_degenerate"]:
            continue
        inferred = _infer_transform_for_broken(s["shape"], ref_bounds_utm)
        s["transform"] = inferred
        s["crs"] = target_crs
        n_fixed += 1
        print(f"  ⚠️  [{s['name']}] transform 真丢失, 已推断:")
        print(f"      假设覆盖同研究区, 推断分辨率 ≈ "
              f"{abs(inferred.a):.1f} m × {abs(inferred.e):.1f} m")
    if n_fixed > 0:
        print(f"  → 共 {n_fixed} 个文件使用推断 transform")

    aligned = np.full((len(tif_paths), ref_height, ref_width),
                      np.nan, dtype=np.float32)
    for i, s in enumerate(sources):
        fname = s["name"]
        method = (Resampling.nearest if "易发性" in fname
                  else Resampling.bilinear)
        dst = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
        try:
            reproject(source=s["data"], destination=dst,
                      src_transform=s["transform"], src_crs=s["crs"],
                      dst_transform=ref_transform, dst_crs=target_crs,
                      resampling=method, src_nodata=np.nan,
                      dst_nodata=np.nan)
        except Exception as e:
            print(f"  [{fname}] 重投影失败: {e}")
            raise
        n_valid = int(np.sum(~np.isnan(dst)))
        print(f"  [{fname}] 有效像元: {n_valid:,} "
              f"({n_valid/dst.size:.4%})")
        if n_valid == 0:
            print(f"  ⚠️  [{fname}] 重投影后全为 NaN")
        aligned[i] = dst
        gc.collect()

    nan_mask = np.isnan(aligned).all(axis=(1, 2))
    if nan_mask.all():
        raise ValueError("❌ 所有因子对齐后全为 NaN, 无法继续")
    if nan_mask.sum() > 0:
        print(f"\n  ⚠️  {int(nan_mask.sum())} 个因子无有效数据:")
        for j, bad in enumerate(nan_mask):
            if bad:
                print(f"     - {sources[j]['name']}")

    profile = {"driver": "GTiff", "height": ref_height, "width": ref_width,
               "count": 1, "dtype": "float32", "crs": use_crs,
               "transform": ref_transform, "nodata": NODATA_FLOAT,
               "compress": "lzw"}
    return aligned, profile


# ============================================================
# 2. 硬约束 + 软评分
# ============================================================
def build_constraints(aligned, landslide_class,
                      depth_min=50.0, depth_max=1000.0,
                      offshore_min_km=12.0, channel_min_km=5.0,
                      wind_min=6.0, wpd_min=200.0, landslide_max=3):
    depth_neg, offshore_m, channel_m, wind, wpd, _ = aligned
    depth_abs = np.abs(depth_neg)
    offshore_km = offshore_m / 1000.0
    channel_km = channel_m / 1000.0

    print(f"  水深范围: {np.nanmin(depth_abs):.1f} ~ "
          f"{np.nanmax(depth_abs):.1f} m")
    print(f"  离岸距离: {np.nanmin(offshore_km):.2f} ~ "
          f"{np.nanmax(offshore_km):.2f} km")
    print(f"  距航道: {np.nanmin(channel_km):.2f} ~ "
          f"{np.nanmax(channel_km):.2f} km")
    print(f"  风速: {np.nanmin(wind):.3f} ~ {np.nanmax(wind):.3f} m/s")
    print(f"  WPD: {np.nanmin(wpd):.2f} ~ {np.nanmax(wpd):.2f} W/m²")
    print(f"  滑坡易发性上限: ≤{landslide_max} 级")

    landslide_score = np.where(np.isnan(landslide_class), np.nan,
                               1.0 - (landslide_class - 1.0) / 4.0
                               ).astype(np.float32)

    mask_hard = ((depth_abs >= depth_min) & (depth_abs <= depth_max) &
                 (offshore_km >= offshore_min_km) &
                 (channel_km >= channel_min_km) &
                 (wind >= wind_min) & (wpd >= wpd_min) &
                 (landslide_class <= landslide_max))
    mask_hard &= ~np.isnan(aligned).any(axis=0)
    print(f"  候选区像元: {mask_hard.sum():,}, 占比: {mask_hard.mean():.2%}")

    def norm_pos(x):
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo + 1e-8)

    soft = np.stack([
        norm_pos(-depth_abs), norm_pos(-offshore_km),
        norm_pos(channel_km), norm_pos(wind), norm_pos(wpd),
        landslide_score,
    ], axis=0).astype(np.float32)
    soft = np.nan_to_num(soft, nan=0.0) * mask_hard[None, :, :]
    return mask_hard, soft, landslide_score, depth_abs, offshore_km, channel_km


# ============================================================
# 3. Patch 提取
# ============================================================
def _fill_patch(p, m):
    m_b = m[None, :, :].astype(np.float32)
    denom = np.maximum(m_b.sum(axis=(1, 2), keepdims=True), 1e-6)
    means = (p * m_b).sum(axis=(1, 2), keepdims=True) / denom
    return (p * m_b + means * (1.0 - m_b)).astype(np.float32)


def extract_patches(factors, mask, patch_size=58, stride=29,
                    min_valid_ratio=0.4, max_patches=30000,
                    rng=None):
    if rng is None:
        rng = np.random.RandomState(RANDOM_STATE)
    n_f, H, W = factors.shape
    patches, masks, coords = [], [], []
    n_scanned = n_ratio_fail = 0
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            n_scanned += 1
            m = mask[i:i+patch_size, j:j+patch_size]
            if m.mean() < min_valid_ratio:
                n_ratio_fail += 1
                continue
            p = factors[:, i:i+patch_size, j:j+patch_size].copy()
            patches.append(_fill_patch(p, m))
            masks.append(m.astype(np.float32))
            coords.append((i, j))
    print(f"  扫描 {n_scanned} 个窗口, 丢弃 {n_ratio_fail} 个, "
          f"保留 {len(patches)} 个")
    if len(patches) == 0:
        raise ValueError("未提取到任何patch")
    if len(patches) > max_patches:
        idx = rng.choice(len(patches), max_patches, replace=False)
        patches = [patches[k] for k in idx]
        masks = [masks[k] for k in idx]
        coords = [coords[k] for k in idx]
    total_mb = sum(p.nbytes for p in patches) / (1024 ** 2)
    print(f"提取 {len(patches)} 个 patch (均值填充), 约 {total_mb:.0f} MB")
    return np.stack(patches), np.stack(masks), np.array(coords)


# ============================================================
# 4. 熵权法
# ============================================================
def _patch_masked_means(patches, masks):
    N = patches.shape[0]
    C = patches.shape[1]
    means = np.zeros((N, C), dtype=np.float32)
    for i in range(N):
        m = masks[i][None, :, :]
        denom = max(m.sum(), 1e-6)
        means[i] = (patches[i] * m).sum(axis=(1, 2)) / denom
    return means


def compute_entropy_weights(patch_means):
    X = patch_means.astype(np.float64)
    n = X.shape[0]
    X_min, X_max = X.min(0, keepdims=True), X.max(0, keepdims=True)
    X_norm = (X - X_min) / (X_max - X_min + 1e-12) * 0.99 + 0.01
    P = X_norm / X_norm.sum(0, keepdims=True)
    E = -np.sum(P * np.log(P), 0) / np.log(n)
    d = 1.0 - E
    return (d / (d.sum() + 1e-12)).astype(np.float32)


def compute_prior_scores(patch_means, entropy_weights):
    lo, hi = patch_means.min(0, keepdims=True), \
             patch_means.max(0, keepdims=True)
    pm_n = (patch_means - lo) / (hi - lo + 1e-8)
    return (pm_n * entropy_weights[None, :]).sum(axis=1).astype(np.float32)


# ============================================================
# 5. 模型
# ============================================================
class SpatialCAE(nn.Module):
    def __init__(self, n_factors=6, embedding_dim=32, patch_size=58):
        super().__init__()
        self.patch_size = patch_size
        self.n_factors = n_factors

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(n_factors, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128), nn.ReLU(inplace=True),
            nn.Linear(128, embedding_dim),
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(embedding_dim, 128 * 7 * 7), nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 7, 7)),
        )
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Upsample(size=patch_size, mode="bilinear",
                        align_corners=False),
            nn.Conv2d(32, n_factors, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        return self.encoder_fc(h)

    def decode(self, z):
        h = self.decoder_fc(z)
        return self.decoder_conv(h)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class ConstrainedCAEDEC(nn.Module):
    def __init__(self, n_factors=6, n_clusters=4, embedding_dim=32,
                 patch_size=58):
        super().__init__()
        self.cae = SpatialCAE(n_factors, embedding_dim, patch_size)
        self.centroids = nn.Parameter(torch.randn(n_clusters, embedding_dim))
        self.mu_k = nn.Parameter(torch.linspace(0.1, 0.9, n_clusters))
        self.n_clusters = n_clusters

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.centroids) ** 2,
                                    dim=2))
        return q / (q.sum(dim=1, keepdim=True) + 1e-8)

    def target_dist(self, q):
        w = q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-8)
        return w / (w.sum(dim=1, keepdim=True) + 1e-8)

    def predict_suitability(self, q):
        return (q * self.mu_k.unsqueeze(0)).sum(dim=1)

    def forward(self, x):
        x_recon, z = self.cae(x)
        q = self.soft_assign(z)
        return x_recon, z, q, self.predict_suitability(q)


class FlatAE(nn.Module):
    def __init__(self, input_shape, embedding_dim=32, hidden_dim=1024):
        super().__init__()
        self.input_shape = tuple(input_shape)
        input_dim = int(np.prod(self.input_shape))
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z).view(-1, *self.input_shape)
        return recon, z


class FlatConstrainedDEC(nn.Module):
    def __init__(self, input_shape, n_clusters=4, embedding_dim=32):
        super().__init__()
        self.cae = FlatAE(input_shape, embedding_dim)
        self.centroids = nn.Parameter(torch.randn(n_clusters, embedding_dim))
        self.mu_k = nn.Parameter(torch.linspace(0.1, 0.9, n_clusters))
        self.n_clusters = n_clusters

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.centroids) ** 2,
                                    dim=2))
        return q / (q.sum(dim=1, keepdim=True) + 1e-8)

    def target_dist(self, q):
        w = q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-8)
        return w / (w.sum(dim=1, keepdim=True) + 1e-8)

    def predict_suitability(self, q):
        return (q * self.mu_k.unsqueeze(0)).sum(dim=1)

    def forward(self, x):
        x_recon, z = self.cae(x)
        q = self.soft_assign(z)
        return x_recon, z, q, self.predict_suitability(q)


# ============================================================
# 6. 训练
# ============================================================
def _make_loader_with_prior(patches, s_prior, batch_size=BATCH_SIZE,
                            shuffle=False):
    X = torch.tensor(patches, dtype=torch.float32)
    S = torch.tensor(s_prior, dtype=torch.float32)
    return DataLoader(TensorDataset(X, S), batch_size=batch_size,
                      shuffle=shuffle, drop_last=False,
                      pin_memory=(DEVICE.type == "cuda"))


def pretrain_cae_only(model, loader, epochs=120, lr=1e-3, verbose=True,
                      patience=EARLY_STOP_PATIENCE):
    opt = torch.optim.Adam(model.cae.parameters(), lr=lr)
    mse = nn.MSELoss()
    model.train()
    history = []
    best_loss = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    for ep in range(epochs):
        total = 0.0
        for batch in loader:
            xb = batch[0].to(DEVICE, non_blocking=True)
            opt.zero_grad()
            x_recon, _ = model.cae(xb)
            loss = mse(x_recon, xb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        avg = total / len(loader.dataset)
        history.append(avg)

        if avg < best_loss - EARLY_STOP_MIN_DELTA:
            best_loss = avg
            best_state = {k: v.detach().clone()
                          for k, v in model.cae.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (ep + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"[Pretrain] Epoch {ep+1}/{epochs}, Loss={avg:.6f}, "
                  f"No-imp={no_improve}, {elapsed:.0f}s")

        if no_improve >= patience:
            if verbose:
                print(f"[Pretrain] 早停于 epoch {ep+1} "
                      f"(best={best_loss:.6f}, {time.time()-t0:.0f}s)")
            break

    if best_state is not None:
        model.cae.load_state_dict(best_state)
    return history


def finetune_constrained_dec(model, loader, epochs=120, lr=None,
                             gamma=1.0, lam=0.01, verbose=True, seed=None,
                             patience=None):
    if seed is None:
        seed = RANDOM_STATE
    if lr is None:
        lr = FINETUNE_LR
    if patience is None:
        patience = FINETUNE_PATIENCE

    model.eval()
    z_list, s_list = [], []
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(DEVICE, non_blocking=True)
            _, z = model.cae(xb)
            z_list.append(z)
            s_list.append(batch[1])
    all_z = torch.cat(z_list, 0).cpu().numpy()
    s_all = torch.cat(s_list, 0).numpy()

    km = KMeans(n_clusters=model.n_clusters, n_init=N_INIT_KMEANS,
                random_state=seed)
    km.fit(all_z)
    model.centroids.data = torch.tensor(
        km.cluster_centers_, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        z_t = torch.tensor(all_z, dtype=torch.float32).to(DEVICE)
        labels_init = model.soft_assign(z_t).argmax(1).cpu().numpy()

    cluster_prior_means = np.array([
        s_all[labels_init == c].mean() if (labels_init == c).sum() > 0
        else 0.5 for c in range(model.n_clusters)])
    sorted_idx = np.argsort(cluster_prior_means)
    mu_init = np.zeros(model.n_clusters)
    mu_values = np.linspace(0.1, 0.9, model.n_clusters)
    for rank, idx in enumerate(sorted_idx):
        mu_init[idx] = mu_values[rank]
    model.mu_k.data = torch.tensor(mu_init, dtype=torch.float32).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    history = {"total": [], "recon": [], "kl": [], "prior": [], "mu_k": []}
    best_loss = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        total = tot_recon = tot_kl = tot_prior = 0.0
        for batch in loader:
            xb = batch[0].to(DEVICE, non_blocking=True)
            s_batch = batch[1].to(DEVICE, non_blocking=True)
            opt.zero_grad()
            x_recon, z, q, s_hat = model(xb)
            p = model.target_dist(q).detach()
            L_recon = mse(x_recon, xb)
            L_KL = F.kl_div(q.log(), p, reduction="batchmean")
            L_prior = mse(s_hat, s_batch)
            loss = L_recon + gamma * L_KL + lam * L_prior
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            tot_recon += L_recon.item() * len(xb)
            tot_kl += L_KL.item() * len(xb)
            tot_prior += L_prior.item() * len(xb)

        n = len(loader.dataset)
        avg = total / n
        history["total"].append(avg)
        history["recon"].append(tot_recon / n)
        history["kl"].append(tot_kl / n)
        history["prior"].append(tot_prior / n)
        history["mu_k"].append(model.mu_k.detach().cpu().numpy().copy())

        if avg < best_loss - EARLY_STOP_MIN_DELTA:
            best_loss = avg
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (ep + 1) % 20 == 0:
            mu_str = np.round(model.mu_k.detach().cpu().numpy(), 3)
            print(f"[Finetune] Epoch {ep+1}/{epochs}, "
                  f"Total={avg:.6f}, No-imp={no_improve}, "
                  f"mu_k={mu_str}, {time.time()-t0:.0f}s")

        if no_improve >= patience:
            if verbose:
                print(f"[Finetune] 早停于 epoch {ep+1} "
                      f"(best={best_loss:.6f}, {time.time()-t0:.0f}s)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ============================================================
# 7. CSI 计算与 Jenks
# ============================================================
def compute_csi_and_map(model, patches, coords, mask, patch_size,
                        H, W, n_clusters):
    model.eval()
    csi_patches = np.zeros(len(coords), dtype=np.float32)
    q_all = np.zeros((len(coords), n_clusters), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(coords), 256):
            xb = torch.tensor(patches[i:i+256],
                              dtype=torch.float32).to(DEVICE)
            _, _, q, s_hat = model(xb)
            csi_patches[i:i+256] = s_hat.cpu().numpy()
            q_all[i:i+256] = q.cpu().numpy()
    csi_map = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for k, (i, j) in enumerate(coords):
        csi_map[i:i+patch_size, j:j+patch_size] += csi_patches[k]
        count[i:i+patch_size, j:j+patch_size] += 1
    count[count == 0] = 1
    csi_map /= count
    csi_map[~mask] = np.nan
    return csi_map, csi_patches, q_all


def jenks_breaks_n(values, n_classes=3, max_sample=5000):
    valid = values[~np.isnan(values)].flatten()
    if len(valid) == 0:
        raise ValueError("待分类数据全为NaN")
    sample = np.sort(valid[np.random.choice(len(valid), max_sample,
                                             replace=False)]
                     if len(valid) > max_sample else valid)
    n = len(sample)
    cs = np.concatenate([[0.0], np.cumsum(sample)])
    cs2 = np.concatenate([[0.0], np.cumsum(sample ** 2)])

    def ssd(i_arr, j_arr):
        s = cs[j_arr + 1] - cs[i_arr]
        s2 = cs2[j_arr + 1] - cs2[i_arr]
        n_ij = (j_arr - i_arr + 1).astype(np.float64)
        return s2 - s * s / n_ij

    INF = 1e18
    dp = np.full((n_classes + 1, n), INF)
    bt = np.zeros((n_classes + 1, n), dtype=int)
    js = np.arange(n, dtype=np.int64)
    dp[1] = ssd(np.zeros(n, dtype=np.int64), js)
    for k in range(2, n_classes + 1):
        for j in range(k - 1, n):
            i_arr = np.arange(k - 2, j, dtype=np.int64)
            vals = dp[k - 1, i_arr] + ssd(i_arr + 1,
                                           np.full_like(i_arr, j))
            bi = int(np.argmin(vals))
            dp[k, j] = vals[bi]
            bt[k, j] = i_arr[bi] + 1
    breaks, j = [], n - 1
    for k in range(n_classes, 1, -1):
        i = bt[k][j]
        breaks.append(sample[i])
        j = i - 1
    return sorted(breaks)


def classify_with_fixed_breaks(csi_map, breaks):
    valid = ~np.isnan(csi_map)
    cls = np.zeros_like(csi_map, dtype=np.int16)
    cls[valid] = 1
    for i, b in enumerate(breaks):
        cls[valid & (csi_map > b)] = i + 2
    return cls


def balanced_breaks(values, n_classes=3, min_frac=0.10, verbose=True):
    valid = ~np.isnan(values)
    n = int(valid.sum())
    if n == 0:
        raise ValueError("待分类数据全为NaN")

    jenks = jenks_breaks_n(values, n_classes=n_classes)
    cls = classify_with_fixed_breaks(values, jenks)
    fracs = [(cls == k).sum() / n for k in range(1, n_classes + 1)]

    if verbose:
        print(f"  [断点] Jenks 分布: {[f'{f:.2%}' for f in fracs]}")
        if min(fracs) < min_frac:
            print(f"  [提示] Jenks 最小类占比 {min(fracs):.2%} < "
                  f"阈值 {min_frac:.0%}, 保留 Jenks 断点 (不回退)")
    return jenks


def compute_stability_metrics(csi_map, base_csi_map, base_breaks, mask):
    valid = (~np.isnan(csi_map)) & (~np.isnan(base_csi_map)) & mask
    cls = classify_with_fixed_breaks(csi_map, base_breaks)
    base_cls = classify_with_fixed_breaks(base_csi_map, base_breaks)
    n_valid = int(valid.sum())
    hc = (cls == 3) & valid
    hb = (base_cls == 3) & valid
    inter = int((hc & hb).sum())
    union = int((hc | hb).sum())
    return {"high_area_pct": round(float(hc.sum() / max(n_valid, 1)), 4),
            "high_IoU": round(inter / max(union, 1), 4),
            "transfer_matrix": [[int(((base_cls == i+1) & (cls == j+1)
                                       & valid).sum())
                                  for j in range(3)] for i in range(3)]}


# ============================================================
# 7b. 空间统计
# ============================================================
def _moran_I(values, mask, max_n=20000):
    valid = mask & ~np.isnan(values)
    n_valid = int(valid.sum())
    if n_valid < 50:
        return np.nan
    step = max(1, int(np.sqrt(n_valid / max_n)))
    sub = values[::step, ::step]
    sub_mask = valid[::step, ::step]
    vals = sub[sub_mask]
    n = len(vals)
    if n < 50:
        return np.nan
    z = vals - vals.mean()
    idx_map = np.full(sub.shape, -1, dtype=np.int32)
    idx_map[sub_mask] = np.arange(n)
    W_sum = Wz_sum = 0.0
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        a0, a1 = max(0, -di), sub.shape[0] - max(0, di)
        b0, b1 = max(0, -dj), sub.shape[1] - max(0, dj)
        c0, c1 = max(0, di), sub.shape[0] - max(0, -di)
        d0, d1 = max(0, dj), sub.shape[1] - max(0, -dj)
        s_self = idx_map[a0:a1, b0:b1]
        s_neigh = idx_map[c0:c1, d0:d1]
        vp = (s_self >= 0) & (s_neigh >= 0)
        W_sum += int(vp.sum())
        Wz_sum += float((z[s_self[vp]] * z[s_neigh[vp]]).sum())
    if W_sum == 0:
        return np.nan
    denom = float((z ** 2).sum())
    return np.nan if denom == 0 else float((n / W_sum) * (Wz_sum / denom))


def _spatial_fragmentation(csi_class, mask, target_class=3):
    target = (csi_class == target_class) & mask
    if target.sum() == 0:
        return 0, 0.0
    _, n_patch = nd_label(target)
    boundary = target & ~binary_erosion(target)
    return int(n_patch), float(boundary.sum() / max(target.sum(), 1))


# ============================================================
# 7c. 单次运行 + 聚合
# ============================================================
def _run_one_config(patches, s_prior, mask_hard, coords, patch_size,
                    H, W, cae_state, n_clusters, gamma, lam, seed,
                    base_csi_map, base_breaks, finetune_epochs, embedding_dim,
                    batch_size=BATCH_SIZE, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    loader = _make_loader_with_prior(patches, s_prior,
                                     batch_size=batch_size, shuffle=False)
    model = ConstrainedCAEDEC(
        n_factors=patches.shape[1], n_clusters=n_clusters,
        embedding_dim=embedding_dim, patch_size=patch_size).to(DEVICE)
    model.cae.load_state_dict(cae_state)

    finetune_constrained_dec(model, loader, epochs=finetune_epochs,
                             gamma=gamma, lam=lam, verbose=verbose,
                             seed=seed)

    mu_k = model.mu_k.detach().cpu().numpy()
    csi_map, _, q_all = compute_csi_and_map(
        model, patches, coords, mask_hard, patch_size, H, W, n_clusters)

    model.eval()
    with torch.no_grad():
        recon_loss = 0.0
        for batch in loader:
            xb = batch[0].to(DEVICE, non_blocking=True)
            xr, _ = model.cae(xb)
            recon_loss += F.mse_loss(xr, xb).item() * len(xb)
        recon_loss /= len(loader.dataset)
        z_all = model.cae.encode(
            torch.tensor(patches, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()
    labels = q_all.argmax(1)
    if len(np.unique(labels)) > 1:
        ch = float(calinski_harabasz_score(z_all, labels))
        sil = float(silhouette_score(z_all, labels))
    else:
        ch, sil = 0.0, 0.0

    stab = compute_stability_metrics(csi_map, base_csi_map,
                                     base_breaks, mask_hard)
    cls = classify_with_fixed_breaks(csi_map, base_breaks)
    n_patch, bd = _spatial_fragmentation(cls, mask_hard, 3)
    moran = _moran_I(csi_map, mask_hard)

    return {"mu_k": mu_k, "mu_k_range": float(mu_k.max() - mu_k.min()),
            "recon_loss": recon_loss, "CH": ch, "Silhouette": sil,
            "high_area_pct": stab["high_area_pct"],
            "high_IoU": stab["high_IoU"],
            "n_patch": n_patch, "boundary_density": bd,
            "moran_I": moran,
            "csi_map": csi_map, "labels": labels, "q_all": q_all}


def _aggregate(runs):
    keys = ["mu_k_range", "recon_loss", "CH", "Silhouette",
            "high_area_pct", "high_IoU", "n_patch",
            "boundary_density", "moran_I"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_std"] = float(np.nanstd(vals))
    mus = np.stack([r["mu_k"] for r in runs], 0)
    out["mu_k_mean"] = mus.mean(0).tolist()
    out["mu_k_std"] = mus.std(0).tolist()
    return out


# ============================================================
# 8. 聚类消融对比 (PCA-DEC / DEC / CAE-DEC)
# ============================================================
def _clustering_metrics(z, labels):
    if z is None or len(z) < 10 or len(np.unique(labels)) < 2:
        return "N/A", "N/A", "N/A"
    try:
        sil = round(float(silhouette_score(z, labels)), 4)
        ch = round(float(calinski_harabasz_score(z, labels)), 2)
        dbi = round(float(davies_bouldin_score(z, labels)), 4)
        return sil, ch, dbi
    except Exception:
        return "N/A", "N/A", "N/A"


def _pca_dec_embedding(patches, s_prior, n_clusters, embedding_dim,
                       epochs, gamma, lam, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    N = len(patches)
    X_flat = patches.reshape(N, -1).astype(np.float32)

    n_comp = min(embedding_dim, X_flat.shape[1], N - 1)
    pca = PCA(n_components=n_comp, random_state=seed)
    z_pca = pca.fit_transform(X_flat).astype(np.float32)

    z_t = torch.tensor(z_pca, dtype=torch.float32).to(DEVICE)
    s_t = torch.tensor(s_prior, dtype=torch.float32).to(DEVICE)

    proj = nn.Sequential(
        nn.Linear(n_comp, embedding_dim),
        nn.BatchNorm1d(embedding_dim), nn.ReLU(),
        nn.Linear(embedding_dim, embedding_dim),
    ).to(DEVICE)
    centroids = nn.Parameter(
        torch.randn(n_clusters, embedding_dim).to(DEVICE))
    mu_k = nn.Parameter(
        torch.linspace(0.1, 0.9, n_clusters).to(DEVICE))

    proj.eval()
    with torch.no_grad():
        z_emb_init = proj(z_t).cpu().numpy()
    proj.train()

    km = KMeans(n_clusters=n_clusters, n_init=N_INIT_KMEANS,
                random_state=seed)
    km.fit(z_emb_init)
    centroids.data = torch.tensor(
        km.cluster_centers_, dtype=torch.float32).to(DEVICE)

    labels_init = km.labels_
    cpm = np.array([s_prior[labels_init == c].mean()
                    if (labels_init == c).sum() > 0 else 0.5
                    for c in range(n_clusters)])
    order = np.argsort(cpm)
    mu_init = np.zeros(n_clusters)
    for rank, idx in enumerate(order):
        mu_init[idx] = 0.1 + rank * 0.8 / max(n_clusters - 1, 1)
    mu_k.data = torch.tensor(mu_init, dtype=torch.float32).to(DEVICE)

    opt = torch.optim.Adam(
        list(proj.parameters()) + [centroids, mu_k], lr=1e-4)

    def soft_assign(z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - centroids) ** 2,
                                    dim=2))
        return q / (q.sum(dim=1, keepdim=True) + 1e-8)

    def target_dist(q):
        w = q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-8)
        return w / (w.sum(dim=1, keepdim=True) + 1e-8)

    proj.train()
    for ep in range(epochs):
        opt.zero_grad()
        z_emb = proj(z_t)
        q = soft_assign(z_emb)
        p = target_dist(q).detach()
        L_KL = F.kl_div(q.log(), p, reduction="batchmean")
        s_hat = (q * mu_k.unsqueeze(0)).sum(dim=1)
        L_prior = F.mse_loss(s_hat, s_t)
        loss = gamma * L_KL + lam * L_prior
        loss.backward()
        opt.step()

    proj.eval()
    with torch.no_grad():
        z_emb = proj(z_t).cpu().numpy()
    return z_emb, None


def _dec_embedding(patches, s_prior, n_clusters, embedding_dim,
                   epochs, gamma, lam, seed,
                   pretrain_epochs=120, pretrain_lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    input_shape = patches.shape[1:]
    model = FlatConstrainedDEC(
        input_shape, n_clusters=n_clusters,
        embedding_dim=embedding_dim).to(DEVICE)

    loader_pre = _make_loader_with_prior(
        patches, s_prior, batch_size=BATCH_SIZE, shuffle=True)
    _ = pretrain_cae_only(model, loader_pre,
                          epochs=pretrain_epochs, lr=pretrain_lr,
                          verbose=False,
                          patience=EARLY_STOP_PATIENCE)

    loader_ft = _make_loader_with_prior(
        patches, s_prior, batch_size=BATCH_SIZE, shuffle=False)
    finetune_constrained_dec(model, loader_ft, epochs=epochs,
                             gamma=gamma, lam=lam,
                             verbose=False, seed=seed)

    model.eval()
    with torch.no_grad():
        X = torch.tensor(patches, dtype=torch.float32).to(DEVICE)
        z_emb = model.cae.encode(X).cpu().numpy()
    return z_emb, None


def clustering_ablation_comparison(patches, s_prior, cae_state,
                                   finetuned_model,
                                   save_dir, n_clusters=4, embedding_dim=32,
                                   epochs=120, gamma=1.0, lam=0.01,
                                   seed=RANDOM_STATE):
    print("\n--- 聚类消融对比 (Silhouette↑ / CH↑ / DBI↓) ---")
    print("    评估口径: 各方法 embedding 上运行 K-means, 统一硬标签")
    print("    对比口径: 三者均含端到端 DEC 微调 (编码器结构为唯一变量)")
    rows = []

    def _eval_with_kmeans(name, z, km_seed):
        km = KMeans(n_clusters=n_clusters, n_init=N_INIT_KMEANS,
                    random_state=km_seed)
        labels = km.fit_predict(z)
        sil, ch, dbi = _clustering_metrics(z, labels)
        rows.append({"Method": name,
                     "Silhouette": sil, "CH": ch, "DBI": dbi})
        print(f"  {name:18s} Silhouette={sil}, CH={ch}, DBI={dbi}")

    print("\n[1/3] PCA-DEC (线性降维 + DEC 微调) ...")
    t0 = time.time()
    z_pca, _ = _pca_dec_embedding(
        patches, s_prior, n_clusters, embedding_dim,
        epochs, gamma, lam, seed)
    _eval_with_kmeans("PCA-DEC", z_pca, seed)
    print(f"  (耗时 {time.time()-t0:.0f}s)")

    print("\n[2/3] DEC (平坦 MLP AE + DEC 微调) ...")
    t0 = time.time()
    z_dec, _ = _dec_embedding(
        patches, s_prior, n_clusters, embedding_dim,
        epochs, gamma, lam, seed)
    _eval_with_kmeans("DEC", z_dec, seed)
    print(f"  (耗时 {time.time()-t0:.0f}s)")

    print("\n[3/3] CAE-DEC (Ours, 卷积 + DEC 微调 + 先验约束) ...")
    finetuned_model.eval()
    with torch.no_grad():
        z_dec_full = finetuned_model.cae.encode(
            torch.tensor(patches, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()
    _eval_with_kmeans("CAE-DEC (Ours)", z_dec_full, seed)

    col_order = ["Method", "Silhouette", "CH", "DBI"]
    df = pd.DataFrame(rows)[col_order]
    out = os.path.join(save_dir, "clustering_ablation.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n聚类消融对比已保存: {out}")
    print(df.to_string(index=False))
    return df


def lambda_sensitivity_analysis(patches, s_prior, mask_hard, coords,
                                patch_size, base_csi_map, base_breaks,
                                cae_state, save_dir, lambda_list,
                                finetune_epochs=120, gamma=1.0,
                                n_clusters=4, embedding_dim=32,
                                n_seeds=3):
    rows, per_lambda_runs = [], {}
    total_models = len(lambda_list) * n_seeds
    model_i = 0
    t_all = time.time()

    for lam in lambda_list:
        print(f"\n===== λ = {lam} ({n_seeds} seeds) =====")
        runs = []
        for s in range(n_seeds):
            model_i += 1
            seed = RANDOM_STATE + s * 100
            t0 = time.time()
            r = _run_one_config(
                patches, s_prior, mask_hard, coords, patch_size,
                base_csi_map.shape[0], base_csi_map.shape[1],
                cae_state, n_clusters, gamma, lam, seed,
                base_csi_map, base_breaks, finetune_epochs, embedding_dim)
            runs.append(r)
            print(f"  [{model_i}/{total_models}] seed {s}: "
                  f"μ_rng={r['mu_k_range']:.3f}, CH={r['CH']:.1f}, "
                  f"Sil={r['Silhouette']:.3f}, IoU={r['high_IoU']:.3f}, "
                  f"Moran={r['moran_I']:.3f}  ({time.time()-t0:.0f}s)")
        per_lambda_runs[lam] = runs
        agg = _aggregate(runs)
        rows.append({
            "lambda": lam,
            "mu_k_mean": np.round(agg["mu_k_mean"], 4).tolist(),
            "mu_k_std": np.round(agg["mu_k_std"], 4).tolist(),
            "mu_k_range_mean": round(agg["mu_k_range_mean"], 4),
            "mu_k_range_std": round(agg["mu_k_range_std"], 4),
            "recon_loss_mean": round(agg["recon_loss_mean"], 6),
            "recon_loss_std": round(agg["recon_loss_std"], 6),
            "CH_mean": round(agg["CH_mean"], 2),
            "CH_std": round(agg["CH_std"], 2),
            "Sil_mean": round(agg["Silhouette_mean"], 4),
            "Sil_std": round(agg["Silhouette_std"], 4),
            "high_area_mean": round(agg["high_area_pct_mean"], 4),
            "high_area_std": round(agg["high_area_pct_std"], 4),
            "high_IoU_mean": round(agg["high_IoU_mean"], 4),
            "high_IoU_std": round(agg["high_IoU_std"], 4),
            "n_patch_mean": round(agg["n_patch_mean"], 1),
            "n_patch_std": round(agg["n_patch_std"], 1),
            "boundary_density_mean": round(agg["boundary_density_mean"], 4),
            "moran_I_mean": round(agg["moran_I_mean"], 4),
            "moran_I_std": round(agg["moran_I_std"], 4),
        })

    df = pd.DataFrame(rows)

    def _z(v):
        v = np.asarray(v, dtype=float)
        return (v - v.mean()) / (v.std() + 1e-9)
    score = (_z(df["Sil_mean"]) + _z(df["high_IoU_mean"])
             + _z(df["moran_I_mean"]) - _z(df["n_patch_mean"])
             + _z(df["CH_mean"]))
    df["composite_score"] = score.round(4)
    best_idx = int(np.argmax(score))
    lam_star = float(df.loc[best_idx, "lambda"])
    print(f"\n[Pareto] 综合最优 λ* = {lam_star} "
          f"(score = {float(score[best_idx]):.3f})")
    print(f"[λ 敏感性] 总耗时 {time.time()-t_all:.0f}s")

    if n_seeds >= 6:
        runs_star = per_lambda_runs[lam_star]
        signif = []
        for lam in lambda_list:
            if lam == lam_star:
                signif.append({"lambda": lam, "p_Sil": np.nan,
                               "p_IoU": np.nan, "p_Moran": np.nan})
                continue
            rec = {"lambda": lam}
            for key, name in [("Silhouette", "p_Sil"),
                              ("high_IoU", "p_IoU"),
                              ("moran_I", "p_Moran")]:
                a = np.nan_to_num(np.array([r[key] for r in runs_star]))
                b = np.nan_to_num(np.array([r[key]
                                             for r in per_lambda_runs[lam]]))
                try:
                    p = 1.0 if np.allclose(a, b) else wilcoxon(a, b)[1]
                except Exception:
                    p = np.nan
                rec[name] = round(float(p), 4) if not np.isnan(p) else np.nan
            signif.append(rec)
        df = df.merge(pd.DataFrame(signif), on="lambda", how="left")
    else:
        df["p_Sil"] = df["p_IoU"] = df["p_Moran"] = np.nan
        print(f"  [提示] N_seeds = {n_seeds} < 6, 跳过 Wilcoxon 显著性检验")

    out = os.path.join(save_dir, "lambda_sensitivity.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"λ 敏感性结果已保存: {out}")
    return df, per_lambda_runs, lam_star


def hyperparam_sensitivity(patches, s_prior, mask_hard, coords,
                           patch_size, base_csi_map, base_breaks,
                           cae_state, save_dir, n_clusters_list,
                           gamma_list, lam=0.01, finetune_epochs=120,
                           embedding_dim=32, n_seeds=3):
    H, W = base_csi_map.shape
    rows = []
    total = len(n_clusters_list) * len(gamma_list) * n_seeds
    model_i = 0
    t_all = time.time()

    for nc in n_clusters_list:
        for gam in gamma_list:
            print(f"\n=== K={nc}, γ={gam} ({n_seeds} seeds) ===")
            chs, sils, ious, morans = [], [], [], []
            for s in range(n_seeds):
                model_i += 1
                t0 = time.time()
                r = _run_one_config(
                    patches, s_prior, mask_hard, coords, patch_size,
                    H, W, cae_state, nc, gam, lam,
                    RANDOM_STATE + s * 100,
                    base_csi_map, base_breaks, finetune_epochs, embedding_dim)
                chs.append(r["CH"]); sils.append(r["Silhouette"])
                ious.append(r["high_IoU"]); morans.append(r["moran_I"])
                print(f"  [{model_i}/{total}] K={nc} γ={gam} s={s}: "
                      f"CH={r['CH']:.1f}, Sil={r['Silhouette']:.3f}, "
                      f"IoU={r['high_IoU']:.3f}  ({time.time()-t0:.0f}s)")
            rows.append({
                "n_clusters": nc, "gamma": gam,
                "CH_mean": round(float(np.mean(chs)), 2),
                "CH_std": round(float(np.std(chs)), 2),
                "Sil_mean": round(float(np.mean(sils)), 4),
                "Sil_std": round(float(np.std(sils)), 4),
                "IoU_mean": round(float(np.mean(ious)), 4),
                "IoU_std": round(float(np.std(ious)), 4),
                "Moran_mean": round(float(np.nanmean(morans)), 4),
                "Moran_std": round(float(np.nanstd(morans)), 4),
            })

    df = pd.DataFrame(rows)
    out = os.path.join(save_dir, "hyperparam_sensitivity.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n超参敏感性结果已保存: {out}")
    print(f"[K×γ 敏感性] 总耗时 {time.time()-t_all:.0f}s")
    return df


# ============================================================
# 9. Table 2
# ============================================================
def export_table2_lambda(df_lambda, save_dir):
    if df_lambda is None or len(df_lambda) == 0:
        print("  [跳过] Table 2 (df_lambda 为空)")
        return
    print("\n--- Table 2: λ 敏感性 (CSV) ---")
    best_idx = int(df_lambda["composite_score"].idxmax())

    csv_rows = []
    for i, r in df_lambda.iterrows():
        is_b = (i == best_idx)
        csv_rows.append({
            "lambda": f"{r['lambda']:.2f}",
            "mu_k_range": f"{r['mu_k_range_mean']:.3f} ± {r['mu_k_range_std']:.3f}",
            "CH": f"{r['CH_mean']:.1f} ± {r['CH_std']:.1f}",
            "Silhouette": f"{r['Sil_mean']:.3f} ± {r['Sil_std']:.3f}",
            "IoU": f"{r['high_IoU_mean']:.3f} ± {r['high_IoU_std']:.3f}",
            "Moran_I": f"{r['moran_I_mean']:.3f} ± {r['moran_I_std']:.3f}",
            "Composite": f"{r['composite_score']:.2f}",
            "Is_best": int(is_b),
        })

    out_csv = os.path.join(save_dir, "Table2_lambda.csv")
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False,
                                  encoding="utf-8-sig")
    print(f"  已保存: {out_csv}  (共 {len(csv_rows)} 行)")


# ============================================================
# 10. 主流程
# ============================================================
def main():
    # ---------- 路径解析: 本地优先, 否则回退到相对路径 ----------
    LOCAL_TIF_DIR = r"C:\Users\admin\Desktop\floating-wind-SLS\CAE-DEC\输入"
    LOCAL_SAVE_DIR = r"C:\Users\admin\Desktop\floating-wind-SLS\CAE-DEC\输出"

    if os.path.isdir(LOCAL_TIF_DIR):
        default_tif_dir = LOCAL_TIF_DIR
        default_save_dir = LOCAL_SAVE_DIR
    else:
        default_tif_dir = "data"
        default_save_dir = "outputs"

    parser = argparse.ArgumentParser(
        description="CAE-DEC 漂浮式海上风电选址适宜性评价")
    parser.add_argument("--tif_dir", type=str, default=default_tif_dir,
                        help="输入 TIF 目录")
    parser.add_argument("--save_dir", type=str, default=default_save_dir,
                        help="输出目录")
    parser.add_argument("--sensitivity", type=str, default="balanced",
                        choices=["off", "fast", "balanced", "full"],
                        help="敏感性分析档位 (默认 balanced)")
    args = parser.parse_args()

    t_start = time.time()
    tif_dir = args.tif_dir
    save_dir = ensure_dir(args.save_dir)

    tif_files = ["水深.tif", "离岸距离.tif", "距航道距离.tif",
                 "风速.tif", "风功率密度.tif",
                 "易发性评价结果重分类.tif"]
    tif_paths = [os.path.join(tif_dir, f) for f in tif_files]
    for p in tif_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"文件不存在: {p}")

    TARGET_RESOLUTION = 100
    PATCH_SIZE = 58
    STRIDE = 29
    MIN_VALID_RATIO = 0.4
    N_CLUSTERS = 4
    EMBEDDING_DIM = 32
    EPOCHS_PRETRAIN = 120
    EPOCHS_FINETUNE = 120
    GAMMA = 1.0
    LAMBDA_FINAL = 0.01

    HC_DEPTH_MIN = 50.0
    HC_DEPTH_MAX = 1000.0
    HC_OFFSHORE_KM_MIN = 12.0
    HC_CHANNEL_KM_MIN = 5.0
    HC_WIND_MIN = 6.0
    HC_WPD_MIN = 200.0
    HC_LANDSLIDE_MAX = 3

    SENSITIVITY_LEVEL = args.sensitivity

    if SENSITIVITY_LEVEL == "off":
        RUN_SENSITIVITY = False
        N_SEEDS = 1
        LAMBDA_LIST = ()
        HP_K_LIST = ()
        HP_GAMMA_LIST = ()
    elif SENSITIVITY_LEVEL == "fast":
        RUN_SENSITIVITY = True
        N_SEEDS = 2
        LAMBDA_LIST = (0.0, 0.01, 0.1)
        HP_K_LIST = (4,)
        HP_GAMMA_LIST = (0.5, 1.0)
    elif SENSITIVITY_LEVEL == "balanced":
        RUN_SENSITIVITY = True
        N_SEEDS = 3
        LAMBDA_LIST = (0.0, 0.01, 0.1, 0.5, 1.0)
        HP_K_LIST = (3, 4, 5)
        HP_GAMMA_LIST = (0.5, 1.0)
    else:
        RUN_SENSITIVITY = True
        N_SEEDS = 5
        LAMBDA_LIST = (0.0, 0.001, 0.01, 0.1, 0.5, 1.0)
        HP_K_LIST = (3, 4, 5, 6)
        HP_GAMMA_LIST = (0.1, 0.5, 1.0, 2.0)

    print("=" * 60)
    print(f"  配置 v4.9.2-github (100 m 分辨率 + PATCH_SIZE={PATCH_SIZE})")
    print(f"  硬约束: 滑坡≤{HC_LANDSLIDE_MAX}级, 离岸≥{HC_OFFSHORE_KM_MIN}km, "
          f"水深{HC_DEPTH_MIN:.0f}-{HC_DEPTH_MAX:.0f}m,")
    print(f"          距航道≥{HC_CHANNEL_KM_MIN}km, 风速≥{HC_WIND_MIN}m/s, "
          f"WPD≥{HC_WPD_MIN}W/m²")
    print("=" * 60)
    print(f"  输入目录: {tif_dir}")
    print(f"  输出目录: {save_dir}")
    print(f"  分辨率: {TARGET_RESOLUTION} m")
    print(f"  Patch: {PATCH_SIZE} 像元 (≈ "
          f"{PATCH_SIZE*TARGET_RESOLUTION/1000:.2f} km)")
    print(f"  Stride: {STRIDE}, K={N_CLUSTERS}, λ={LAMBDA_FINAL}, "
          f"γ={GAMMA}")
    print(f"  Embedding: {EMBEDDING_DIM}, Batch: {BATCH_SIZE}")
    print(f"  敏感性模式: {SENSITIVITY_LEVEL} (N_SEEDS={N_SEEDS})")
    print("  输出: 6 个文件 (5 CSV + 1 TIF), 无图件, 无 Shapefile")

    print("\n=== 0. 输入分辨率预检 ===")
    max_input_res = check_input_resolutions(tif_paths)
    if max_input_res is not None and max_input_res > TARGET_RESOLUTION * 3:
        print(f"  [警告] 输入最大分辨率 {max_input_res:.0f} m "
              f"远大于目标 {TARGET_RESOLUTION} m")

    print("\n=== 1. TIF对齐 ===")
    aligned, profile = align_rasters(tif_paths,
                                     target_resolution=TARGET_RESOLUTION,
                                     target_crs="EPSG:32633")

    print("\n=== 2. 滑坡易发性验证 ===")
    landslide_raw = aligned[5]
    valid_mask = ~np.isnan(landslide_raw)
    if valid_mask.sum() == 0:
        raise ValueError("易发性栅格全为NaN")
    landslide_class = np.full_like(landslide_raw, np.nan, dtype=np.float32)
    _lr = landslide_raw[valid_mask]
    landslide_class[valid_mask] = np.clip(
        np.floor(_lr + 0.5), 1, 5).astype(np.float32)
    uniq = np.unique(landslide_class[valid_mask]).astype(int)
    print(f"  出现级别: {uniq.tolist()}")
    if not set(uniq.tolist()).issubset({1, 2, 3, 4, 5}):
        raise ValueError("应仅含1~5整数级别")
    for c in range(1, 6):
        cnt = int((landslide_class == c).sum())
        print(f"  级别{c}: {cnt:,} ({cnt/max(valid_mask.sum(),1):.2%})")
    aligned[5] = landslide_class

    print("\n=== 3. 硬约束构建 ===")
    mask_hard, soft, landslide_score, depth_abs, offshore_km, channel_km = \
        build_constraints(
            aligned, landslide_class,
            depth_min=HC_DEPTH_MIN, depth_max=HC_DEPTH_MAX,
            offshore_min_km=HC_OFFSHORE_KM_MIN,
            channel_min_km=HC_CHANNEL_KM_MIN,
            wind_min=HC_WIND_MIN, wpd_min=HC_WPD_MIN,
            landslide_max=HC_LANDSLIDE_MAX)

    print("\n=== 4. Patch提取 (均值填充 + 返回 masks) ===")
    patches, patch_masks, coords = extract_patches(
        soft, mask_hard,
        patch_size=PATCH_SIZE, stride=STRIDE,
        min_valid_ratio=MIN_VALID_RATIO,
        max_patches=30000)
    n_factors = patches.shape[1]
    H_full, W_full = soft.shape[1], soft.shape[2]

    print("\n=== 5. 熵权法 (基于 mask 排除填充区) ===")
    patch_means = _patch_masked_means(patches, patch_masks)
    entropy_weights = compute_entropy_weights(patch_means)
    _print_names = ["Water Depth", "Distance to Shoreline",
                    "Distance to Shipping Lane", "Wind Speed",
                    "Wind Power Density", "SLS Score"]
    for nm, w in zip(_print_names, entropy_weights):
        print(f"    {nm}: {w:.4f}")
    s_prior = compute_prior_scores(patch_means, entropy_weights)
    print(f"  s_prior: {s_prior.min():.4f} ~ {s_prior.max():.4f}")

    pd.DataFrame({"factor": _print_names, "weight": entropy_weights}
                 ).to_csv(os.path.join(save_dir, "entropy_weights.csv"),
                          index=False, encoding="utf-8-sig")

    print("\n=== 5b. 共享 CAE 预训练 (对称架构) ===")
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    pretrain_model = ConstrainedCAEDEC(
        n_factors=n_factors, n_clusters=N_CLUSTERS,
        embedding_dim=EMBEDDING_DIM, patch_size=PATCH_SIZE).to(DEVICE)
    pretrain_loader = _make_loader_with_prior(patches, s_prior,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)
    pretrain_cae_only(pretrain_model, pretrain_loader,
                      epochs=EPOCHS_PRETRAIN, verbose=True)
    CAE_STATE = {k: v.detach().clone().cpu()
                 for k, v in pretrain_model.cae.state_dict().items()}
    print(f"  CAE 权重已缓存，累计 {time.time()-t_start:.0f}s")

    print(f"\n=== 5c. 基线模型训练 (微调 CAE-DEC, lr={FINETUNE_LR}) ===")
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    baseline_model = ConstrainedCAEDEC(
        n_factors=n_factors, n_clusters=N_CLUSTERS,
        embedding_dim=EMBEDDING_DIM, patch_size=PATCH_SIZE).to(DEVICE)
    baseline_model.cae.load_state_dict(CAE_STATE)
    baseline_loader = _make_loader_with_prior(patches, s_prior,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)
    finetune_constrained_dec(baseline_model, baseline_loader,
                             epochs=EPOCHS_FINETUNE, gamma=GAMMA,
                             lam=LAMBDA_FINAL, verbose=True,
                             seed=RANDOM_STATE)
    final_mu = baseline_model.mu_k.detach().cpu().numpy()
    print(f"\n  基线 μ_k = {np.round(final_mu, 4)}, "
          f"极差 = {final_mu.max() - final_mu.min():.4f}")
    baseline_csi, _, baseline_q = compute_csi_and_map(
        baseline_model, patches, coords, mask_hard, PATCH_SIZE,
        H_full, W_full, N_CLUSTERS)
    print(f"  基线 CSI: {np.nanmin(baseline_csi):.4f} ~ "
          f"{np.nanmax(baseline_csi):.4f}, "
          f"均值={np.nanmean(baseline_csi):.4f}")

    baseline_breaks = balanced_breaks(baseline_csi, n_classes=3,
                                      min_frac=0.10, verbose=True)
    print(f"  基线断点: {[f'{b:.6f}' for b in baseline_breaks]}")

    print("\n=== 5e. 聚类消融对比 (PCA-DEC / DEC / CAE-DEC) ===")
    try:
        df_ablation = clustering_ablation_comparison(
            patches, s_prior, CAE_STATE,
            baseline_model,
            save_dir, n_clusters=N_CLUSTERS,
            embedding_dim=EMBEDDING_DIM,
            epochs=EPOCHS_FINETUNE,
            gamma=GAMMA,
            lam=LAMBDA_FINAL,
            seed=RANDOM_STATE)
    except Exception as e:
        print(f"  聚类消融对比失败: {e}")
        import traceback
        traceback.print_exc()

    LAMBDA_STAR = LAMBDA_FINAL
    df_lambda = None
    if RUN_SENSITIVITY:
        print("\n=== 6a. λ 敏感性分析 (→ Table 2 CSV) ===")
        df_lambda, runs_lambda, LAMBDA_STAR = lambda_sensitivity_analysis(
            patches, s_prior, mask_hard, coords, PATCH_SIZE,
            baseline_csi, baseline_breaks, CAE_STATE, save_dir,
            lambda_list=LAMBDA_LIST, finetune_epochs=EPOCHS_FINETUNE,
            gamma=GAMMA, n_clusters=N_CLUSTERS,
            embedding_dim=EMBEDDING_DIM, n_seeds=N_SEEDS)
        print(f"\n  >>> Pareto 最优 λ* = {LAMBDA_STAR}")

        print("\n=== 6b. K×γ 敏感性分析 ===")
        df_hp = hyperparam_sensitivity(
            patches, s_prior, mask_hard, coords, PATCH_SIZE,
            baseline_csi, baseline_breaks, CAE_STATE, save_dir,
            n_clusters_list=HP_K_LIST, gamma_list=HP_GAMMA_LIST,
            lam=LAMBDA_FINAL, finetune_epochs=EPOCHS_FINETUNE,
            embedding_dim=EMBEDDING_DIM, n_seeds=N_SEEDS)

        print("\n=== 6c. 导出期刊表格 (CSV) ===")
        export_table2_lambda(df_lambda, save_dir)
    else:
        print("\n=== 6. 跳过敏感性分析 ===")

    print("\n=== 7. 输出连续 CSI (TIF) ===")
    csi_path = os.path.join(save_dir, "CSI_continuous.tif")
    save_tif(baseline_csi.astype("float32"), profile, csi_path,
             dtype="float32", nodata=NODATA_FLOAT)

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("处理完成！")
    print("  entropy_weights.csv")
    print("  clustering_ablation.csv")
    print("  lambda_sensitivity.csv")
    print("  hyperparam_sensitivity.csv")
    print("  Table2_lambda.csv")
    print("  CSI_continuous.tif")
    print(f"输出目录: {save_dir}")
    print(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()