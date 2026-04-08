"""
lrp_resnet50.py  —  Layer-wise Relevance Propagation for ResNet50
=================================================================
适配 noneed_doc/resnet/resnet_model_org.py 定义的 TF1 ResNet50 模型。

功能:
  - 加载 TF1 checkpoint，将 BN 参数吸收进 Conv 权重
  - 对每个 bottleneck block 用 比例分配法 (ratio-based split) 处理跳跃连接
  - 生成显示图片哪些区域对预测某个 attribute 有贡献的热力图

LRP 规则:
  - 默认: epsilon-rule (所有卷积/全连接层)
  - 第一层可选: z+-rule (更清晰的像素级热力图)
  - 跳跃连接: 按激活幅值比例分配相关性 (来自 https://5ei74r0.github.io/lrp-for-resnet.page/)

使用方法 (命令行):
    # 单张图片
    python lrp_resnet50.py --checkpoint_dir ../save/resnet \\
                           --image_path image.png \\
                           --class_idx 0

    # 批量处理文件夹
    python lrp_resnet50.py --checkpoint_dir ../save/resnet \\
                           --image_folder static/images/vis_filtered_thumbnails \\
                           --output_folder lrp_output \\
                           --class_idx 3

    # 查看所有 attribute 名称
    python lrp_resnet50.py --checkpoint_dir ../save/resnet --list_classes

依赖: tensorflow>=2.0, numpy, matplotlib, Pillow, scipy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import tensorflow as tf

# ────────────────────────────────────────────────────────────────────────────
# 常量 (与 resnet_model_org.py 保持一致)
# ────────────────────────────────────────────────────────────────────────────
BN_EPSILON   = 0.001
NUM_BLOCKS   = [3, 4, 6, 3]   # ResNet-50
# scale2 的第一个 block stride=1 (不同于其他 scale)
SCALE_STRIDES = [1, 2, 2, 2]


# ────────────────────────────────────────────────────────────────────────────
# Checkpoint 加载与 BN 吸收
# ────────────────────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_dir: str) -> dict[str, np.ndarray]:
    """从 TF1 checkpoint 目录读取所有变量，返回 {name: ndarray}。"""
    ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"在 '{checkpoint_dir}' 中找不到 checkpoint。\n"
            f"请确认目录下有 checkpoint、.index、.data 文件。"
        )
    reader = tf.train.load_checkpoint(ckpt_path)
    var_map = {}
    for name in reader.get_variable_to_shape_map():
        clean = name.replace(':0', '')
        var_map[clean] = reader.get_tensor(name)
    print(f"[✓] 从 '{ckpt_path}' 载入 {len(var_map)} 个变量")
    return var_map


def absorb_bn(
    conv_w: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    eps: float = BN_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将 BatchNorm 参数吸收进 Conv 权重，使 BN+Conv 等价于一个带 bias 的 Conv。

    conv_w: [kH, kW, C_in, C_out]
    gamma/beta/mean/var: [C_out]

    返回: (w_new [kH, kW, C_in, C_out], b_new [C_out])
    """
    scale = gamma / np.sqrt(var + eps)                        # [C_out]
    w_new = conv_w * scale[np.newaxis, np.newaxis, np.newaxis, :]
    b_new = beta - mean * scale
    return w_new.astype(np.float32), b_new.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# numpy / TF 混合的前向运算（用 TF 加速卷积，保持 numpy 接口）
# ────────────────────────────────────────────────────────────────────────────

def _conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray,
            stride: int = 1, padding: str = 'SAME') -> np.ndarray:
    """单张图片的 conv2d：x [H,W,Cin], w [kH,kW,Cin,Cout] → [H',W',Cout]"""
    x_t = tf.constant(x[np.newaxis], dtype=tf.float32)
    w_t = tf.constant(w, dtype=tf.float32)
    b_t = tf.constant(b, dtype=tf.float32)
    out = tf.nn.conv2d(x_t, w_t, strides=[1, stride, stride, 1],
                       padding=padding) + b_t
    return out.numpy()[0]


def _conv2d_transpose(x: np.ndarray, w: np.ndarray,
                      stride: int = 1, padding: str = 'SAME',
                      output_shape: list | None = None) -> np.ndarray:
    """转置卷积（LRP 反向传播用）：x [H',W',Cout] → [H,W,Cin]"""
    x_t = tf.constant(x[np.newaxis], dtype=tf.float32)
    w_t = tf.constant(w, dtype=tf.float32)
    if output_shape is None:
        h_in, w_in, c_out = x.shape
        c_in = w.shape[2]
        kh, kw_k = w.shape[:2]
        if padding == 'SAME':
            h_out, w_out = h_in * stride, w_in * stride
        else:
            h_out = (h_in - 1) * stride + kh
            w_out = (w_in - 1) * stride + kw_k
        out_shape = [1, h_out, w_out, c_in]
    else:
        out_shape = [1] + list(output_shape)
    out = tf.nn.conv2d_transpose(x_t, w_t,
                                  output_shape=out_shape,
                                  strides=[1, stride, stride, 1],
                                  padding=padding)
    return out.numpy()[0]


def _max_pool(x: np.ndarray, ksize: int = 3, stride: int = 2,
              padding: str = 'SAME') -> np.ndarray:
    x_t = tf.constant(x[np.newaxis], dtype=tf.float32)
    out = tf.nn.max_pool(x_t, ksize=[1, ksize, ksize, 1],
                         strides=[1, stride, stride, 1], padding=padding)
    return out.numpy()[0]


# ────────────────────────────────────────────────────────────────────────────
# LRP 规则
# ────────────────────────────────────────────────────────────────────────────

def _eps_stable(z: np.ndarray, eps: float) -> np.ndarray:
    """给 z 加上 eps*sign(z) 并避免接近 0 时除法不稳定。"""
    sign = np.where(z >= 0, 1.0, -1.0)
    z_eps = z + eps * sign
    # 若仍接近 0，强制为 eps
    z_eps = np.where(np.abs(z_eps) < 1e-12, eps, z_eps)
    return z_eps


def lrp_conv_epsilon(
    a: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    R: np.ndarray,
    stride: int = 1,
    padding: str = 'SAME',
    eps: float = 1e-6,
) -> np.ndarray:
    """
    epsilon-rule LRP for Conv layer.

    a [H,W,Cin], w [kH,kW,Cin,Cout], b [Cout], R [H',W',Cout]
    → R_lower [H,W,Cin]

    公式: R_i = a_i * sum_j( w_ij / (z_j + eps*sign(z_j)) * R_j )
    """
    z = _conv2d(a, w, b, stride=stride, padding=padding)   # [H',W',Cout]
    z_eps = _eps_stable(z, eps)
    s = (R / z_eps).astype(np.float32)                      # [H',W',Cout]
    c = _conv2d_transpose(s, w, stride=stride, padding=padding,
                          output_shape=list(a.shape))        # [H,W,Cin]
    return a * c


def lrp_conv_zplus(
    a: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    R: np.ndarray,
    stride: int = 1,
    padding: str = 'SAME',
) -> np.ndarray:
    """
    z+-rule LRP：只用正权重，适合第一层（像素级热力图更清晰）。
    """
    w_pos = np.maximum(w, 0.0).astype(np.float32)
    b_pos = np.maximum(b, 0.0).astype(np.float32)
    a_pos = np.maximum(a, 0.0).astype(np.float32)

    z = _conv2d(a_pos, w_pos, b_pos, stride=stride, padding=padding)
    z = np.where(np.abs(z) < 1e-10, 1e-10, z)

    s = (R / z).astype(np.float32)
    c = _conv2d_transpose(s, w_pos, stride=stride, padding=padding,
                          output_shape=list(a.shape))
    return a_pos * c


def lrp_fc_epsilon(
    a: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    R: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    epsilon-rule LRP for fully-connected layer.
    a [Nin], w [Nin,Nout], b [Nout], R [Nout] → R_lower [Nin]
    """
    z = a @ w + b                                   # [Nout]
    z_eps = _eps_stable(z, eps)
    s = R / z_eps                                   # [Nout]
    c = w @ s                                       # [Nin]
    return a * c


def lrp_global_avgpool(
    a: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    LRP for global average pooling.
    a [H,W,C], R [C] → R_lower [H,W,C]

    每个空间位置均等贡献，按激活幅值加权。
    """
    H, W, _ = a.shape
    eps = 1e-10
    z = a.mean(axis=(0, 1))                        # [C]
    z_safe = np.where(np.abs(z) < eps, eps, z)
    scale = R / z_safe                             # [C]
    return a * scale[np.newaxis, np.newaxis, :] / (H * W)


def lrp_maxpool(
    a: np.ndarray,
    R: np.ndarray,
    ksize: int = 3,
    stride: int = 2,
    padding: str = 'SAME',
) -> np.ndarray:
    """
    LRP for max pooling：winner-takes-all，用梯度技巧实现。
    a [H,W,C], R [H',W',C] → R_lower [H,W,C]
    """
    a_t = tf.constant(a[np.newaxis], dtype=tf.float32)
    pool_np = _max_pool(a, ksize=ksize, stride=stride, padding=padding)

    eps = 1e-10
    z_safe = np.where(np.abs(pool_np) < eps, eps, pool_np)
    scale = (R / z_safe).astype(np.float32)          # [H',W',C]

    # 用 GradientTape 获取 max 位置的梯度（等价于 winner indicator）
    a_var = tf.Variable(a_t)
    with tf.GradientTape() as tape:
        p = tf.nn.max_pool(a_var,
                           ksize=[1, ksize, ksize, 1],
                           strides=[1, stride, stride, 1],
                           padding=padding)
        loss = tf.reduce_sum(p * tf.constant(scale[np.newaxis]))
    c = tape.gradient(loss, a_var).numpy()[0]   # [H,W,C]
    return a * c


# ────────────────────────────────────────────────────────────────────────────
# ResNet50 LRP 引擎
# ────────────────────────────────────────────────────────────────────────────

class LRPResNet50:
    """
    对 resnet_model_org.py 定义的 ResNet50 执行 LRP 解释。

    变量命名规则（来自 resnet_model_org.py 的 variable_scope）:
        scale1/weights, scale1/gamma, scale1/beta,
        scale1/moving_mean, scale1/moving_variance
        scale2/block1/a/weights, scale2/block1/a/gamma, ...
        scale2/block1/shortcut/weights, ...  (仅维度变化时存在)
        fc/weights, fc/biases
    """

    def __init__(
        self,
        checkpoint_dir: str,
        num_classes: int = 17,
        num_blocks: list[int] | None = None,
        image_size: int = 224,
    ):
        self.num_classes = num_classes
        self.num_blocks  = num_blocks or NUM_BLOCKS
        self.image_size  = image_size

        raw = load_checkpoint(checkpoint_dir)
        self.ckpt = {k.replace(':0', ''): v for k, v in raw.items()}
        self._build_weight_cache()

    # ── 变量读取辅助 ────────────────────────────────────────────────────────

    def _get(self, name: str) -> np.ndarray:
        if name in self.ckpt:
            return self.ckpt[name]
        # 给出友好错误信息
        sample = list(self.ckpt.keys())[:15]
        raise KeyError(
            f"变量 '{name}' 在 checkpoint 中不存在。\n"
            f"可用变量（前15个）: {sample}\n"
            f"请用 --inspect 参数查看全部变量名。"
        )

    def _conv_bn(self, scope: str) -> tuple[np.ndarray, np.ndarray]:
        """读取 Conv+BN scope，返回 BN 吸收后的 (w, b)。"""
        w     = self._get(f'{scope}/weights')
        gamma = self._get(f'{scope}/gamma')
        beta  = self._get(f'{scope}/beta')
        mean  = self._get(f'{scope}/moving_mean')
        var   = self._get(f'{scope}/moving_variance')
        return absorb_bn(w, gamma, beta, mean, var)

    def _fc(self, scope: str) -> tuple[np.ndarray, np.ndarray]:
        return self._get(f'{scope}/weights'), self._get(f'{scope}/biases')

    # ── 权重缓存（提前加载，避免重复 IO）──────────────────────────────────

    def _build_weight_cache(self):
        cache: dict = {}

        # scale1: 7×7 conv
        cache['scale1'] = self._conv_bn('scale1')

        # scale2–5
        scale_names = ['scale2', 'scale3', 'scale4', 'scale5']
        for scale_name, n_blocks in zip(scale_names, self.num_blocks):
            for blk in range(1, n_blocks + 1):
                pfx = f'{scale_name}/block{blk}'
                cache[f'{pfx}/a'] = self._conv_bn(f'{pfx}/a')
                cache[f'{pfx}/b'] = self._conv_bn(f'{pfx}/b')
                cache[f'{pfx}/c'] = self._conv_bn(f'{pfx}/c')
                # shortcut projection（仅维度变化时存在）
                sc_key = f'{pfx}/shortcut/weights'
                if sc_key in self.ckpt:
                    cache[f'{pfx}/shortcut'] = self._conv_bn(f'{pfx}/shortcut')
                else:
                    cache[f'{pfx}/shortcut'] = None

        cache['fc'] = self._fc('fc')
        self.cache = cache
        print(f"[✓] 权重缓存构建完成（含 BN 吸收）")

    # ── 前向传播（记录中间激活用于 LRP 反向）────────────────────────────

    def forward(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        完整前向传播。
        image: [H, W, C] numpy float32 (已预处理)
        返回: (logits [num_classes], acts dict)
        """
        acts: dict = {}
        x = image.astype(np.float32)
        acts['input'] = x

        # scale1: 7×7 conv, stride=2, ReLU
        w, b = self.cache['scale1']
        x = _conv2d(x, w, b, stride=2, padding='SAME')
        x = np.maximum(x, 0.0)
        acts['scale1_out'] = x

        # maxpool 3×3, stride=2
        acts['maxpool_in'] = x
        x = _max_pool(x, ksize=3, stride=2, padding='SAME')
        acts['maxpool_out'] = x

        # scale2–5
        scale_names = ['scale2', 'scale3', 'scale4', 'scale5']
        for scale_name, n_blocks, first_stride in zip(
                scale_names, self.num_blocks, SCALE_STRIDES):
            for blk in range(1, n_blocks + 1):
                pfx = f'{scale_name}/block{blk}'
                stride = first_stride if blk == 1 else 1
                x, blk_acts = self._block_forward(x, pfx, stride)
                acts[pfx] = blk_acts

        acts['before_avgpool'] = x

        # Global Average Pooling
        x = x.mean(axis=(0, 1))   # [C=2048]
        acts['after_avgpool'] = x

        # FC
        w_fc, b_fc = self.cache['fc']
        logits = x @ w_fc + b_fc
        acts['logits'] = logits

        return logits, acts

    def _block_forward(
        self, x_in: np.ndarray, pfx: str, stride: int
    ) -> tuple[np.ndarray, dict]:
        """单个 bottleneck block 的前向传播，记录所有中间值。"""
        ba: dict = {'in': x_in}

        # a: 1×1 conv, stride=stride, ReLU
        w, b = self.cache[f'{pfx}/a']
        h = _conv2d(x_in, w, b, stride=stride, padding='SAME')
        h = np.maximum(h, 0.0)
        ba['a_in'] = x_in
        ba['a_out'] = h

        # b: 3×3 conv, stride=1, ReLU
        w, b = self.cache[f'{pfx}/b']
        h = _conv2d(h, w, b, stride=1, padding='SAME')
        h = np.maximum(h, 0.0)
        ba['b_in'] = ba['a_out']
        ba['b_out'] = h

        # c: 1×1 conv, stride=1, 无 ReLU（在 add 之后才有 ReLU）
        w, b = self.cache[f'{pfx}/c']
        h = _conv2d(h, w, b, stride=1, padding='SAME')
        ba['c_in'] = ba['b_out']
        ba['c_out'] = h   # main branch output（before add）

        # shortcut
        sc = self.cache[f'{pfx}/shortcut']
        if sc is not None:
            w_sc, b_sc = sc
            shortcut = _conv2d(x_in, w_sc, b_sc, stride=stride, padding='SAME')
        else:
            shortcut = x_in
        ba['shortcut_out'] = shortcut

        # add + ReLU
        ba['main_pre'] = h          # 主路径（before add）
        ba['skip_pre'] = shortcut   # 跳跃路径（before add）
        out = np.maximum(h + shortcut, 0.0)
        ba['out'] = out

        return out, ba

    # ── LRP 反向传播 ────────────────────────────────────────────────────────

    def lrp(
        self,
        image: np.ndarray,
        class_idx: int,
        lrp_rule: str = 'epsilon',
        eps: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        计算 LRP 热力图。

        Args:
            image:     预处理后的输入图片 [H, W, C]
            class_idx: 要解释的 attribute/class 索引
            lrp_rule:  'epsilon'（默认）或 'zplus'（第一层用 z+-rule）
            eps:       epsilon-rule 稳定因子

        Returns:
            heatmap [H, W]  相关性热力图（正值=有贡献）
            logits  [num_classes]
        """
        if not 0 <= class_idx < self.num_classes:
            raise ValueError(
                f"class_idx={class_idx} 超出范围 [0, {self.num_classes})。"
            )

        logits, acts = self.forward(image)

        # 初始化输出层相关性：只保留目标类别的 logit
        R = np.zeros(self.num_classes, dtype=np.float32)
        R[class_idx] = float(logits[class_idx])

        # FC 层
        w_fc, b_fc = self.cache['fc']
        R = lrp_fc_epsilon(acts['after_avgpool'], w_fc, b_fc, R, eps=eps)

        # Global Average Pooling
        R = lrp_global_avgpool(acts['before_avgpool'], R)

        # scale5 → scale2（逆序）
        scale_names = ['scale2', 'scale3', 'scale4', 'scale5']
        for scale_name, n_blocks in reversed(
                list(zip(scale_names, self.num_blocks))):
            for blk in range(n_blocks, 0, -1):
                pfx = f'{scale_name}/block{blk}'
                R = self._block_lrp(R, acts[pfx], pfx, eps=eps)

        # Maxpool
        R = lrp_maxpool(acts['maxpool_in'], R, ksize=3, stride=2, padding='SAME')

        # scale1 conv（可选 z+-rule）
        w1, b1 = self.cache['scale1']
        if lrp_rule == 'zplus':
            R = lrp_conv_zplus(acts['input'], w1, b1, R, stride=2, padding='SAME')
        else:
            R = lrp_conv_epsilon(acts['input'], w1, b1, R,
                                  stride=2, padding='SAME', eps=eps)

        heatmap = R.sum(axis=-1)   # [H, W]
        return heatmap, logits

    def _block_lrp(
        self,
        R_out: np.ndarray,
        ba: dict,
        pfx: str,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """
        单个 bottleneck block 的 LRP 反向传播。

        跳跃连接处使用比例分配法（ratio-based split）:
            R_main_i = R_out_i * |main_i| / (|main_i| + |skip_i| + ε)
            R_skip_i = R_out_i * |skip_i| / (|main_i| + |skip_i| + ε)

        这样保证相关性守恒: R_main + R_skip = R_out
        """
        main = ba['main_pre']    # 主路径输出（before add）
        skip = ba['skip_pre']    # 跳跃路径输出（before add）

        # ── 比例分配 ─────────────────────────────────────────────────────
        denom   = np.abs(main) + np.abs(skip) + 1e-10
        R_main  = R_out * np.abs(main) / denom
        R_skip  = R_out * np.abs(skip) / denom

        # ── 主路径反向 ────────────────────────────────────────────────────
        # c: conv1×1（无 stride）
        w, b = self.cache[f'{pfx}/c']
        R_main = lrp_conv_epsilon(ba['c_in'], w, b, R_main,
                                   stride=1, padding='SAME', eps=eps)

        # b: conv3×3（无 stride）
        w, b = self.cache[f'{pfx}/b']
        R_main = lrp_conv_epsilon(ba['b_in'], w, b, R_main,
                                   stride=1, padding='SAME', eps=eps)

        # a: conv1×1（可能有 stride）
        w, b = self.cache[f'{pfx}/a']
        stride_a = _infer_stride(ba['a_in'], ba['a_out'])
        R_main = lrp_conv_epsilon(ba['a_in'], w, b, R_main,
                                   stride=stride_a, padding='SAME', eps=eps)

        # ── 跳跃路径反向 ──────────────────────────────────────────────────
        sc = self.cache[f'{pfx}/shortcut']
        if sc is not None:
            w_sc, b_sc = sc
            stride_sc = _infer_stride(ba['in'], ba['shortcut_out'])
            R_skip = lrp_conv_epsilon(ba['in'], w_sc, b_sc, R_skip,
                                       stride=stride_sc, padding='SAME', eps=eps)
        # 若是恒等跳跃（无 projection），R_skip 直接传回

        return R_main + R_skip


def _infer_stride(a_in: np.ndarray, a_out: np.ndarray) -> int:
    """根据输入/输出空间尺寸推断 stride。"""
    h_in  = a_in.shape[0]
    h_out = a_out.shape[0]
    if h_in == h_out:
        return 1
    if h_in > h_out:
        return round(h_in / h_out)
    return 1


# ────────────────────────────────────────────────────────────────────────────
# 图像预处理与热力图可视化
# ────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_path: str, image_size: int = 224) -> np.ndarray:
    """
    加载并预处理图片（ImageNet 均值减法，与 ResNet 训练保持一致）。
    返回: [H, W, 3] float32
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((image_size, image_size), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32)
    # ImageNet 像素均值
    mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    return img_np - mean


def save_heatmap(
    heatmap: np.ndarray,
    original_image_path: str,
    output_path: str,
    class_name: str = '',
    alpha: float = 0.5,
):
    """
    将 LRP 热力图叠加到原图上并保存三联图（原图 | 热力图 | 叠加图）。
    """
    # 原图
    img_orig = Image.open(original_image_path).convert('RGB')
    img_orig = img_orig.resize((heatmap.shape[1], heatmap.shape[0]), Image.BILINEAR)
    img_np = np.array(img_orig, dtype=np.float32) / 255.0

    # 归一化热力图（只保留正相关性）
    hm = np.maximum(heatmap, 0.0)
    if hm.max() > 0:
        hm = hm / hm.max()

    colored = cm.get_cmap('jet')(hm)[:, :, :3]
    blend   = np.clip(alpha * colored + (1 - alpha) * img_np, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    title_suffix = f" — {class_name}" if class_name else ""

    axes[0].imshow(img_np)
    axes[0].set_title('原图')
    axes[0].axis('off')

    im = axes[1].imshow(hm, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f'LRP 热力图{title_suffix}')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(blend)
    axes[2].set_title('叠加图')
    axes[2].axis('off')

    plt.suptitle(f'LRP 解释: {class_name}', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] {output_path}")


# ────────────────────────────────────────────────────────────────────────────
# 批量处理
# ────────────────────────────────────────────────────────────────────────────

def process_folder(
    model: LRPResNet50,
    image_folder: str,
    output_folder: str,
    class_idx: int,
    class_names: list[str],
    image_size: int = 224,
    lrp_rule: str = 'epsilon',
):
    """对文件夹内所有图片批量生成 LRP 热力图。"""
    os.makedirs(output_folder, exist_ok=True)
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_paths = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    if not image_paths:
        print(f"[!] 在 '{image_folder}' 中找不到图片文件。")
        return

    class_name = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
    print(f"\n处理 {len(image_paths)} 张图片，解释属性 [{class_idx}] '{class_name}' ...")

    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_folder, f'{stem}_lrp_cls{class_idx}.png')
        try:
            image = preprocess_image(img_path, image_size=image_size)
            heatmap, logits = model.lrp(image, class_idx=class_idx,
                                         lrp_rule=lrp_rule)
            save_heatmap(heatmap, img_path, out_path, class_name=class_name)
            pred_score = float(1 / (1 + np.exp(-logits[class_idx])))   # sigmoid
            print(f"    {stem}: sigmoid({class_idx})={pred_score:.4f}")
        except Exception as exc:
            print(f"  [ERROR] {stem}: {exc}")


# ────────────────────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────────────────────

def load_class_names(num_classes: int) -> list[str]:
    """尝试从 resources 目录加载 attribute 名称。"""
    candidates = [
        'resources/data/17tags_meta.txt',
        'resources/data/cifar10.txt',
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                names = [ln.strip() for ln in f if ln.strip()]
            if len(names) >= num_classes:
                print(f"[✓] 从 '{path}' 加载类别名称")
                return names[:num_classes]
    return [f'class_{i}' for i in range(num_classes)]


def inspect_checkpoint(checkpoint_dir: str):
    """打印 checkpoint 中所有变量名和形状（用于调试）。"""
    ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt_path is None:
        print(f"[!] '{checkpoint_dir}' 中找不到 checkpoint")
        return
    reader = tf.train.load_checkpoint(ckpt_path)
    shape_map = reader.get_variable_to_shape_map()
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"共 {len(shape_map)} 个变量:\n")
    for name in sorted(shape_map):
        print(f"  {name:<60s}  {shape_map[name]}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='LRP 热力图生成工具 — ResNet50 (TF1 checkpoint)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--checkpoint_dir', required=True,
                   help='TF1 model checkpoint 目录路径')
    p.add_argument('--image_path',
                   help='单张图片路径')
    p.add_argument('--image_folder',
                   help='批量处理：输入图片文件夹')
    p.add_argument('--output_path',   default='lrp_heatmap.png',
                   help='单张图片输出路径 (default: lrp_heatmap.png)')
    p.add_argument('--output_folder', default='lrp_output',
                   help='批量输出文件夹 (default: lrp_output/)')
    p.add_argument('--class_idx',     type=int, default=0,
                   help='要解释的 attribute 索引 (0-indexed, default: 0)')
    p.add_argument('--num_classes',   type=int, default=17,
                   help='输出类别数 (default: 17, X-ray attributes)')
    p.add_argument('--image_size',    type=int, default=224,
                   help='输入图片 resize 尺寸 (default: 224)')
    p.add_argument('--lrp_rule',
                   choices=['epsilon', 'zplus'], default='epsilon',
                   help='LRP 规则: epsilon (default) 或 zplus (第一层)')
    p.add_argument('--list_classes',  action='store_true',
                   help='列出所有 attribute 名称后退出')
    p.add_argument('--inspect',       action='store_true',
                   help='打印 checkpoint 所有变量名后退出（调试用）')
    return p.parse_args()


def main():
    args = parse_args()

    if args.inspect:
        inspect_checkpoint(args.checkpoint_dir)
        return

    class_names = load_class_names(args.num_classes)

    if args.list_classes:
        print(f"\n共 {args.num_classes} 个 attribute:")
        for i, name in enumerate(class_names):
            print(f"  {i:3d}: {name}")
        return

    print(f"加载模型: {args.checkpoint_dir}")
    model = LRPResNet50(
        checkpoint_dir=args.checkpoint_dir,
        num_classes=args.num_classes,
        image_size=args.image_size,
    )
    cname = class_names[args.class_idx]
    print(f"解释 attribute [{args.class_idx}]: '{cname}'")
    print(f"LRP 规则: {args.lrp_rule}\n")

    if args.image_path:
        image = preprocess_image(args.image_path, image_size=args.image_size)
        heatmap, logits = model.lrp(image, class_idx=args.class_idx,
                                     lrp_rule=args.lrp_rule)
        probs = 1 / (1 + np.exp(-logits))
        print("预测概率 (sigmoid):")
        for i, (name, prob) in enumerate(zip(class_names, probs)):
            marker = " ◀" if i == args.class_idx else ""
            print(f"  [{i:2d}] {name:<30s}: {prob:.4f}{marker}")
        save_heatmap(heatmap, args.image_path, args.output_path,
                     class_name=cname)

    elif args.image_folder:
        process_folder(
            model, args.image_folder, args.output_folder,
            class_idx=args.class_idx,
            class_names=class_names,
            image_size=args.image_size,
            lrp_rule=args.lrp_rule,
        )
    else:
        print("[!] 请指定 --image_path 或 --image_folder")
        sys.exit(1)


if __name__ == '__main__':
    main()
