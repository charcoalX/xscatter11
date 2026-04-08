from __future__ import print_function, division, absolute_import, unicode_literals

import os, sys, base64
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resnet_src'))
try:
    import resnet_model_org as model
    from convert_to_tfrecords import tags_meta
    LABELS   = [m[1] for m in tags_meta]
    NUM_TAGS = len(LABELS)
    print("[OK] Loaded {0} classes".format(NUM_TAGS))
except ImportError as e:
    print("[ERR] Import error: {0}".format(e))
    sys.exit(1)

RAW_IMG_SIZE   = 256
MODEL_IMG_SIZE = 224
BATCH_SIZE     = 1


def load_jpg_image(jpg_path):
    """Load JPG as grayscale, log-normalize, return float32 (1, 256, 256)."""
    image = Image.open(jpg_path).convert('L')
    image = image.resize((256, 256), Image.LANCZOS)
    image = np.array(image, dtype=np.float32)
    image = np.clip(image, 1e-12, None)
    image = np.log(image) / np.log(1.0414)
    image[np.isinf(image)] = 0
    return image[np.newaxis, ...]   # (1, 256, 256)


def image_for_display(raw):
    """Normalize log image (256,256) to [0,1] float32."""
    img = raw.squeeze()
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    return img.astype(np.float32)


def build_graph(graph, ckpt_path):
    with graph.as_default():
        raw_images_op = tf.placeholder(
            tf.float32, [BATCH_SIZE, RAW_IMG_SIZE, RAW_IMG_SIZE], name="raw_images")
        images = tf.expand_dims(raw_images_op, 3)   # (1,256,256,1)

        tf.placeholder(tf.float32, [BATCH_SIZE, NUM_TAGS], name="labels")

        images = tf.image.resize_images(
            images, np.array([MODEL_IMG_SIZE, MODEL_IMG_SIZE]))

        std_images = []
        for idx in range(BATCH_SIZE):
            std_image = tf.image.per_image_standardization(images[idx])
            std_images.append(tf.expand_dims(std_image, 0))
        images_op = tf.concat(std_images, 0)   # (1,224,224,1)

        logits_op = model.inference(images_op, is_training=False, num_classes=NUM_TAGS)
        prob_op   = tf.sigmoid(logits_op)

        print("[i] Building gradient ops for {0} classes ...".format(NUM_TAGS))
        logits_list = tf.unstack(logits_op, axis=1)
        grad_ops = []
        for c_idx, logit_c in enumerate(logits_list):
            g  = tf.gradients(logit_c, images_op)[0]
            gi = g * images_op
            grad_ops.append(gi)
            if (c_idx + 1) % 5 == 0:
                print("    built {0}/{1}".format(c_idx + 1, NUM_TAGS))
        print("[OK] Gradient ops ready")

        saver = tf.train.Saver(tf.global_variables())
        return {
            "raw_images_op": raw_images_op,
            "prob_op":       prob_op,
            "grad_ops":      grad_ops,
            "images_op":     images_op,
            "saver":         saver,
        }


def relevance_to_heatmap(R):
    R = R.squeeze()
    R = np.maximum(R, 0)
    rmin, rmax = R.min(), R.max()
    return (R - rmin) / (rmax - rmin + 1e-9)


def float_img_to_b64(arr, cmap=None):
    if cmap is not None:
        arr = plt.get_cmap(cmap)(arr)[:, :, :3]
    arr_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if arr_uint8.ndim == 2:
        img = Image.fromarray(arr_uint8, mode='L').convert('RGB')
    else:
        img = Image.fromarray(arr_uint8)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def overlay_b64(gray_img, heatmap):
    rgb   = np.stack([gray_img] * 3, axis=-1)
    jet   = plt.get_cmap("jet")(heatmap)[:, :, :3]
    blend = np.clip(0.55 * rgb + 0.45 * jet, 0, 1)
    return float_img_to_b64(blend)


class LRPEngine(object):
    def __init__(self, ckpt_path, num_classes=17):
        self.graph = tf.Graph()
        self.ops   = build_graph(self.graph, ckpt_path)
        config = tf.ConfigProto(
            intra_op_parallelism_threads=2,
            inter_op_parallelism_threads=2)
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            print("[i] Restoring weights from: {0}".format(ckpt_path))
            self.ops["saver"].restore(self.sess, ckpt_path)
            print("[OK] Weights restored")

    def compute(self, image_path, class_idx):
        """Returns (overlay_b64_str, pred_prob_float)."""
        image_feed = load_jpg_image(image_path)
        feed = {self.ops["raw_images_op"]: image_feed}
        prob_vals, gi_val = self.sess.run(
            [self.ops["prob_op"], self.ops["grad_ops"][class_idx]],
            feed_dict=feed
        )
        pred_prob   = float(prob_vals[0][class_idx])
        R           = relevance_to_heatmap(gi_val)
        heatmap_b64 = float_img_to_b64(R, cmap='jet')
        return heatmap_b64, pred_prob
