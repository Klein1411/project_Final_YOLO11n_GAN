import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "Enhancement Model" / "Model Output" / "gen_model_037100.h5"
DEFAULT_DATASET = PROJECT_DIR / "dataset.npz"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
tf = None
load_model = None


def add_dll_paths():
    dll_dirs = [
        PROJECT_DIR,
        PROJECT_DIR / "venv" / "Scripts",
        Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin"),
    ]
    for dll_dir in dll_dirs:
        if dll_dir.exists():
            try:
                os.add_dll_directory(str(dll_dir))
            except (AttributeError, OSError):
                pass
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")


def configure_runtime():
    global tf, load_model

    add_dll_paths()
    import tensorflow as tensorflow
    from tensorflow.keras.models import load_model as keras_load_model

    tf = tensorflow
    load_model = keras_load_model

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return gpus


def to_01(x):
    return np.clip(x.astype("float32"), 0.0, 1.0)


def model_input_from_uint8(x):
    return (x.astype("float32") - 127.5) / 127.5


def model_output_to_01(x):
    return np.clip((x.astype("float32") + 1.0) / 2.0, 0.0, 1.0)


def load_npz_dataset(dataset_path):
    data = np.load(dataset_path)
    high = data["arr_0"]
    low = data["arr_1"]
    names = [f"sample_{i:05d}" for i in range(len(low))]
    return low, high, names


def load_image(path, size=(256, 256)):
    with Image.open(path) as img:
        img = img.convert("RGB").resize(size)
        return np.asarray(img, dtype=np.uint8)


def load_folder_dataset(low_dir, gt_dir):
    low_dir = Path(low_dir)
    gt_dir = Path(gt_dir)
    low_files = {
        p.name: p for p in sorted(low_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS
    }
    gt_files = {
        p.name: p for p in sorted(gt_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS
    }
    names = sorted(set(low_files).intersection(gt_files))
    if not names:
        raise ValueError("No matching image names found between low_dir and gt_dir.")

    low = np.asarray([load_image(low_files[name]) for name in names], dtype=np.uint8)
    high = np.asarray([load_image(gt_files[name]) for name in names], dtype=np.uint8)
    stems = [Path(name).stem for name in names]
    return low, high, stems


def luminance(rgb_01):
    return (
        0.2126 * rgb_01[..., 0]
        + 0.7152 * rgb_01[..., 1]
        + 0.0722 * rgb_01[..., 2]
    )


def entropy_luminance(rgb_01):
    y = np.clip(luminance(rgb_01) * 255.0, 0, 255).astype(np.uint8)
    hist = np.bincount(y.ravel(), minlength=256).astype("float64")
    prob = hist / np.maximum(hist.sum(), 1.0)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def colorfulness(rgb_01):
    rgb = rgb_01.astype("float32") * 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    return float(std_root + 0.3 * mean_root)


def lab_delta_e_mean(a_01, b_01):
    a_lab = cv2.cvtColor(a_01.astype("float32"), cv2.COLOR_RGB2LAB)
    b_lab = cv2.cvtColor(b_01.astype("float32"), cv2.COLOR_RGB2LAB)
    return float(np.mean(np.linalg.norm(a_lab - b_lab, axis=-1)))


def universal_image_quality_index(a_01, b_01):
    a = luminance(a_01).astype("float64")
    b = luminance(b_01).astype("float64")
    mean_a = a.mean()
    mean_b = b.mean()
    var_a = a.var()
    var_b = b.var()
    cov = ((a - mean_a) * (b - mean_b)).mean()
    denom = (var_a + var_b) * (mean_a**2 + mean_b**2)
    if denom <= 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float((4.0 * cov * mean_a * mean_b) / denom)


def normalized_cross_correlation(a_01, b_01):
    a = luminance(a_01).astype("float64")
    b = luminance(b_01).astype("float64")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(a * b) / denom)


def tf_scalar_metric(fn, a_01, b_01):
    try:
        a = tf.convert_to_tensor(a_01[None, ...], dtype=tf.float32)
        b = tf.convert_to_tensor(b_01[None, ...], dtype=tf.float32)
        return float(fn(a, b).numpy()[0])
    except Exception:
        return float("nan")


def full_reference_metrics(pred_01, gt_01, prefix):
    pred_01 = to_01(pred_01)
    gt_01 = to_01(gt_01)

    return {
        f"{prefix}_psnr": tf_scalar_metric(
            lambda x, y: tf.image.psnr(x, y, max_val=1.0), pred_01, gt_01
        ),
        f"{prefix}_ssim": tf_scalar_metric(
            lambda x, y: tf.image.ssim(x, y, max_val=1.0), pred_01, gt_01
        ),
    }


def image_quality_descriptors(rgb_01, prefix):
    y = luminance(rgb_01)
    return {
        f"{prefix}_brightness_mean": float(np.mean(y)),
        f"{prefix}_brightness_std": float(np.std(y)),
        f"{prefix}_contrast_rms": float(np.std(y)),
        f"{prefix}_entropy": entropy_luminance(rgb_01),
        f"{prefix}_colorfulness": colorfulness(rgb_01),
        f"{prefix}_saturation_mean": float(
            np.mean(cv2.cvtColor(rgb_01.astype("float32"), cv2.COLOR_RGB2HSV)[..., 1])
        ),
    }


def save_sample_grid(low_01, enhanced_01, gt_01, path):
    def label(img_01, text):
        img = (np.clip(img_01, 0, 1) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        return img

    grid = np.concatenate(
        [label(low_01, "Low"), label(enhanced_01, "Enhanced"), label(gt_01, "Ground Truth")],
        axis=1,
    )
    cv2.imwrite(str(path), grid)


def summarize(rows):
    numeric_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    summary = {}
    for key in numeric_keys:
        values = np.asarray([row[key] for row in rows], dtype="float64")
        values = values[np.isfinite(values)]
        if values.size == 0:
            summary[key] = {"mean": None, "std": None, "min": None, "max": None}
        else:
            summary[key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return summary


def batched_predict(model, low_uint8, batch_size):
    enhanced_batches = []
    timings = []
    for start in range(0, len(low_uint8), batch_size):
        batch = low_uint8[start : start + batch_size]
        x = model_input_from_uint8(batch)
        t0 = time.perf_counter()
        pred = model.predict(x, verbose=0)
        elapsed = time.perf_counter() - t0
        timings.extend([elapsed / len(batch)] * len(batch))
        enhanced_batches.append(model_output_to_01(pred))
    return np.concatenate(enhanced_batches, axis=0), timings


def evaluate(args):
    gpus = configure_runtime()
    print("GPUs:", gpus)

    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    enhanced_dir = output_dir / "enhanced"
    samples_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    if args.low_dir and args.gt_dir:
        low_uint8, gt_uint8, names = load_folder_dataset(args.low_dir, args.gt_dir)
    else:
        low_uint8, gt_uint8, names = load_npz_dataset(args.dataset)

    if args.limit:
        low_uint8 = low_uint8[: args.limit]
        gt_uint8 = gt_uint8[: args.limit]
        names = names[: args.limit]

    print("Loading model:", model_path)
    model = load_model(model_path, compile=False)
    print("Metrics: PSNR, SSIM")
    print("Evaluating samples:", len(low_uint8))

    warmup = model_input_from_uint8(low_uint8[:1])
    model.predict(warmup, verbose=0)

    enhanced_01, timings = batched_predict(model, low_uint8, args.batch_size)
    low_01 = low_uint8.astype("float32") / 255.0
    gt_01 = gt_uint8.astype("float32") / 255.0

    rows = []
    for i, name in enumerate(names):
        row = {"index": i, "name": name, "inference_seconds": float(timings[i])}
        row.update(full_reference_metrics(low_01[i], gt_01[i], "low_vs_gt"))
        row.update(full_reference_metrics(enhanced_01[i], gt_01[i], "enhanced_vs_gt"))

        row["psnr_gain"] = row["enhanced_vs_gt_psnr"] - row["low_vs_gt_psnr"]
        row["ssim_gain"] = row["enhanced_vs_gt_ssim"] - row["low_vs_gt_ssim"]
        rows.append(row)

        enhanced_img = (enhanced_01[i] * 255.0).round().astype(np.uint8)
        Image.fromarray(enhanced_img).save(enhanced_dir / f"{name}_enhanced.png")
        if i < args.save_samples:
            save_sample_grid(low_01[i], enhanced_01[i], gt_01[i], samples_dir / f"{name}_grid.png")

    csv_path = output_dir / "per_image_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": str(model_path),
        "num_samples": len(rows),
        "batch_size": args.batch_size,
        "metrics": summarize(rows),
    }
    summary_path = output_dir / "summary_metrics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", csv_path)
    print("Saved:", summary_path)
    print("Saved enhanced images:", enhanced_dir)
    print("Saved sample grids:", samples_dir)
    print("\nKey summary:")
    key_summary = [
        "low_vs_gt_psnr",
        "enhanced_vs_gt_psnr",
        "psnr_gain",
        "low_vs_gt_ssim",
        "enhanced_vs_gt_ssim",
        "ssim_gain",
        "inference_seconds",
    ]
    for key in key_summary:
        stats = summary["metrics"].get(key, {})
        print(f"{key}: mean={stats.get('mean')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained low-light enhancement generator."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to gen_model_*.h5")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to dataset.npz")
    parser.add_argument("--low-dir", default=None, help="Folder with low-light images")
    parser.add_argument("--gt-dir", default=None, help="Folder with ground-truth images")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output folder")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N samples")
    parser.add_argument("--save-samples", type=int, default=20, help="Number of comparison grids")
    args = parser.parse_args()

    if not args.dataset and (not args.low_dir or not args.gt_dir):
        parser.error("Use either --dataset or both --low-dir and --gt-dir.")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
