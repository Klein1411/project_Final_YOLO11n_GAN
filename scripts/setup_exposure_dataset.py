import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def brightness_mean(image_path, metric_size):
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize((metric_size, metric_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
    luminance = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return float(luminance.mean())


def list_scene_images(scene_dir):
    return sorted(
        [p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: natural_key(p.name),
    )


def choose_target(images, target_brightness, target_min, target_max):
    candidates = [
        item for item in images if target_min <= item["brightness"] <= target_max
    ]
    if not candidates:
        candidates = images
    return min(candidates, key=lambda item: abs(item["brightness"] - target_brightness))


def build_scene_pairs(
    dataset_name,
    scene_dir,
    metric_size,
    target_brightness,
    target_min,
    target_max,
    min_gap,
    max_low_per_scene,
):
    files = list_scene_images(scene_dir)
    if len(files) < 2:
        return [], {"reason": "fewer_than_2_images", "num_images": len(files)}

    images = [
        {
            "path": image_path,
            "name": image_path.name,
            "brightness": brightness_mean(image_path, metric_size),
        }
        for image_path in files
    ]
    images = sorted(images, key=lambda item: item["brightness"])
    target = choose_target(images, target_brightness, target_min, target_max)

    low_candidates = [
        item for item in images if item["brightness"] <= target["brightness"] - min_gap
    ]
    if max_low_per_scene > 0:
        low_candidates = low_candidates[:max_low_per_scene]

    pairs = []
    for low in low_candidates:
        pair_name = safe_name(
            f"{dataset_name}_{scene_dir.name}_low_{Path(low['name']).stem}_gt_{Path(target['name']).stem}"
        )
        pairs.append(
            {
                "dataset": dataset_name,
                "scene": scene_dir.name,
                "pair_name": pair_name,
                "low_path": str(low["path"]),
                "gt_path": str(target["path"]),
                "low_file": low["name"],
                "gt_file": target["name"],
                "low_brightness": low["brightness"],
                "gt_brightness": target["brightness"],
                "brightness_gap": target["brightness"] - low["brightness"],
                "num_scene_images": len(files),
            }
        )

    if not pairs:
        return [], {
            "reason": "no_low_image_darker_than_target_by_min_gap",
            "num_images": len(files),
            "target_file": target["name"],
            "target_brightness": target["brightness"],
        }
    return pairs, None


def collect_pairs(args):
    all_pairs = []
    skipped = []
    datasets = [
        ("part1", Path(args.part1)),
        ("part2", Path(args.part2)),
    ]

    for dataset_name, dataset_dir in datasets:
        if not dataset_dir.exists():
            raise FileNotFoundError(dataset_dir)

        scene_dirs = sorted(
            [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.lower() != "label"],
            key=lambda p: natural_key(p.name),
        )
        if args.max_scenes_per_part:
            scene_dirs = scene_dirs[: args.max_scenes_per_part]

        for scene_dir in scene_dirs:
            pairs, skip_info = build_scene_pairs(
                dataset_name=dataset_name,
                scene_dir=scene_dir,
                metric_size=args.metric_size,
                target_brightness=args.target_brightness,
                target_min=args.target_min,
                target_max=args.target_max,
                min_gap=args.min_gap,
                max_low_per_scene=args.max_low_per_scene,
            )
            all_pairs.extend(pairs)
            if skip_info:
                skipped.append(
                    {
                        "dataset": dataset_name,
                        "scene": scene_dir.name,
                        **skip_info,
                    }
                )

    return all_pairs, skipped


def split_pairs_by_scene(pairs, eval_ratio, seed):
    scene_keys = sorted(
        {(pair["dataset"], pair["scene"]) for pair in pairs},
        key=lambda item: (item[0], natural_key(item[1])),
    )
    rng = np.random.default_rng(seed)
    shuffled = list(scene_keys)
    rng.shuffle(shuffled)

    eval_count = int(np.ceil(len(shuffled) * eval_ratio))
    eval_scenes = set(shuffled[:eval_count])

    train_pairs = [pair for pair in pairs if (pair["dataset"], pair["scene"]) not in eval_scenes]
    eval_pairs = [pair for pair in pairs if (pair["dataset"], pair["scene"]) in eval_scenes]
    return train_pairs, eval_pairs


def prepare_output_dir(output_dir, overwrite):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    for split in ["train", "eval"]:
        (output_dir / split / "low").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "ground_truth").mkdir(parents=True, exist_ok=True)
    return output_dir


def save_image_pair(pair, split_dir, resize):
    suffix = ".png"
    output_name = pair["pair_name"] + suffix
    low_out = split_dir / "low" / output_name
    gt_out = split_dir / "ground_truth" / output_name

    for src, dst in [(pair["low_path"], low_out), (pair["gt_path"], gt_out)]:
        with Image.open(src) as img:
            img = img.convert("RGB")
            if resize:
                img = img.resize(tuple(resize))
            img.save(dst)

    return output_name


def write_manifest(path, rows):
    fieldnames = [
        "split",
        "output_name",
        "pair_name",
        "dataset",
        "scene",
        "low_file",
        "gt_file",
        "low_brightness",
        "gt_brightness",
        "brightness_gap",
        "num_scene_images",
        "low_path",
        "gt_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def materialize_dataset(output_dir, train_pairs, eval_pairs, resize):
    manifest_rows = []
    for split, pairs in [("train", train_pairs), ("eval", eval_pairs)]:
        split_dir = output_dir / split
        for pair in pairs:
            output_name = save_image_pair(pair, split_dir, resize)
            manifest_rows.append(
                {
                    "split": split,
                    "output_name": output_name,
                    **pair,
                }
            )
    return manifest_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create low/ground_truth train-eval pairs from exposure-bracketed Dataset_Part1 and Dataset_Part2."
    )
    parser.add_argument("--part1", default="Dataset_Part1")
    parser.add_argument("--part2", default="Dataset_Part2")
    parser.add_argument("--output-dir", default="dataset_exposure_combined")
    parser.add_argument("--eval-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-brightness", type=float, default=0.60)
    parser.add_argument("--target-min", type=float, default=0.45)
    parser.add_argument("--target-max", type=float, default=0.75)
    parser.add_argument("--min-gap", type=float, default=0.08)
    parser.add_argument(
        "--max-low-per-scene",
        type=int,
        default=3,
        help="Use the N darkest valid low images per scene. Use 0 for all valid lows.",
    )
    parser.add_argument("--metric-size", type=int, default=256)
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=[256, 256],
        help="Resize output images. Use --no-resize to keep original size.",
    )
    parser.add_argument("--no-resize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-scenes-per-part",
        type=int,
        default=None,
        help="Debug option: only process first N scenes from each part.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    resize = None if args.no_resize else args.resize

    pairs, skipped = collect_pairs(args)
    train_pairs, eval_pairs = split_pairs_by_scene(pairs, args.eval_ratio, args.seed)

    summary = {
        "part1": str(args.part1),
        "part2": str(args.part2),
        "output_dir": str(args.output_dir),
        "target_brightness": args.target_brightness,
        "target_min": args.target_min,
        "target_max": args.target_max,
        "min_gap": args.min_gap,
        "max_low_per_scene": args.max_low_per_scene,
        "resize": resize,
        "num_pairs": len(pairs),
        "num_train_pairs": len(train_pairs),
        "num_eval_pairs": len(eval_pairs),
        "num_skipped_scenes": len(skipped),
    }

    print(json.dumps(summary, indent=2))

    if args.dry_run:
        print("Dry run only. No files were written.")
        return

    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    manifest_rows = materialize_dataset(output_dir, train_pairs, eval_pairs, resize)

    write_manifest(output_dir / "manifest.csv", manifest_rows)
    write_manifest(
        output_dir / "train_manifest.csv",
        [row for row in manifest_rows if row["split"] == "train"],
    )
    write_manifest(
        output_dir / "eval_manifest.csv",
        [row for row in manifest_rows if row["split"] == "eval"],
    )
    with (output_dir / "skipped_scenes.json").open("w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2)
    with (output_dir / "setup_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved dataset:", output_dir)
    print("Train low:", output_dir / "train" / "low")
    print("Train ground_truth:", output_dir / "train" / "ground_truth")
    print("Eval low:", output_dir / "eval" / "low")
    print("Eval ground_truth:", output_dir / "eval" / "ground_truth")


if __name__ == "__main__":
    main()
