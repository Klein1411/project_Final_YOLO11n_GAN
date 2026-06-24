import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = {
    "psnr": {
        "title": "PSNR",
        "ylabel": "PSNR (dB)",
        "better": "higher",
    },
    "ssim": {
        "title": "SSIM",
        "ylabel": "SSIM",
        "better": "higher",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot quantitative metrics for low-light enhancement evaluation."
    )
    parser.add_argument(
        "--eval-dir",
        default="evaluation_eval15",
        help="Folder containing per_image_metrics.csv and summary_metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder to save plots. Default: <eval-dir>/plots",
    )
    return parser.parse_args()


def read_rows(csv_path):
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        for key, value in list(row.items()):
            if key in {"name"}:
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def read_summary(summary_path):
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_columns(metric):
    return f"low_vs_gt_{metric}", f"enhanced_vs_gt_{metric}"


def has_metric(rows, metric):
    low_col, enhanced_col = metric_columns(metric)
    return rows and low_col in rows[0] and enhanced_col in rows[0]


def mean_from_rows(rows, column):
    values = [row[column] for row in rows if isinstance(row.get(column), (int, float))]
    return sum(values) / len(values)


def values_from_rows(rows, column):
    return [row[column] for row in rows if isinstance(row.get(column), (int, float))]


def metric_change(row, metric):
    low_col, enhanced_col = metric_columns(metric)
    if METRICS[metric]["better"] == "lower":
        return row[low_col] - row[enhanced_col]
    return row[enhanced_col] - row[low_col]


def format_number(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def save_mean_comparison(rows, metrics, output_path):
    labels = []
    low_values = []
    enhanced_values = []

    for metric in metrics:
        low_col, enhanced_col = metric_columns(metric)
        labels.append(METRICS[metric]["title"])
        low_values.append(mean_from_rows(rows, low_col))
        enhanced_values.append(mean_from_rows(rows, enhanced_col))

    x = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], low_values, width, label="Low vs GT")
    ax.bar([i + width / 2 for i in x], enhanced_values, width, label="Enhanced vs GT")
    ax.set_title("Mean Quantitative Metrics")
    ax.set_ylabel("Metric value")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_metric_per_image(rows, metric, output_path):
    low_col, enhanced_col = metric_columns(metric)
    names = [str(row["name"]) for row in rows]
    low_values = [row[low_col] for row in rows]
    enhanced_values = [row[enhanced_col] for row in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(names, low_values, marker="o", label="Low vs GT")
    ax.plot(names, enhanced_values, marker="o", label="Enhanced vs GT")
    ax.set_title(f"{METRICS[metric]['title']} Per Image")
    ax.set_ylabel(METRICS[metric]["ylabel"])
    ax.set_xlabel("Image")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_metric_change(rows, metric, output_path):
    low_col, enhanced_col = metric_columns(metric)
    names = [str(row["name"]) for row in rows]
    changes = [row[enhanced_col] - row[low_col] for row in rows]

    if METRICS[metric]["better"] == "lower":
        changes = [row[low_col] - row[enhanced_col] for row in rows]
        ylabel = f"{METRICS[metric]['title']} reduction"
        title = f"{METRICS[metric]['title']} Reduction Per Image"
    else:
        ylabel = f"{METRICS[metric]['title']} gain"
        title = f"{METRICS[metric]['title']} Gain Per Image"

    colors = ["#2f7d32" if value >= 0 else "#b3261e" for value in changes]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(names, changes, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Image")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_report(summary, available_metrics, output_path):
    metrics = summary.get("metrics", {})
    lines = [
        "# Quantitative Metrics Plot Report",
        "",
        f"Model: `{summary.get('model', 'unknown')}`",
        f"Number of samples: `{summary.get('num_samples', 'unknown')}`",
        "",
        "## Available metrics",
        "",
    ]

    for metric_index, metric in enumerate(available_metrics, start=2):
        low_col, enhanced_col = metric_columns(metric)
        low_mean = metrics.get(low_col, {}).get("mean")
        enhanced_mean = metrics.get(enhanced_col, {}).get("mean")
        lines.append(f"- {METRICS[metric]['title']}: low mean = `{low_mean}`, enhanced mean = `{enhanced_mean}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot_explanation(rows, available_metrics, output_path):
    lines = [
        "# Cach doc va danh gia cac bieu do",
        "",
        "File nay giai thich cach nhin cac bieu do duoc tao tu `per_image_metrics.csv`.",
        "Muc tieu la so sanh anh low-light ban dau voi anh da enhance cua model, deu doi chieu voi ground truth.",
        "",
        "## 1. `mean_metrics_comparison.png`",
        "",
        "Bieu do nay so sanh gia tri trung binh cua tung metric tren toan bo tap danh gia.",
        "",
        "- Cot `Low vs GT`: chat luong anh dau vao low-light so voi ground truth.",
        "- Cot `Enhanced vs GT`: chat luong anh model tao ra so voi ground truth.",
        "- Voi PSNR va SSIM, cot Enhanced cao hon Low la tot.",
        "",
        "Dung bieu do nay de ket luan tong quan model co cai thien anh hay khong.",
        "",
    ]

    for metric_index, metric in enumerate(available_metrics, start=2):
        low_col, enhanced_col = metric_columns(metric)
        low_values = values_from_rows(rows, low_col)
        enhanced_values = values_from_rows(rows, enhanced_col)
        changes = [metric_change(row, metric) for row in rows]
        improved_count = sum(1 for value in changes if value > 0)
        unchanged_count = sum(1 for value in changes if value == 0)
        worsened_count = len(changes) - improved_count - unchanged_count
        mean_low = sum(low_values) / len(low_values)
        mean_enhanced = sum(enhanced_values) / len(enhanced_values)
        mean_change = sum(changes) / len(changes)
        best_row = max(rows, key=lambda row: metric_change(row, metric))
        worst_row = min(rows, key=lambda row: metric_change(row, metric))

        if METRICS[metric]["better"] == "lower":
            direction = "cang thap cang tot"
            good_condition = "duong nghia la LPIPS giam, anh enhanced gan ground truth hon ve cam nhan"
            change_name = "reduction"
        else:
            direction = "cang cao cang tot"
            good_condition = "duong nghia la metric tang, anh enhanced tot hon anh low"
            change_name = "gain"

        lines.extend(
            [
                f"## {metric_index}. {METRICS[metric]['title']}",
                "",
                f"{METRICS[metric]['title']} la metric `{direction}`.",
                "",
                f"Gia tri trung binh Low vs GT: `{format_number(mean_low)}`.",
                f"Gia tri trung binh Enhanced vs GT: `{format_number(mean_enhanced)}`.",
                f"Muc cai thien trung binh: `{format_number(mean_change)}`.",
                f"So anh cai thien: `{improved_count}/{len(rows)}`.",
                f"So anh xau hon: `{worsened_count}/{len(rows)}`.",
                "",
                f"### `{metric}_per_image.png`",
                "",
                "Bieu do nay cho thay metric tren tung anh.",
                "",
                "- Duong `Low vs GT` la diem cua anh low-light ban dau.",
                "- Duong `Enhanced vs GT` la diem cua anh sau khi qua model.",
                f"- Voi {METRICS[metric]['title']}, {direction}.",
                "",
                "Cach danh gia:",
                "",
                "- Neu duong Enhanced tot hon Low tren hau het anh, model cai thien on dinh.",
                "- Neu hai duong gan nhau, model cai thien it.",
                "- Neu Enhanced te hon Low o mot so anh, can xem lai cac anh do vi model co the lam sai mau, mat chi tiet, hoac lam sang qua muc.",
                "",
                f"### `{metric}_change_per_image.png`",
                "",
                f"Bieu do nay ve muc `{change_name}` cua tung anh.",
                "",
                f"- Gia tri duong: {good_condition}.",
                "- Gia tri gan 0: model gan nhu khong cai thien.",
                "- Gia tri am: model lam anh te hon theo metric nay.",
                "",
                f"Anh cai thien manh nhat: `{best_row['name']}` voi muc `{change_name}` = `{format_number(metric_change(best_row, metric))}`.",
                f"Anh cai thien kem nhat: `{worst_row['name']}` voi muc `{change_name}` = `{format_number(metric_change(worst_row, metric))}`.",
                "",
            ]
        )

    lines.extend(
        [
            f"## {len(available_metrics) + 2}. Cach ket luan ngan gon",
            "",
            "Khi viet bao cao, nen ket luan theo thu tu:",
            "",
            "1. Nhin `mean_metrics_comparison.png` de noi model co cai thien tong the khong.",
            "2. Nhin cac bieu do `*_per_image.png` de xem cai thien co dong deu tren tung anh khong.",
            "3. Nhin cac bieu do `*_change_per_image.png` de tim anh tot nhat va anh yeu nhat.",
            "4. Neu PSNR tang nhung SSIM khong tang nhieu, model co the dung pixel hon nhung chua giu cau truc tot.",
            "5. Neu SSIM tang nhung PSNR thap, anh co the giu cau truc nhung con lech do sang hoac mau.",
            "",
            "Ket luan tot nen dua tren nhieu metric cung luc, khong dua vao mot bieu do duy nhat.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir) if args.output_dir else eval_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / "per_image_metrics.csv"
    summary_path = eval_dir / "summary_metrics.json"

    rows = read_rows(csv_path)
    summary = read_summary(summary_path)

    available_metrics = [metric for metric in METRICS if has_metric(rows, metric)]

    if not available_metrics:
        raise ValueError("No supported metrics found in per_image_metrics.csv.")

    save_mean_comparison(rows, available_metrics, output_dir / "mean_metrics_comparison.png")

    for metric in available_metrics:
        save_metric_per_image(rows, metric, output_dir / f"{metric}_per_image.png")
        save_metric_change(rows, metric, output_dir / f"{metric}_change_per_image.png")

    write_report(summary, available_metrics, output_dir / "plot_report.md")
    write_plot_explanation(
        rows,
        available_metrics,
        output_dir / "plot_explanation.md",
    )

    print("Saved plots to:", output_dir)


if __name__ == "__main__":
    main()
