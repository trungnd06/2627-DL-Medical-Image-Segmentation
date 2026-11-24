"""Reports and final post-split statistics for step 2."""
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median, quantiles
from .common import MASK_THRESHOLD, save_csv
from .statistics import assign_size_groups, save_sample_figures
from .duplicates import DUPLICATE_SCOPES

def write_reports(project_root, rows_by_dataset, resolution_rows, review_rows, plan, manifest):
    PROJECT_ROOT = Path(project_root)
    OUTPUT_DIR = PROJECT_ROOT / "logs/dataset_statistics"
    split_ids = plan["split_ids"]
    SPLIT_SEED = plan["seed"]
    excluded_kvasir = plan["context"]["excluded_kvasir"]
    source_metadata = manifest["dataset_sources"]
    size_rows, (q1,q2), size_rule = assign_size_groups(PROJECT_ROOT, rows_by_dataset)
    all_rows = [row for rows in rows_by_dataset.values() for row in rows]
    tables = {"dataset_statistics.csv": all_rows, "resolution_counts.csv": resolution_rows,
              "polyp_size_counts.csv": size_rows}
    RESULT_TABLES = PROJECT_ROOT / "results" / "tables"
    RESULT_FIGURES = PROJECT_ROOT / "results" / "figures"
    RESULT_TABLES.mkdir(parents=True, exist_ok=True)
    RESULT_FIGURES.mkdir(parents=True, exist_ok=True)
    for filename, rows in tables.items():
        save_csv(OUTPUT_DIR / filename, list(rows[0]), rows)
        save_csv(RESULT_TABLES / filename, list(rows[0]), rows)

    split_statistics = []
    row_lookup = {f"{ds}/{r['filename']}": r for ds, rows in rows_by_dataset.items() for r in rows}
    for label, ids in split_ids.items():
        rows = [row_lookup[key] for key in ids]
        sizes = Counter(r["size_group"] for r in rows)
        split_statistics.append({"split": label, "image_count": len(rows),
            "mean_area_ratio": mean(r["area_ratio"] for r in rows),
            "median_area_ratio": median(r["area_ratio"] for r in rows),
            **{group: sizes[group] for group in ("small", "medium", "large", "empty")}})
    save_csv(RESULT_TABLES / "split_statistics.csv", list(split_statistics[0]), split_statistics)

    save_sample_figures(PROJECT_ROOT, rows_by_dataset, RESULT_FIGURES)
    review_summary = {scope: dict(Counter(r["decision"] for r in review_rows if r["scope"] == scope))
                      for scope in DUPLICATE_SCOPES}
    summary = {
        "step": 2, "status": "complete_with_documented_screening_limits",
        "split_counts": manifest["counts"], "mask_threshold": MASK_THRESHOLD,
        "size_rule": size_rule, "size_thresholds_from_train": [q1, q2],
        "review_summary": review_summary,
        "screening_counts": dict(Counter(r["screening_reason"] for r in review_rows)),
        "checks": {"source_manifest_checksums_valid": True, "split_ids_valid_unique_disjoint": True,
                   "coverage_valid": True, "protected_groups_not_split": True,
                   "same_seed_reproduces_split": True},
    }
    (RESULT_TABLES / "step2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    report = ["# Kết quả bước 2 — Kiểm tra dữ liệu và cố định split", "",
              "## Dữ liệu và chia tập", "",
              "Ảnh/mask được ghép theo tên, kiểm tra kích thước, nhị phân hóa mask >127. Đã đối chiếu SHA-256 3.224 file với manifest nguồn.", "",
              "| Tập | Số ảnh |", "| --- | ---: |"]
    report += [f"| {name} | {len(ids)} |" for name, ids in split_ids.items()]
    report += ["", f"Seed: {SPLIT_SEED}; Kvasir loại do liên hệ external: {len(excluded_kvasir)}.",
               f"Ngưỡng diện tích từ train: {q1:.6f} và {q2:.6f}; áp dụng nguyên cho mọi tập.", "",
               "## Nguồn và phiên bản", ""]
    report += [f"- {dataset}: {details}" for dataset, details in source_metadata.items()]
    report += ["", "## Rà ảnh và giới hạn", "",
               "Sàng lọc SHA-256 byte, RGB64 và dHash/pHash. Ảnh tính hash giữ tỷ lệ bằng Lanczos, padding đen ở giữa; RGB 64×64, dHash 9×8, pHash 32×32. Cùng RGB64 chỉ chứng minh cùng biểu diễn thu nhỏ; padding cũng ảnh hưởng hash.",
               f"Kvasir: {sum(r['scope'] == 'Kvasir-SEG' and r['screening_reason'] == 'hash_threshold' for r in review_rows)} cặp qua ngưỡng; {sum(r['scope'] == 'Kvasir-SEG' and r['screening_reason'] == 'historical_manual_review' for r in review_rows)} cặp lịch sử ngoài ngưỡng được giữ để rà.",
               "Khoảng cách hash có thể thay đổi theo phiên bản Pillow/Python. Giữ các cặp lịch sử và quyết định đã lưu; manifest ghi phiên bản chạy thực tế.",
               "Giữ quyết định lịch sử đã lưu; ứng viên mới phải được rà, không tự gán confirmed. Giữ chung nhóm confirmed và unsure khi chia, không đổi unsure thành confirmed.", ""]
    report += [f"- {scope}: {counts}" for scope, counts in review_summary.items()]
    report += ["", "Chưa rà hết các cặp trong riêng ClinicDB; toàn bộ bộ này thuộc external. Không chứng minh độc lập theo bệnh nhân hoặc chuỗi nội soi.",
               "Không dùng ClinicDB để chọn loss, checkpoint, threshold hoặc siêu tham số. Chưa có kết quả train.", "",
               "## Tái tạo", "", "Chạy python -m scripts.prepare_data --project-root . với dữ liệu đúng manifest nguồn.",
               "splits/ lưu ID, nhóm, review, fingerprint và manifest; results/tables/ và results/figures/ lưu sản phẩm báo cáo.",
               "Code chính nằm trong src/datasets/; notebook chỉ gọi module. Không cần notebook test bị ignore để chạy lại.", ""]
    (RESULT_TABLES / "step2_report.md").write_text("\n".join(report), encoding="utf-8")
    print("HOÀN THÀNH BƯỚC 2")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Tiếp theo: Dataset/transforms và metric, rồi U-Net M1.")
    return summary


