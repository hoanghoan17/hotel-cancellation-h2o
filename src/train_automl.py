# -*- coding: utf-8 -*-
"""
H2O-3 AutoML Training Script
Dự án: Phân tích và dự đoán khả năng khách hàng hủy đặt phòng khách sạn bằng H2O AutoML
Tập huấn luyện: data/train_clean.csv
Tập kiểm tra:   data/test_clean.csv
Cấu hình:       src/features_config.json
"""

import os
import sys
import json
import time
import h2o
from h2o.automl import H2OAutoML
from h2o.frame import H2OFrame

# Đảm bảo in tiếng Việt UTF-8 chuẩn trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train_clean.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_clean.csv")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "features_config.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

def init_h2o():
    """Khởi tạo hoặc kết nối cụm H2O-3 an toàn."""
    print("=" * 70)
    print("BƯỚC 1: KHỞI TẠO HOẶC KẾT NỐI H2O CLUSTER")
    print("=" * 70)
    try:
        h2o.init(ip="localhost", port=54321, start_h2o=False)
        print("[OK] Đã kết nối tới H2O Cluster đang hoạt động.")
    except Exception:
        print("Chưa có cluster sẵn. Đang khởi tạo cluster mới với 4GB RAM...")
        try:
            h2o.init(max_mem_size="4G", nthreads=-1)
            print("[OK] H2O Cluster đã khởi động thành công.")
        except Exception as e:
            print(f"[LỖI] Không thể khởi động H2O Cluster: {e}")
            sys.exit(1)

def load_feature_config(config_path: str) -> dict:
    """Nạp cấu hình danh sách đặc trưng đã thống nhất ở bước chuẩn bị dữ liệu."""
    print("\n" + "=" * 70)
    print("BƯỚC 2: NẠP CẤU HÌNH ĐẶC TRƯNG")
    print("=" * 70)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    print(f"- Dự án: {config.get('project_name')}")
    print(f"- Biến mục tiêu (Target): '{config.get('target')}'")
    print(f"- Số lượng đặc trưng đầu vào (Features): {config.get('n_features')}")
    print(f"- Danh sách đặc trưng: {', '.join(config.get('features')[:8])}... (tổng cộng {len(config.get('features'))})")
    print(f"- Các cột rò rỉ bị loại bỏ: {config.get('leakage_columns_excluded')}")
    return config

def load_and_prepare_frames(train_path: str, test_path: str, config: dict):
    """Nạp tập train và test sạch vào H2OFrame và ép kiểu danh mục đồng nhất."""
    print("\n" + "=" * 70)
    print("BƯỚC 3: NẠP TẬP TRAIN VÀ TEST VÀO H2OFRAME")
    print("=" * 70)
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu train/test trong thư mục: {DATA_DIR}")
        
    print(f"Đang nạp tập Train: {train_path}")
    train_hf = h2o.upload_file(path=train_path, destination_frame="train_clean")
    print(f"[OK] Train Frame: {train_hf.nrow:,} dòng, {train_hf.ncol} cột.")
    
    print(f"Đang nạp tập Test: {test_path}")
    test_hf = h2o.upload_file(path=test_path, destination_frame="test_clean")
    print(f"[OK] Test Frame: {test_hf.nrow:,} dòng, {test_hf.ncol} cột.")
    
    target = config["target"]
    cat_cols = config.get("categorical_features", [])
    
    print("\nĐồng bộ kiểu dữ liệu cho Train và Test:")
    # Ép kiểu target sang factor
    train_hf[target] = train_hf[target].asfactor()
    test_hf[target] = test_hf[target].asfactor()
    print(f"- Ép kiểu biến mục tiêu '{target}' -> factor/enum.")
    
    # Ép kiểu các biến categorical
    for col in cat_cols:
        if col in train_hf.columns:
            train_hf[col] = train_hf[col].asfactor()
        if col in test_hf.columns:
            test_hf[col] = test_hf[col].asfactor()
    print(f"- Đã ép kiểu {len(cat_cols)} biến phân loại sang factor/enum.")
    
    return train_hf, test_hf

def train_automl_pipeline(train_hf: H2OFrame, features: list, target: str, max_runtime_secs: int = 60, seed: int = 42):
    """Huấn luyện mô hình bằng H2O AutoML."""
    print("\n" + "=" * 70)
    print("BƯỚC 4: HUẤN LUYỆN H2O AUTOML")
    print("=" * 70)
    print(f"Cấu hình AutoML:")
    print(f"- Thời gian tối đa (max_runtime_secs): {max_runtime_secs} giây")
    print(f"- Tiêu chí xếp hạng (sort_metric): AUC (Area Under the ROC Curve)")
    print(f"- Tiêu chí dừng sớm (stopping_metric): AUC")
    print(f"- Đánh giá nội bộ: 5-Fold Cross-Validation")
    print(f"- Random Seed: {seed}")
    print(f"- Số đặc trưng đầu vào (X): {len(features)}")
    print(f"- Biến mục tiêu (y): '{target}'")
    print("\nBắt đầu huấn luyện AutoML (quá trình có thể mất khoảng 1-2 phút)...")
    
    start_time = time.time()
    aml = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        sort_metric="AUC",
        stopping_metric="AUC",
        stopping_rounds=3,
        stopping_tolerance=0.001,
        seed=seed,
        project_name="hotel_cancellation_automl"
    )
    
    try:
        aml.train(x=features, y=target, training_frame=train_hf)
    except Exception as e:
        print(f"[LỖI TRONG QUÁ TRÌNH HUẤN LUYỆN AUTOML]: {e}")
        sys.exit(1)
        
    elapsed = time.time() - start_time
    print(f"[THÀNH CÔNG] Quá trình huấn luyện AutoML hoàn tất sau: {elapsed:.2f} giây.")
    return aml

def display_leaderboard(aml: H2OAutoML) -> H2OFrame:
    """Hiển thị bảng xếp hạng Leaderboard của H2O AutoML."""
    print("\n" + "=" * 70)
    print("BƯỚC 5: AUTOML LEADERBOARD")
    print("=" * 70)
    lb = aml.leaderboard
    print(f"Tổng số mô hình đã được huấn luyện: {lb.nrow}\n")
    
    # Chuyển bảng Leaderboard sang pandas để in rõ ràng
    lb_df = lb.as_data_frame()
    # Hiển thị các cột quan trọng
    display_cols = [c for c in ["model_id", "auc", "logloss", "aucpr", "mean_per_class_error", "rmse", "mse"] if c in lb_df.columns]
    print(lb_df[display_cols].to_string(index=False))
    return lb

def evaluate_leader_model(leader_model, test_hf: H2OFrame, target: str):
    """Đánh giá mô hình tốt nhất (Leader) trên tập test độc lập."""
    print("\n" + "=" * 70)
    print("BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH LEADER TRÊN TẬP TEST ĐỘC LẬP")
    print("=" * 70)
    print(f"Mô hình đứng đầu (Leader Model): {leader_model.model_id}")
    print(f"Thuật toán: {leader_model.algo}")
    
    print("\nĐang tính toán hiệu năng trên 23,948 dòng tập Test độc lập...")
    perf = leader_model.model_performance(test_data=test_hf)
    
    auc = perf.auc()
    try:
        aucpr = perf.pr_auc()
    except Exception:
        try:
            aucpr = perf.aucpr()
        except Exception:
            aucpr = 0.0
    logloss = perf.logloss()
    
    print("\n" + "-" * 50)
    print(f"KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP:")
    print(f"- ROC-AUC (Khả năng phân biệt hai lớp) : {auc:.4f} ({auc*100:.2f}%)")
    print(f"- PR-AUC (Precision-Recall AUC)         : {aucpr:.4f} ({aucpr*100:.2f}%)")
    print(f"- Logloss (Sai số logarit)              : {logloss:.4f}")
    print("-" * 50)
    
    # Ma trận nhầm lẫn (Confusion Matrix)
    try:
        cm = perf.confusion_matrix()
        print("\nMa trận nhầm lẫn (Confusion Matrix) trên tập Test:")
        print(cm)
    except Exception as e:
        print(f"\n[LƯU Ý] Không trích xuất được Confusion Matrix: {e}")
    
    metrics = {
        "auc": float(auc),
        "aucpr": float(aucpr),
        "logloss": float(logloss)
    }
    return metrics, perf

def save_best_model_and_metadata(aml: H2OAutoML, leader_model, config: dict, test_metrics: dict):
    """Lưu mô hình tốt nhất (Binary + MOJO) và xuất file metadata đầy đủ."""
    print("\n" + "=" * 70)
    print("BƯỚC 7: LƯU MÔ HÌNH VÀ METADATA")
    print("=" * 70)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Lưu Binary Model của H2O
    binary_path = h2o.save_model(model=leader_model, path=MODELS_DIR, force=True)
    print(f"[OK] Đã lưu mô hình nhị phân H2O tại:\n     {binary_path}")
    
    # 2. Lưu MOJO (Model Object, Optimized) để nhúng triển khai nhanh
    mojo_path = None
    try:
        mojo_path = leader_model.save_mojo(path=MODELS_DIR, force=True)
        print(f"[OK] Đã lưu file MOJO độc lập tại:\n     {mojo_path}")
    except Exception as e:
        print(f"[LƯU Ý] Không xuất được MOJO cho {leader_model.algo}: {e}")
        
    # 3. Trích xuất mức độ quan trọng của đặc trưng (Variable Importance)
    varimp_list = []
    try:
        vi = leader_model.varimp(use_pandas=True)
        if vi is not None:
            print("\nTop 10 đặc trưng quan trọng nhất (Variable Importance):")
            top10 = vi.head(10)[["variable", "relative_importance", "scaled_importance", "percentage"]]
            print(top10.to_string(index=False))
            varimp_list = vi.to_dict(orient="records")
    except Exception:
        # Nếu leader là StackedEnsemble, lấy mô hình đơn lẻ tốt nhất (GBM/DRF) từ leaderboard để lấy varimp
        try:
            print("\n[INFO] Leader là Stacked Ensemble. Đang trích xuất Variable Importance từ mô hình cơ sở tốt nhất...")
            lb_df = aml.leaderboard.as_data_frame()
            # Tìm mô hình cơ sở không phải StackedEnsemble
            base_model_ids = [m for m in lb_df["model_id"].tolist() if "StackedEnsemble" not in m]
            if base_model_ids:
                best_base_id = base_model_ids[0]
                best_base_model = h2o.get_model(best_base_id)
                vi = best_base_model.varimp(use_pandas=True)
                if vi is not None:
                    print(f"Top 10 đặc trưng quan trọng từ mô hình cơ sở '{best_base_id}':")
                    top10 = vi.head(10)[["variable", "relative_importance", "scaled_importance", "percentage"]]
                    print(top10.to_string(index=False))
                    varimp_list = vi.to_dict(orient="records")
        except Exception as e2:
            print(f"[LƯU Ý] Không trích xuất được varimp: {e2}")
        
    # 4. Lưu metadata phục vụ ứng dụng Streamlit và Báo cáo
    lb_records = aml.leaderboard.as_data_frame().to_dict(orient="records")
    
    metadata = {
        "model_id": leader_model.model_id,
        "algorithm": leader_model.algo,
        "binary_model_path": binary_path,
        "mojo_path": mojo_path,
        "target": config["target"],
        "features": config["features"],
        "n_features": len(config["features"]),
        "categorical_features": config.get("categorical_features", []),
        "numerical_features": config.get("numerical_features", []),
        "test_metrics": test_metrics,
        "leaderboard": lb_records,
        "variable_importance_top10": varimp_list[:10] if varimp_list else []
    }
    
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"[OK] Đã lưu file metadata chi tiết tại:\n     {METADATA_PATH}")

def main():
    print("BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN H2O AUTOML (HOTEL CANCELLATION)\n")
    # 1. Khởi tạo H2O
    init_h2o()
    
    # 2. Đọc cấu hình features
    config = load_feature_config(CONFIG_PATH)
    
    # 3. Nạp tập Train & Test
    train_hf, test_hf = load_and_prepare_frames(TRAIN_CSV, TEST_CSV, config)
    
    # 4. Huấn luyện H2O AutoML (chạy thử 60s để kiểm tra toàn bộ pipeline)
    aml = train_automl_pipeline(
        train_hf=train_hf,
        features=config["features"],
        target=config["target"],
        max_runtime_secs=60,
        seed=config.get("random_seed", 42)
    )
    
    # 5. In Leaderboard
    display_leaderboard(aml)
    
    # 6. Đánh giá mô hình Leader trên tập Test
    leader = aml.leader
    test_metrics, perf = evaluate_leader_model(leader, test_hf, config["target"])
    
    # 7. Lưu mô hình và metadata
    save_best_model_and_metadata(aml, leader, config, test_metrics)
    
    print("\n" + "=" * 70)
    print("HOÀN THÀNH BƯỚC HUẤN LUYỆN H2O AUTOML THÀNH CÔNG!")
    print("=" * 70)

if __name__ == "__main__":
    main()
