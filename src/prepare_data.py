# -*- coding: utf-8 -*-
"""
H2O-3 Data Preparation Script
Dự án: Phân tích và dự đoán khả năng khách hàng hủy đặt phòng khách sạn bằng H2O AutoML
File nguồn: data/hotel_bookings.csv
"""

import os
import sys
import json
import h2o
from h2o.frame import H2OFrame

# Đảm bảo in tiếng Việt UTF-8 chuẩn trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_PATH = os.path.join(DATA_DIR, "hotel_bookings.csv")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "features_config.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Danh sách cột rò rỉ thông tin bắt buộc loại bỏ
LEAKAGE_COLUMNS = ["reservation_status", "reservation_status_date"]

# Danh sách biến phân loại cần ép kiểu factor
CATEGORICAL_COLUMNS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type"
]

def init_h2o_session():
    """Khởi tạo hoặc kết nối cụm H2O-3."""
    print("=" * 70)
    print("BƯỚC 1: KHỞI TẠO HOẶC KẾT NỐI H2O CLUSTER")
    print("=" * 70)
    try:
        h2o.init(ip="localhost", port=54321, start_h2o=False)
        print("[OK] Đã kết nối tới H2O Cluster đang hoạt động.")
    except Exception:
        print("Chưa có cluster sẵn. Đang khởi tạo cluster mới với 4GB RAM...")
        h2o.init(max_mem_size="4G", nthreads=-1)
        print("[OK] H2O Cluster đã khởi động thành công.")

def load_data(filepath: str) -> H2OFrame:
    """Nạp file dữ liệu CSV vào H2OFrame phân tán."""
    print("\n" + "=" * 70)
    print("BƯỚC 2: NẠP DỮ LIỆU TỪ CSV VÀO H2OFRAME")
    print("=" * 70)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
    
    print(f"Đang nạp file: {filepath}")
    hf = h2o.upload_file(path=filepath, destination_frame="hotel_raw")
    n_rows, n_cols = hf.dim
    print(f"[OK] Đã nạp thành công H2OFrame: {n_rows:,} dòng, {n_cols} cột.")
    return hf

def validate_target_variable(hf: H2OFrame, target_col: str = "is_canceled"):
    """Kiểm tra biến mục tiêu và xác nhận bài toán phân loại nhị phân."""
    print("\n" + "=" * 70)
    print("BƯỚC 3: KIỂM TRA BIẾN MỤC TIÊU (TARGET)")
    print("=" * 70)
    if target_col not in hf.columns:
        raise ValueError(f"Không tìm thấy cột mục tiêu '{target_col}' trong dữ liệu!")
    
    print(f"Cột mục tiêu: '{target_col}'")
    # Ép kiểu target sang factor để H2O nhận diện Binary Classification
    hf[target_col] = hf[target_col].asfactor()
    
    levels = hf[target_col].levels()[0]
    print(f"- Các nhãn (Levels) của biến mục tiêu: {levels}")
    
    if len(levels) != 2:
        raise ValueError(f"Biến mục tiêu có {len(levels)} nhãn, không phải bài toán phân loại nhị phân!")
    
    print("[XÁC NHẬN] Biến mục tiêu có chính xác 2 nhãn ['0', '1'] -> Bài toán Phân loại nhị phân (Binary Classification).")
    
    # Thống kê phân bố nhãn
    table_df = hf[target_col].table().as_data_frame()
    total = table_df["Count"].sum()
    table_df["Tỷ lệ (%)"] = (table_df["Count"] / total) * 100
    label_map = {0: "Không hủy (Check-out)", 1: "Đã hủy (Canceled)", "0": "Không hủy (Check-out)", "1": "Đã hủy (Canceled)"}
    table_df["Ý nghĩa"] = table_df[target_col].map(label_map)
    print("\nPhân bố nhãn trên toàn bộ dữ liệu:")
    print(table_df.to_string(index=False))

def preprocess_and_feature_selection(hf: H2OFrame):
    """Xử lý kiểu dữ liệu, làm sạch và lựa chọn danh sách đặc trưng đầu vào."""
    print("\n" + "=" * 70)
    print("BƯỚC 4: TIỀN XỬ LÝ & LỰA CHỌN ĐẶC TRƯNG ĐẦU VÀO")
    print("=" * 70)
    
    # 1. Xử lý giá trị thiếu (Missing Values Imputation) trực tiếp trên H2OFrame
    print("1. Xử lý giá trị thiếu (không xóa dòng dữ liệu):")
    
    # children: thiếu 4 giá trị -> điền 0
    na_children = int(hf["children"].isna().sum())
    if na_children > 0:
        print(f"   - Điền 0 cho {na_children} giá trị thiếu trong cột 'children'.")
        hf["children"] = hf["children"].isna().ifelse(0, hf["children"])
    
    # agent: thiếu 16,340 giá trị -> điền 0 (đại diện không qua đại lý)
    na_agent = int(hf["agent"].isna().sum())
    if na_agent > 0:
        print(f"   - Điền 0 cho {na_agent:,} giá trị thiếu trong cột 'agent' (khách tự đặt lẻ).")
        hf["agent"] = hf["agent"].isna().ifelse(0, hf["agent"])
    
    # company: 94.3% rỗng -> tạo đặc trưng nhị phân 'has_company' (1: có công ty bảo trợ, 0: không)
    print("   - Tạo đặc trưng nhị phân 'has_company' thay thế cho cột 'company' (tránh 94% NA).")
    hf["has_company"] = hf["company"].isna().ifelse(0, 1).asfactor()
    
    # 2. Ép kiểu biến phân loại (Categorical / Factor)
    print("\n2. Ép kiểu dữ liệu danh mục (Categorical / Factor) cho H2O:")
    for col in CATEGORICAL_COLUMNS:
        if col in hf.columns:
            hf[col] = hf[col].asfactor()
            print(f"   - '{col}': {len(hf[col].levels()[0])} danh mục")
    
    # 3. Lựa chọn danh sách đặc trưng (Features Selection)
    target = "is_canceled"
    excluded_columns = [target] + LEAKAGE_COLUMNS + ["company"]  # Đã thay 'company' bằng 'has_company'
    
    print("\n3. Kiểm tra các cột bị loại bỏ:")
    for col in excluded_columns:
        if col in hf.columns:
            reason = "Biến mục tiêu" if col == target else ("Rò rỉ dữ liệu (Data Leakage)" if col in LEAKAGE_COLUMNS else "Đã thay bằng đặc trưng has_company")
            print(f"   - Loại bỏ '{col}': {reason}")
            
    features = [col for col in hf.columns if col not in excluded_columns]
    
    print(f"\n[TỔNG KẾT] Đã chọn {len(features)} đặc trưng đầu vào hợp lệ.")
    print(f"Danh sách đặc trưng ({len(features)}):")
    for i, feat in enumerate(features, 1):
        col_type = hf.types[feat]
        print(f"   {i:>2}. {feat:32s} (Kiểu H2O: {col_type})")
        
    return features, target

def split_and_validate_data(hf: H2OFrame, features: list, target: str, seed: int = 42):
    """Chia tập Train (80%) và Test (20%) bằng H2OFrame.split_frame có seed."""
    print("\n" + "=" * 70)
    print("BƯỚC 5: PHÂN CHIA TẬP TRAIN (80%) VÀ TEST (20%)")
    print("=" * 70)
    
    print(f"Thực hiện chia dữ liệu theo tỷ lệ 80/20 với seed={seed} (đảm bảo tái lập)...")
    splits = hf.split_frame(ratios=[0.8], seed=seed, destination_frames=["hotel_train", "hotel_test"])
    train_hf = splits[0]
    test_hf = splits[1]
    
    total_rows = hf.nrow
    train_rows = train_hf.nrow
    test_rows = test_hf.nrow
    
    print(f"- Tổng số dòng ban đầu : {total_rows:,}")
    print(f"- Số dòng tập TRAIN (80%): {train_rows:,} ({train_rows/total_rows*100:.2f}%)")
    print(f"- Số dòng tập TEST  (20%): {test_rows:,} ({test_rows/total_rows*100:.2f}%)")
    
    # Kiểm tra bảo toàn dữ liệu và tính rời rạc
    if train_rows + test_rows != total_rows:
        raise ValueError(f"Lỗi: Tổng số dòng train ({train_rows}) + test ({test_rows}) không bằng tổng ({total_rows})!")
    print("[OK] Tập Train và Test rời rạc hoàn toàn, bảo toàn 100% số lượng bản ghi.")
    
    # Kiểm tra phân bố nhãn is_canceled trên cả 2 tập
    print("\nKiểm tra phân bố nhãn trên tập TRAIN:")
    train_dist = train_hf[target].table().as_data_frame()
    train_dist["Tỷ lệ (%)"] = (train_dist["Count"] / train_rows) * 100
    print(train_dist.to_string(index=False))
    
    print("\nKiểm tra phân bố nhãn trên tập TEST:")
    test_dist = test_hf[target].table().as_data_frame()
    test_dist["Tỷ lệ (%)"] = (test_dist["Count"] / test_rows) * 100
    print(test_dist.to_string(index=False))
    
    return train_hf, test_hf

def save_configuration_and_splits(train_hf: H2OFrame, test_hf: H2OFrame, features: list, target: str, seed: int = 42):
    """Lưu lại danh sách features, metadata và các tập dữ liệu để tái sử dụng."""
    print("\n" + "=" * 70)
    print("BƯỚC 6: LƯU CẤU HÌNH & XUẤT TẬP DỮ LIỆU")
    print("=" * 70)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    
    # Phân loại biến số và biến danh mục
    categorical_features = [f for f in features if train_hf.types[f] == "enum"]
    numerical_features = [f for f in features if train_hf.types[f] in ["int", "real"]]
    
    config = {
        "project_name": "Hotel Booking Cancellation Prediction with H2O AutoML",
        "target": target,
        "features": features,
        "n_features": len(features),
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "random_seed": seed,
        "train_rows": train_hf.nrow,
        "test_rows": test_hf.nrow,
        "total_rows": train_hf.nrow + test_hf.nrow,
        "leakage_columns_excluded": LEAKAGE_COLUMNS,
        "train_frame_id": "hotel_train",
        "test_frame_id": "hotel_test"
    }
    
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"[OK] Đã lưu cấu hình đặc trưng tại: {CONFIG_PATH}")
    
    # Xuất file CSV sạch cho train và test để có thể nạp nhanh bất kỳ lúc nào
    train_csv = os.path.join(DATA_DIR, "train_clean.csv")
    test_csv = os.path.join(DATA_DIR, "test_clean.csv")
    print(f"Đang lưu tập Train ({train_hf.nrow:,} dòng) sang: {train_csv}")
    h2o.download_csv(train_hf, filename=train_csv)
    print(f"Đang lưu tập Test ({test_hf.nrow:,} dòng) sang: {test_csv}")
    h2o.download_csv(test_hf, filename=test_csv)
    print(f"[OK] Đã lưu 2 file dữ liệu sạch sẵn sàng cho bước huấn luyện H2O AutoML.")

def main():
    print("BẮT ĐẦU QUY TRÌNH CHUẨN BỊ DỮ LIỆU CHO H2O AUTOML\n")
    # 1. Khởi tạo H2O
    init_h2o_session()
    
    # 2. Nạp dữ liệu
    hf = load_data(DATA_PATH)
    
    # 3. Kiểm tra biến mục tiêu
    validate_target_variable(hf, target_col="is_canceled")
    
    # 4. Tiền xử lý & chọn đặc trưng
    features, target = preprocess_and_feature_selection(hf)
    
    # 5. Phân chia Train/Test 80/20
    train_hf, test_hf = split_and_validate_data(hf, features, target, seed=42)
    
    # 6. Lưu cấu hình và tập dữ liệu
    save_configuration_and_splits(train_hf, test_hf, features, target, seed=42)
    
    print("\n" + "=" * 70)
    print("HOÀN THÀNH BƯỚC CHUẨN BỊ DỮ LIỆU THÀNH CÔNG!")
    print("Tất cả cấu hình và tập dữ liệu đã sẵn sàng cho bước huấn luyện H2O AutoML.")
    print("=" * 70)

if __name__ == "__main__":
    main()
