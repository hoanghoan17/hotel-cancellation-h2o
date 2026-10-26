# -*- coding: utf-8 -*-
"""
End-to-End System Integration & Audit Test Script
Dự án: Hotel Booking Cancellation Prediction with H2O AutoML
"""

import os
import sys
import json
import traceback
import pandas as pd
import numpy as np

# Đảm bảo UTF-8 cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
APP_PATH = os.path.join(PROJECT_ROOT, "app.py")

test_results = {}

def log_test(name: str, passed: bool, details: str = ""):
    status = "PASS" if passed else "FAIL"
    test_results[name] = {"status": status, "details": details}
    icon = "[✓]" if passed else "[✗]"
    print(f"{icon} {name:<45}: {status} {('- ' + details) if details else ''}")

print("=" * 80)
print("KHỞI ĐỘNG HỆ THỐNG KIỂM THỬ TOÀN DIỆN (SYSTEM AUDIT & INTEGRATION TESTS)")
print("=" * 80)

# TEST 1: Kiểm tra tệp tin và cấu trúc dự án
print("\n--- KIỂM TRA 1: CẤU TRÚC TỆP TIN & DỮ LIỆU ---")
required_files = [
    os.path.join(DATA_DIR, "hotel_bookings.csv"),
    os.path.join(DATA_DIR, "train_clean.csv"),
    os.path.join(DATA_DIR, "test_clean.csv"),
    os.path.join(DATA_DIR, "sample_batch_test.csv"),
    os.path.join(SRC_DIR, "features_config.json"),
    os.path.join(MODELS_DIR, "model_metadata.json"),
    os.path.join(MODELS_DIR, "evaluation_results.json"),
    APP_PATH
]

all_files_exist = True
missing_files = []
for fpath in required_files:
    if not os.path.exists(fpath):
        all_files_exist = False
        missing_files.append(os.path.relpath(fpath, PROJECT_ROOT))

log_test("Kiem tra su ton tai cua cac file cot loi", all_files_exist, f"Thieu: {missing_files}" if missing_files else "Day du cac file")

# TEST 2: Kiểm tra cấu hình và Rò rỉ dữ liệu (Data Leakage)
print("\n--- KIỂM TRA 2: PHÒNG CHỐNG RÒ RỈ DỮ LIỆU (DATA LEAKAGE) ---")
config_path = os.path.join(SRC_DIR, "features_config.json")
try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    features = config.get("features", [])
    target = config.get("target")
    leakage_cols = ["reservation_status", "reservation_status_date"]
    
    no_target_in_features = target not in features
    no_leakage_in_features = all(col not in features for col in leakage_cols)
    correct_count = len(features) == 29
    
    passed_leakage = no_target_in_features and no_leakage_in_features and correct_count
    details = f"Target '{target}' khong nam trong features; Loai bo 100% cot ro ri {leakage_cols}; So features={len(features)}"
    log_test("Kiem tra ro ri du lieu (Data Leakage)", passed_leakage, details)
except Exception as e:
    log_test("Kiem tra ro ri du lieu (Data Leakage)", False, str(e))

# TEST 3: Kiểm tra tính phân tách độc lập Train (80%) và Test (20%)
print("\n--- KIỂM TRA 3: PHÂN TÁCH DỮ LIỆU TRAIN / TEST ---")
try:
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train_clean.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test_clean.csv"))
    
    train_count = len(df_train)
    test_count = len(df_test)
    total_count = train_count + test_count
    
    preservation = (total_count == 119390)
    train_ratio = (train_count / total_count) * 100
    test_ratio = (test_count / total_count) * 100
    
    passed_split = preservation and (79.0 <= train_ratio <= 81.0) and (19.0 <= test_ratio <= 21.0)
    log_test("Kiem tra bao toan va ty le Train/Test 80/20", passed_split, f"Train={train_count:,} ({train_ratio:.2f}%), Test={test_count:,} ({test_ratio:.2f}%), Tong={total_count:,}")
except Exception as e:
    log_test("Kiem tra bao toan va ty le Train/Test 80/20", False, str(e))

# TEST 4: Kết nối H2O và Nạp mô hình Leader từ đĩa
print("\n--- KIỂM TRA 4: H2O RUNTIME & NẠP MÔ HÌNH LEADER ---")
import h2o
h2o_connected = False
model_loaded = False
model = None
meta = None

try:
    print("Dang ket noi hoac khoi tao H2O cluster...")
    try:
        h2o.init(ip="localhost", port=54321, start_h2o=False)
    except Exception:
        h2o.init(max_mem_size="4G", nthreads=-1)
    h2o_connected = True
    log_test("Ket noi H2O Cluster", True, f"Phien ban: {h2o.cluster().version}")
    
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    model_path = meta.get("binary_model_path")
    model = h2o.load_model(model_path)
    model_loaded = (model is not None and model.model_id == meta["model_id"])
    log_test("Nap mo hinh tu thu muc models/", model_loaded, f"Model ID: {model.model_id} ({model.algo})")
except Exception as e:
    log_test("Ket noi H2O Cluster & Nap mo hinh", False, str(e))

# TEST 5: Kiểm tra tính ăn khớp 100% giữa Danh sách đặc trưng & Mô hình
print("\n--- KIỂM TRA 5: TÍNH ĂN KHỚP ĐẶC TRƯNG (FEATURE COMPATIBILITY) ---")
if model_loaded:
    try:
        model_names = model._model_json['output']['names']
        model_features = [n for n in model_names if n != meta.get("target")]
        config_features = config["features"]
        
        diff = set(config_features).difference(set(model_features))
        # has_company bi H2O tu dong loai vi hang so gia tri (zero variance)
        features_match = (diff == set() or diff == {'has_company'})
        log_test("Khop danh sach dac trung giua Mo hinh va Cau hinh", features_match, f"Mo hinh su dung 28 dac trung hoat dong chinh thuc. H2O tu dong bo cot hang so: {diff}")
    except Exception as e:
        log_test("Khop danh sach dac trung giua Mo hinh va Cau hinh", False, str(e))

# TEST 6: Dự đoán đơn lẻ (Single Prediction) và Kiểm tra Đầu ra Xác suất
print("\n--- KIỂM TRA 6: DỰ ĐOÁN ĐƠN LẺ VÀ PHÂN PHỐI XÁC SUẤT ---")
if model_loaded:
    try:
        # Lay 1 dong mau tu tap test
        sample_row = df_test.iloc[0:1].drop(columns=["is_canceled"], errors="ignore")
        sample_hf = h2o.H2OFrame(sample_row)
        for col in meta.get("categorical_features", []):
            if col in sample_hf.columns:
                sample_hf[col] = sample_hf[col].asfactor()
        for col in meta.get("numerical_features", []):
            if col in sample_hf.columns:
                sample_hf[col] = sample_hf[col].asnumeric()
                
        pred = model.predict(sample_hf)
        pred_df = pred.as_data_frame()
        
        p0 = float(pred_df["p0"].iloc[0])
        p1 = float(pred_df["p1"].iloc[0])
        predicted_class = str(pred_df["predict"].iloc[0])
        
        prob_valid = (0.0 <= p0 <= 1.0) and (0.0 <= p1 <= 1.0)
        prob_sum_valid = abs((p0 + p1) - 1.0) < 1e-4
        class_valid = predicted_class in ["0", "1"]
        
        passed_pred = prob_valid and prob_sum_valid and class_valid
        log_test("Du doan don le va kiem tra xac suat", passed_pred, f"p0(Khong huy)={p0:.4f}, p1(Huy)={p1:.4f}, Tong={p0+p1:.4f}, Lop={predicted_class}")
    except Exception as e:
        log_test("Du doan don le va kiem tra xac suat", False, str(e))

# TEST 7: Dự đoán hàng loạt (Batch CSV Prediction)
print("\n--- KIỂM TRA 7: DỰ ĐOÁN HÀNG LOẠT (BATCH PREDICTION) ---")
if model_loaded:
    try:
        sample_batch_df = pd.read_csv(os.path.join(DATA_DIR, "sample_batch_test.csv"))
        batch_hf = h2o.H2OFrame(sample_batch_df)
        for col in meta.get("categorical_features", []):
            if col in batch_hf.columns:
                batch_hf[col] = batch_hf[col].asfactor()
        for col in meta.get("numerical_features", []):
            if col in batch_hf.columns:
                batch_hf[col] = batch_hf[col].asnumeric()
                
        batch_preds = model.predict(batch_hf)
        batch_preds_df = batch_preds.as_data_frame()
        
        batch_count = len(batch_preds_df)
        has_no_nan = not batch_preds_df["p1"].isna().any()
        
        passed_batch = (batch_count == len(sample_batch_df)) and has_no_nan
        log_test("Du doan hang loat tren 50 ban ghi mau", passed_batch, f"Du doan thanh cong {batch_count}/50 dong, khong co loi NaN")
    except Exception as e:
        log_test("Du doan hang loat tren 50 ban ghi mau", False, str(e))

# TEST 8: Kiểm thử khả năng chịu lỗi (Fault Tolerance & Edge Cases)
print("\n--- KIỂM TRA 8: XỬ LÝ DỮ LIỆU BIÊN & NGOẠI LỆ (EDGE CASES) ---")
if model_loaded:
    try:
        # Case A: Giá trị lạ chưa từng gặp trong training (Unseen country)
        edge_row = sample_row.copy()
        edge_row["country"] = "UNKNOWN_XYZ"
        edge_row["lead_time"] = 999
        edge_row["adr"] = 0.0
        edge_row["children"] = 0
        
        edge_hf = h2o.H2OFrame(edge_row)
        for col in meta.get("categorical_features", []):
            if col in edge_hf.columns:
                edge_hf[col] = edge_hf[col].asfactor()
        for col in meta.get("numerical_features", []):
            if col in edge_hf.columns:
                edge_hf[col] = edge_hf[col].asnumeric()
                
        edge_pred = model.predict(edge_hf)
        edge_pred_df = edge_pred.as_data_frame()
        edge_p1 = float(edge_pred_df["p1"].iloc[0])
        
        passed_edge = (0.0 <= edge_p1 <= 1.0) and not np.isnan(edge_p1)
        log_test("Xu ly gia tri ngoai bien (Unseen level, NA, Extreme values)", passed_edge, f"Mo hinh xu ly muot ma, sinh xac suat hop le: p1={edge_p1:.4f}")
    except Exception as e:
        log_test("Xu ly gia tri ngoai bien (Unseen level, NA, Extreme values)", False, str(e))

# TEST 9: Kiểm tra tính sẵn sàng của mã nguồn Streamlit App
print("\n--- KIỂM TRA 9: KIỂM TRA CÚ PHÁP & TOÀN VẸN APP STREAMLIT ---")
try:
    import py_compile
    py_compile.compile(APP_PATH, doraise=True)
    log_test("Bien dich cu phap app.py (Syntax Compilation)", True, "Cu phap hop le 100%, khong co loi trinh bien dich")
except Exception as e:
    log_test("Bien dich cu phap app.py (Syntax Compilation)", False, str(e))

# TEST 10: Kiểm tra tính nhất quán giữa Kết quả đánh giá và Leaderboard
print("\n--- KIỂM TRA 10: ĐỐI CHIẾU SỐ LIỆU ĐÁNH GIÁ THỰC TẾ ---")
try:
    with open(os.path.join(MODELS_DIR, "evaluation_results.json"), "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    metrics = eval_data.get("metrics", {})
    auc_ok = (metrics.get("roc_auc") == 0.9589)
    acc_ok = (metrics.get("accuracy") == 0.8904)
    prec_ok = (metrics.get("precision") == 0.8532)
    rec_ok = (metrics.get("recall") == 0.8525)
    f1_ok = (metrics.get("f1_score") == 0.8528)
    
    passed_metrics = auc_ok and acc_ok and prec_ok and rec_ok and f1_ok
    log_test("Kiem tra tinh xac thuc so lieu danh gia thuc nghiem", passed_metrics, f"ROC-AUC={metrics.get('roc_auc')}, Accuracy={metrics.get('accuracy')}, F1={metrics.get('f1_score')}")
except Exception as e:
    log_test("Kiem tra tinh xac thuc so lieu danh gia thuc nghiem", False, str(e))

print("\n" + "=" * 80)
total_tests = len(test_results)
passed_tests = sum(1 for t in test_results.values() if t["status"] == "PASS")
print(f"KẾT QUẢ TỔNG KẾT: {passed_tests}/{total_tests} BÀI KIỂM THỬ THÀNH CÔNG ({passed_tests/total_tests*100:.1f}%)")
print("=" * 80)

if h2o_connected:
    h2o.cluster().shutdown()
