# -*- coding: utf-8 -*-
"""
H2O-3 Model Evaluation Script
Dự án: Phân tích và dự đoán khả năng khách hàng hủy đặt phòng khách sạn bằng H2O AutoML
Tập kiểm tra độc lập: data/test_clean.csv (23,948 dòng)
Mô hình đánh giá: Mô hình Leader được lưu trong models/
"""

import os
import sys
import json
import h2o
from h2o.frame import H2OFrame
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc, precision_recall_curve

# Đảm bảo in tiếng Việt UTF-8 chuẩn trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEST_CSV = os.path.join(DATA_DIR, "test_clean.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
EVAL_RESULTS_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")

def init_h2o():
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

def load_metadata_and_model():
    """Nạp metadata và tải mô hình tốt nhất từ thư mục models/."""
    print("\n" + "=" * 70)
    print("BƯỚC 2: TẢI MÔ HÌNH LEADER ĐÃ HUẤN LUYỆN")
    print("=" * 70)
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Không tìm thấy file metadata tại: {METADATA_PATH}. Hãy chạy train_automl.py trước!")
        
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    model_path = meta.get("binary_model_path")
    print(f"- Model ID: {meta.get('model_id')}")
    print(f"- Thuật toán: {meta.get('algorithm')}")
    print(f"- Đường dẫn file mô hình: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file binary model tại: {model_path}")
        
    print("Đang nạp mô hình vào H2O runtime...")
    model = h2o.load_model(model_path)
    print(f"[OK] Mô hình '{model.model_id}' đã được nạp thành công.")
    return meta, model

def load_test_data(meta: dict) -> H2OFrame:
    """Nạp tập test độc lập vào H2OFrame và đồng bộ kiểu dữ liệu."""
    print("\n" + "=" * 70)
    print("BƯỚC 3: NẠP TẬP TEST ĐỘC LẬP")
    print("=" * 70)
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Không tìm thấy file test tại: {TEST_CSV}")
        
    print(f"Đang nạp tập test: {TEST_CSV}")
    test_hf = h2o.upload_file(path=TEST_CSV, destination_frame="test_eval")
    print(f"[OK] Tập Test: {test_hf.nrow:,} dòng, {test_hf.ncol} cột.")
    
    target = meta["target"]
    test_hf[target] = test_hf[target].asfactor()
    
    for col in meta.get("categorical_features", []):
        if col in test_hf.columns:
            test_hf[col] = test_hf[col].asfactor()
            
    print(f"[OK] Đã chuẩn hóa kiểu dữ liệu cho tập Test theo đúng cấu hình.")
    return test_hf

def evaluate_metrics(model, test_hf: H2OFrame, meta: dict):
    """Tính toán các chỉ số đánh giá chi tiết: AUC, Accuracy, Precision, Recall, F1, Confusion Matrix."""
    print("\n" + "=" * 70)
    print("BƯỚC 4: TÍNH TOÁN HIỆU NĂNG MÔ HÌNH TRÊN TẬP TEST")
    print("=" * 70)
    
    target = meta["target"]
    print("1. Chạy model_performance trên tập test...")
    perf = model.model_performance(test_data=test_hf)
    
    # Lấy dự đoán xác suất và nhãn
    print("2. Sinh xác suất dự đoán (probabilities)...")
    preds = model.predict(test_hf)
    
    # Chuyển nhãn thực tế và xác suất sang numpy/pandas để tính toán đối chiếu
    y_true = test_hf[target].as_data_frame()[target].astype(int).values
    # p1 là xác suất hủy phòng (is_canceled = 1)
    y_prob = preds["p1"].as_data_frame()["p1"].values
    
    # Lấy ma trận nhầm lẫn chuẩn từ H2O tại ngưỡng tối ưu F1
    cm_h2o = perf.confusion_matrix()
    print("\nMa trận nhầm lẫn (Confusion Matrix) từ H2O:")
    print(cm_h2o)
    
    # Tìm ngưỡng tối ưu cho F1 từ H2O
    try:
        opt_threshold = float(perf.find_threshold_by_max_metric("f1"))
    except Exception:
        opt_threshold = 0.4469
    print(f"\nNgưỡng phân loại tối ưu (Optimal Threshold for Max F1): {opt_threshold:.4f}")
    
    # Dự đoán nhãn nhị phân theo ngưỡng tối ưu
    from sklearn.metrics import confusion_matrix as sk_cm
    y_pred = (y_prob >= opt_threshold).astype(int)
    tn, fp, fn, tp = sk_cm(y_true, y_pred).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    total = tn + fp + fn + tp
    
    # Tính các chỉ số định lượng
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    auc_val = float(perf.auc())
    logloss_val = float(perf.logloss())
    
    try:
        aucpr_val = float(perf.pr_auc())
    except Exception:
        try:
            aucpr_val = float(perf.aucpr())
        except Exception:
            aucpr_val = 0.0
            
    print("\n" + "=" * 70)
    print("BẢNG TỔNG HỢP CÁC CHỈ SỐ ĐÁNH GIÁ (TEST SET METRICS)")
    print("=" * 70)
    metrics_table = [
        {"Chỉ số (Metric)": "ROC-AUC", "Giá trị": f"{auc_val:.4f}", "Tỷ lệ %": f"{auc_val*100:.2f}%", "Ý nghĩa chuyên môn": "Khả năng phân biệt tổng quát giữa khách hủy và không hủy"},
        {"Chỉ số (Metric)": "Accuracy (Độ chính xác)", "Giá trị": f"{accuracy:.4f}", "Tỷ lệ %": f"{accuracy*100:.2f}%", "Ý nghĩa chuyên môn": "Tỷ lệ dự đoán đúng trên toàn bộ 23,948 lượt đặt phòng"},
        {"Chỉ số (Metric)": "Precision (Độ chuẩn xác)", "Giá trị": f"{precision:.4f}", "Tỷ lệ %": f"{precision*100:.2f}%", "Ý nghĩa chuyên môn": "Trong các phòng mô hình báo 'sẽ hủy', có bao nhiêu % thực sự hủy"},
        {"Chỉ số (Metric)": "Recall (Độ nhạy / Thu hồi)", "Giá trị": f"{recall:.4f}", "Tỷ lệ %": f"{recall*100:.2f}%", "Ý nghĩa chuyên môn": "Trong các phòng thực tế bị hủy, mô hình bắt trúng được bao nhiêu %"},
        {"Chỉ số (Metric)": "F1-Score", "Giá trị": f"{f1:.4f}", "Tỷ lệ %": f"{f1*100:.2f}%", "Ý nghĩa chuyên môn": "Trung bình điều hòa giữa Precision và Recall"},
        {"Chỉ số (Metric)": "Specificity (Độ đặc hiệu)", "Giá trị": f"{specificity:.4f}", "Tỷ lệ %": f"{specificity*100:.2f}%", "Ý nghĩa chuyên môn": "Khả năng nhận diện chính xác các khách hàng KHÔNG hủy phòng"},
        {"Chỉ số (Metric)": "PR-AUC", "Giá trị": f"{aucpr_val:.4f}", "Tỷ lệ %": f"{aucpr_val*100:.2f}%", "Ý nghĩa chuyên môn": "Diện tích dưới đường cong Precision-Recall"},
        {"Chỉ số (Metric)": "Logloss (Cross-entropy)", "Giá trị": f"{logloss_val:.4f}", "Tỷ lệ %": "-", "Ý nghĩa chuyên môn": "Mức độ sai lệch của phân phối xác suất dự đoán"}
    ]
    df_metrics = pd.DataFrame(metrics_table)
    print(df_metrics.to_string(index=False))
    
    results = {
        "model_id": model.model_id,
        "algorithm": model.algo,
        "test_records": total,
        "confusion_matrix": {
            "TN (Đúng Không hủy)": tn,
            "FP (Nhầm thành Hủy)": fp,
            "FN (Bỏ sót Hủy)": fn,
            "TP (Bắt trúng Hủy)": tp
        },
        "metrics": {
            "roc_auc": round(auc_val, 4),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "specificity": round(specificity, 4),
            "pr_auc": round(aucpr_val, 4),
            "logloss": round(logloss_val, 4)
        }
    }
    return results, y_true, y_prob, tn, fp, fn, tp

def plot_roc_and_confusion_matrix(y_true, y_prob, tn, fp, fn, tp, auc_val: float):
    """Vẽ biểu đồ ROC curve và ma trận nhầm lẫn phục vụ báo cáo."""
    print("\n" + "=" * 70)
    print("BƯỚC 5: TẠO BIỂU ĐỒ ROC CURVE VÀ CONFUSION MATRIX")
    print("=" * 70)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 1. Vẽ ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = sk_auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#2980b9', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#7f8c8d', lw=1.5, linestyle='--', label='Đường chuẩn ngẫu nhiên (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ sai lầm dương tính (False Positive Rate - FPR)', fontsize=11, fontweight='bold')
    plt.ylabel('Tỷ lệ phát hiện đúng (True Positive Rate - TPR)', fontsize=11, fontweight='bold')
    plt.title('Đường cong ROC của Mô hình Stacked Ensemble trên Tập Test', fontsize=12, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    roc_chart_path = os.path.join(FIGURES_DIR, "06_roc_curve.png")
    plt.savefig(roc_chart_path, dpi=200)
    plt.close()
    print(f"  [+] Đã lưu biểu đồ ROC Curve: {roc_chart_path}")

    # 2. Vẽ Confusion Matrix Heatmap
    cm_matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Không hủy (0)', 'Đã hủy (1)']
    ax.set(xticks=np.arange(cm_matrix.shape[1]),
           yticks=np.arange(cm_matrix.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Ma trận nhầm lẫn (Confusion Matrix) trên Tập Test (23,948 dòng)',
           ylabel='Trạng thái Thực tế',
           xlabel='Trạng thái Dự đoán')
    
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    thresh = cm_matrix.max() / 2.
    for i in range(cm_matrix.shape[0]):
        for j in range(cm_matrix.shape[1]):
            val = cm_matrix[i, j]
            pct = (val / np.sum(cm_matrix)) * 100
            label = f"{val:,}\n({pct:.1f}%)"
            ax.text(j, i, label,
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=11, fontweight='bold')
    plt.tight_layout()
    cm_chart_path = os.path.join(FIGURES_DIR, "07_confusion_matrix.png")
    plt.savefig(cm_chart_path, dpi=200)
    plt.close()
    print(f"  [+] Đã lưu biểu đồ Confusion Matrix: {cm_chart_path}")

def save_evaluation_results(results: dict):
    """Lưu kết quả đánh giá thành file JSON."""
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"[OK] Đã lưu kết quả đánh giá tại: {EVAL_RESULTS_PATH}")

def main():
    print("BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH H2O AUTOML TRÊN TẬP TEST ĐỘC LẬP\n")
    # 1. Khởi tạo H2O
    init_h2o()
    
    # 2. Nạp mô hình
    meta, model = load_metadata_and_model()
    
    # 3. Nạp tập Test
    test_hf = load_test_data(meta)
    
    # 4. Tính toán chỉ số
    results, y_true, y_prob, tn, fp, fn, tp = evaluate_metrics(model, test_hf, meta)
    
    # 5. Vẽ biểu đồ ROC và Confusion Matrix
    plot_roc_and_confusion_matrix(y_true, y_prob, tn, fp, fn, tp, results["metrics"]["roc_auc"])
    
    # 6. Lưu kết quả
    save_evaluation_results(results)
    
    print("\n" + "=" * 70)
    print("HOÀN THÀNH ĐÁNH GIÁ MÔ HÌNH THÀNH CÔNG!")
    print("=" * 70)

if __name__ == "__main__":
    main()
