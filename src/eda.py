# -*- coding: utf-8 -*-
"""
H2O-3 Exploratory Data Analysis (EDA) Script
De tai: Phan tich va du doan kha nang khach hang huy dat phong khach san bang H2O AutoML
File du lieu: data/hotel_bookings.csv
"""

import os
import sys

# Dam bao terminal Windows in dung tieng Viet UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

import h2o
from h2o.frame import H2OFrame
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Chay che do khong can man hinh GUI
import matplotlib.pyplot as plt

# Duong dan thu muc goc cua du an
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "hotel_bookings.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

def step1_check_data_file(filepath: str) -> bool:
    """Kiem tra su ton tai va kha nang truy cap file du lieu CSV."""
    print("=" * 70)
    print("BƯỚC 1: KIỂM TRA FILE DỮ LIỆU CSV")
    print("=" * 70)
    if not os.path.exists(filepath):
        print(f"[LỖI] Khong tim thay file du lieu tai: {filepath}")
        return False
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[OK] File du lieu ton tai: {filepath}")
    print(f"      Dung luong file: {size_mb:.2f} MB")
    return True

def step2_init_h2o():
    """Khoi tao hoac ket noi den H2O Cluster cuc bo."""
    print("\n" + "=" * 70)
    print("BƯỚC 2: KHỞI TẠO / KẾT NỐI H2O CLUSTER")
    print("=" * 70)
    try:
        # Thu ket noi den cluster co san truoc
        print("Kiem tra H2O Cluster hien tai...")
        h2o.init(ip="localhost", port=54321, start_h2o=False)
        print("[OK] Da ket noi den H2O Cluster dang chay san.")
    except Exception:
        print("Chua co cluster chay san. Dang khoi tao H2O Cluster cuc bo moi (max_mem_size='4G')...")
        h2o.init(max_mem_size="4G", nthreads=-1)
        print("[OK] H2O Cluster da duoc khoi tao thanh cong.")
    
    print("      - Trạng thái cụm:")
    try:
        h2o.cluster().show_status()
    except Exception:
        pass
    print("      - H2O Flow URL: http://localhost:54321")

def step3_load_h2o_frame(filepath: str) -> H2OFrame:
    """Doc du lieu tu file CSV vao H2OFrame phan tan bang h2o.upload_file."""
    print("\n" + "=" * 70)
    print("BƯỚC 3: ĐỌC DỮ LIỆU VÀO H2OFRAME")
    print("=" * 70)
    print(f"Đang truyền và phân tích file vào H2O DKV Store: {filepath}")
    # Sử dụng upload_file để truyền dữ liệu qua REST API an toàn, tránh lỗi hạn chế truy cập file của Java JVM backend
    hf = h2o.upload_file(path=filepath, destination_frame="hotel_bookings_raw")
    print(f"[OK] Dữ liệu đã được nạp thành công vào DKV Store với Frame Key: 'hotel_bookings_raw'")
    return hf

def step4_data_structure(hf: H2OFrame):
    """Hien thi so dong, so cot, danh sach ten cot va kieu du lieu tren H2OFrame."""
    print("\n" + "=" * 70)
    print("BƯỚC 4: CẤU TRÚC DỮ LIỆU TỔNG QUAN")
    print("=" * 70)
    n_rows, n_cols = hf.dim
    print(f"- Tổng số dòng (Records): {n_rows:,}")
    print(f"- Tổng số cột (Features): {n_cols}")
    
    print("\nChi tiết các cột và kiểu dữ liệu trong H2O:")
    col_types = hf.types
    na_counts = hf.nacnt()
    col_summary = []
    for i, col in enumerate(hf.columns):
        col_summary.append({
            "STT": i + 1,
            "Tên Cột": col,
            "Kiểu H2O": col_types[col],
            "Số giá trị thiếu (NA)": na_counts[i]
        })
    df_summary = pd.DataFrame(col_summary)
    print(df_summary.to_string(index=False))

def step5_check_missing_and_duplicates(hf: H2OFrame):
    """Kiem tra gia tri thieu (NA) va du lieu trung lap."""
    print("\n" + "=" * 70)
    print("BƯỚC 5: KIỂM TRA GIÁ TRỊ THIẾU VÀ DỮ LIỆU TRÙNG LẶP")
    print("=" * 70)
    
    # 1. Gia tri thieu
    n_rows = hf.nrow
    na_counts = hf.nacnt()
    missing_dict = {}
    for i, col in enumerate(hf.columns):
        na_cnt = na_counts[i]
        if na_cnt > 0:
            pct = (na_cnt / n_rows) * 100
            missing_dict[col] = (na_cnt, pct)
            
    if missing_dict:
        print("[!] Phát hiện các cột có giá trị thiếu (NA):")
        for col, (cnt, pct) in missing_dict.items():
            print(f"    + {col:25s}: {cnt:>7,} giá trị thiếu ({pct:6.2f}%)")
    else:
        print("[OK] Không có giá trị thiếu trong tập dữ liệu.")

    # 2. Kiem tra so dong trung lap
    print("\nKiểm tra các bản ghi trùng lặp (Duplicate Rows):")
    try:
        # Sử dụng pandas để tính duplicate trên toàn bộ dòng
        df_sub = hf.as_data_frame(use_pandas=True)
        dup_count = int(df_sub.duplicated().sum())
        dup_pct = (dup_count / len(df_sub)) * 100
        print(f"    + Số dòng trùng lặp hoàn toàn: {dup_count:,} dòng ({dup_pct:.2f}%)")
        print("    + Giải thích bản chất: Trong dữ liệu đặt phòng khách sạn, việc trùng lặp các thuộc tính")
        print("      (như cùng loại phòng, ngày đến, số đêm, phân khúc thị trường) là bình thường do có nhiều khách")
        print("      hàng cùng hành vi hoặc cùng đoàn. Không nên tự ý xóa bỏ toàn bộ các dòng này vì sẽ làm mất")
        print("      tính đại diện thực tế của phân phối dữ liệu gốc.")
    except Exception as e:
        print(f"    [Lưu ý] Kiểm tra duplicate: {e}")

    return missing_dict

def step6_target_distribution(hf: H2OFrame):
    """Thong ke ty le dat phong bi huy va khong bi huy (is_canceled)."""
    print("\n" + "=" * 70)
    print("BƯỚC 6: THỐNG KÊ BIẾN MỤC TIÊU 'is_canceled'")
    print("=" * 70)
    
    target_table = hf["is_canceled"].table().as_data_frame()
    total = target_table["Count"].sum()
    target_table["Ty_Le_%"] = (target_table["Count"] / total) * 100
    target_table["Y_Nghia"] = target_table["is_canceled"].map({0: "Không hủy (Check-out thành công)", 1: "Đã hủy (Canceled)"})
    print(target_table.to_string(index=False))
    
    canceled_pct = target_table.loc[target_table["is_canceled"] == 1, "Ty_Le_%"].values[0]
    not_canceled_pct = target_table.loc[target_table["is_canceled"] == 0, "Ty_Le_%"].values[0]
    print(f"\n=> Nhan xet ve do can bang du lieu:")
    print(f"   - Ty le huy phong: {canceled_pct:.2f}%")
    print(f"   - Ty le khong huy: {not_canceled_pct:.2f}%")
    print(f"   - Nhan xet: Du lieu co su chenh lech nhung khong qua mat can bang nghiem trong (Ty le khoang 37% vs 63%).")
    print(f"     Day la ty le phu hop de huan luyen Binary Classification voi cac thuat toan GBM, DRF, Ensemble trong H2O.")
    return target_table

def step7_bivariate_analysis(hf: H2OFrame):
    """Phan tich quan he giua cac thuoc tinh quan trong voi is_canceled."""
    print("\n" + "=" * 70)
    print("BƯỚC 7: PHÂN TÍCH CÁC THUỘC TÍNH QUAN TRỌNG VỚI TỶ LỆ HỦY")
    print("=" * 70)
    
    # 1. lead_time (Thoi gian tu luc dat den luc nhan phong)
    if "lead_time" in hf.columns:
        print("1. Phân tích 'lead_time' (Số ngày đặt trước):")
        mean_all = hf["lead_time"].mean()[0]
        # Loc tren H2OFrame
        lead_canceled = hf[hf["is_canceled"] == 1, "lead_time"].mean()[0]
        lead_not_canceled = hf[hf["is_canceled"] == 0, "lead_time"].mean()[0]
        median_canc = hf[hf["is_canceled"] == 1, "lead_time"].median()[0]
        median_not_canc = hf[hf["is_canceled"] == 0, "lead_time"].median()[0]
        print(f"   - Trung bình toàn bộ: {mean_all:.1f} ngày")
        print(f"   - Nhóm ĐÃ HỦY (is_canceled = 1)   : Trung bình = {lead_canceled:.1f} ngày | Trung vị = {median_canc:.1f} ngày")
        print(f"   - Nhóm KHÔNG HỦY (is_canceled = 0): Trung bình = {lead_not_canceled:.1f} ngày | Trung vị = {median_not_canc:.1f} ngày")
        print("   => Nhận xét: Khách đặt phòng trước càng xa ngày nhận phòng thì xác suất hủy càng cao (Lead time nhóm hủy gấp đôi nhóm không hủy).")

    # 2. deposit_type (Loai tien coc)
    if "deposit_type" in hf.columns:
        print("\n2. Phân tích 'deposit_type' (Chính sách tiền cọc):")
        sub_dep = hf[["deposit_type", "is_canceled"]].as_data_frame()
        dep_pivot = pd.crosstab(sub_dep["deposit_type"], sub_dep["is_canceled"])
        dep_pivot.columns = ["Khong_Huy", "Huy"]
        dep_pivot["Tong"] = dep_pivot["Khong_Huy"] + dep_pivot["Huy"]
        dep_pivot["Ty_Le_Huy_%"] = (dep_pivot["Huy"] / dep_pivot["Tong"]) * 100
        print(dep_pivot.to_string())
        print("   => Nhận xét quan trọng:")
        print("      - 'No Deposit': Tỷ lệ hủy ~28.4% (phần lớn khách thông thường).")
        print("      - 'Non Refund' (Cọc không hoàn lại): Tỷ lệ hủy cực kỳ cao (~99.4%)!")
        print("        Giải thích: Đây là nhóm khách đoàn/đại lý lữ hành đặt giữ chỗ hàng loạt từ rất sớm với giá ưu đãi cọc chết,")
        print("        nếu không bán được tour họ chấp nhận hủy/bỏ cọc.")

    # 3. market_segment (Phan khuc thi truong)
    if "market_segment" in hf.columns:
        print("\n3. Phân tích 'market_segment' (Phân khúc thị trường):")
        sub_seg = hf[["market_segment", "is_canceled"]].as_data_frame()
        seg_pivot = pd.crosstab(sub_seg["market_segment"], sub_seg["is_canceled"])
        seg_pivot.columns = ["Khong_Huy", "Huy"]
        seg_pivot["Tong"] = seg_pivot["Khong_Huy"] + seg_pivot["Huy"]
        seg_pivot["Ty_Le_Huy_%"] = (seg_pivot["Huy"] / seg_pivot["Tong"]) * 100
        seg_pivot = seg_pivot.sort_values(by="Tong", ascending=False)
        print(seg_pivot.to_string())
        print("   => Nhận xét: Nhóm 'Groups' có tỷ lệ hủy cao nhất (~61%), tiếp theo là 'Online TA' (~36.7%).")
        print("      Nhóm 'Direct' (khách tự liên hệ đặt trực tiếp) có tỷ lệ hủy thấp nhất (~15.3%).")

    # 4. previous_cancellations (So lan tung huy dat phong trong qua khu)
    if "previous_cancellations" in hf.columns:
        print("\n4. Phân tích 'previous_cancellations' (Lịch sử từng hủy trước đó):")
        sub_canc = hf[["previous_cancellations", "is_canceled"]].as_data_frame()
        canc_pivot = pd.crosstab(sub_canc["previous_cancellations"], sub_canc["is_canceled"])
        canc_pivot.columns = ["Khong_Huy", "Huy"]
        canc_pivot["Tong"] = canc_pivot["Khong_Huy"] + canc_pivot["Huy"]
        canc_pivot["Ty_Le_Huy_%"] = (canc_pivot["Huy"] / canc_pivot["Tong"]) * 100
        print(canc_pivot.head(6).to_string())
        print("   => Nhận xét: Khách hàng từng có lịch sử hủy đặt phòng trong quá khứ có nguy cơ hủy lại rất cao.")

    # 5. booking_changes (So lan thay doi thong tin dat phong)
    if "booking_changes" in hf.columns:
        print("\n5. Phân tích 'booking_changes' (Số lần thay đổi đặt phòng):")
        sub_chg = hf[["booking_changes", "is_canceled"]].as_data_frame()
        chg_pivot = pd.crosstab(sub_chg["booking_changes"], sub_chg["is_canceled"])
        chg_pivot.columns = ["Khong_Huy", "Huy"]
        chg_pivot["Tong"] = chg_pivot["Khong_Huy"] + chg_pivot["Huy"]
        chg_pivot["Ty_Le_Huy_%"] = (chg_pivot["Huy"] / chg_pivot["Tong"]) * 100
        print(chg_pivot.head(6).to_string())
        print("   => Nhận xét: Khách hàng càng có nhiều lần tương tác thay đổi đặt phòng (đổi ngày, đổi loại phòng)")
        print("      thì tỷ lệ hủy càng giảm đáng kể (chứng tỏ chuyến đi có thực và khách đang tích cực chuẩn bị).")

def step8_generate_charts(hf: H2OFrame, output_dir: str):
    """Tao cac bieu do truc quan hoa de chen vao bao cao do an."""
    print("\n" + "=" * 70)
    print("BƯỚC 8: TẠO CÁC BIỂU ĐỒ BÁO CÁO (LƯU VÀO THƯ MỤC figures/)")
    print("=" * 70)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Bieu do phan bo is_canceled
    target_df = hf["is_canceled"].table().as_data_frame()
    labels = ["Không hủy (0)", "Đã hủy (1)"]
    counts = target_df["Count"].values
    colors = ["#2ecc71", "#e74c3c"]
    
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, counts, color=colors, width=0.45, edgecolor="black", alpha=0.9)
    plt.title("Phân bố số lượng đặt phòng theo trạng thái hủy (is_canceled)", fontsize=12, fontweight='bold', pad=15)
    plt.ylabel("Số lượng đặt phòng", fontsize=11)
    for bar in bars:
        h = bar.get_height()
        pct = (h / sum(counts)) * 100
        plt.text(bar.get_x() + bar.get_width()/2., h + 1500, f"{int(h):,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight='bold')
    plt.ylim(0, max(counts) * 1.18)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "01_cancellation_distribution.png")
    plt.savefig(chart1_path, dpi=200)
    plt.close()
    print(f"  [+] Đã lưu biểu đồ 1: {chart1_path}")

    # 2. Bieu do Lead Time giua 2 nhom
    sub_lead = hf[["is_canceled", "lead_time"]].as_data_frame()
    lead_canc = sub_lead.loc[sub_lead["is_canceled"] == 1, "lead_time"].values
    lead_not_canc = sub_lead.loc[sub_lead["is_canceled"] == 0, "lead_time"].values
    
    plt.figure(figsize=(8, 5))
    bp = plt.boxplot([lead_not_canc, lead_canc], tick_labels=["Không hủy", "Đã hủy"], patch_artist=True,
                boxprops=dict(facecolor="#3498db", color='black'),
                medianprops=dict(color='yellow', linewidth=2),
                showfliers=False)
    plt.title("So sánh Lead Time (Số ngày đặt trước) giữa 2 nhóm", fontsize=12, fontweight='bold', pad=15)
    plt.ylabel("Số ngày đặt trước (Lead time)", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "02_lead_time_distribution.png")
    plt.savefig(chart2_path, dpi=200)
    plt.close()
    print(f"  [+] Đã lưu biểu đồ 2: {chart2_path}")

    # 3. Bieu do ty le huy theo Deposit Type
    sub_dep = hf[["deposit_type", "is_canceled"]].as_data_frame()
    dep_pivot = pd.crosstab(sub_dep["deposit_type"], sub_dep["is_canceled"])
    dep_pivot.columns = ["Khong_Huy", "Huy"]
    dep_rate = (dep_pivot["Huy"] / (dep_pivot["Khong_Huy"] + dep_pivot["Huy"])) * 100
    
    plt.figure(figsize=(7, 5))
    bars = plt.bar(dep_rate.index, dep_rate.values, color="#e67e22", edgecolor="black", width=0.45, alpha=0.9)
    plt.title("Tỷ lệ hủy phòng theo loại tiền cọc (Deposit Type)", fontsize=12, fontweight='bold', pad=15)
    plt.ylabel("Tỷ lệ hủy (%)", fontsize=11)
    plt.ylim(0, 115)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "03_cancellation_by_deposit_type.png")
    plt.savefig(chart3_path, dpi=200)
    plt.close()
    print(f"  [+] Đã lưu biểu đồ 3: {chart3_path}")

    # 4. Bieu do ty le huy theo Market Segment
    sub_seg = hf[["market_segment", "is_canceled"]].as_data_frame()
    seg_pivot = pd.crosstab(sub_seg["market_segment"], sub_seg["is_canceled"])
    seg_pivot.columns = ["Khong_Huy", "Huy"]
    seg_pivot["Total"] = seg_pivot["Khong_Huy"] + seg_pivot["Huy"]
    seg_pivot = seg_pivot[seg_pivot["Total"] > 100]  # Bo qua cac nhom qua it
    seg_rate = (seg_pivot["Huy"] / seg_pivot["Total"]) * 100
    seg_rate = seg_rate.sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(seg_rate.index, seg_rate.values, color="#9b59b6", edgecolor="black", width=0.55, alpha=0.9)
    plt.title("Tỷ lệ hủy phòng theo phân khúc thị trường (Market Segment)", fontsize=12, fontweight='bold', pad=15)
    plt.ylabel("Tỷ lệ hủy (%)", fontsize=11)
    plt.xticks(rotation=20, ha='right')
    plt.ylim(0, max(seg_rate.values) + 12)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, "04_cancellation_by_market_segment.png")
    plt.savefig(chart4_path, dpi=200)
    plt.close()
    print(f"  [+] Đã lưu biểu đồ 4: {chart4_path}")

    # 5. Bieu do cac cot bi thieu du lieu (Missing Values)
    na_counts = hf.nacnt()
    missing_data = []
    for i, col in enumerate(hf.columns):
        if na_counts[i] > 0:
            missing_data.append({"Column": col, "Missing_Count": na_counts[i], "Pct": (na_counts[i] / hf.nrow) * 100})
    if missing_data:
        df_miss = pd.DataFrame(missing_data).sort_values(by="Pct", ascending=True)
        plt.figure(figsize=(8, 4.5))
        bars = plt.barh(df_miss["Column"], df_miss["Pct"], color="#e74c3c", edgecolor="black", height=0.5, alpha=0.9)
        plt.title("Tỷ lệ phần trăm giá trị thiếu (Missing Values %)", fontsize=12, fontweight='bold', pad=15)
        plt.xlabel("Tỷ lệ thiếu (%)", fontsize=11)
        plt.xlim(0, 105)
        for bar in bars:
            w = bar.get_width()
            col_name = bar.get_y()
            plt.text(w + 1, bar.get_y() + bar.get_height()/2., f"{w:.2f}%", va='center', fontsize=9, fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        chart5_path = os.path.join(output_dir, "05_missing_values.png")
        plt.savefig(chart5_path, dpi=200)
        plt.close()
        print(f"  [+] Đã lưu biểu đồ 5: {chart5_path}")

def step9_preprocessing_proposal(hf: H2OFrame):
    """De xuat phuong an tien xu ly du lieu chi tiet cho cac giai doan sau."""
    print("\n" + "=" * 70)
    print("BƯỚC 9: ĐỀ XUẤT PHƯƠNG ÁN XỬ LÝ DỮ LIỆU TRƯỚC KHI HUẤN LUYỆN")
    print("=" * 70)
    proposal = """
1. CỘT CẦN LOẠI BỎ (DATA LEAKAGE - RÒ RỈ DỮ LIỆU):
   - 'reservation_status': Trạng thái phòng ('Check-Out', 'Canceled', 'No-Show'). Cột này chỉ có sau khi sự kiện đã xảy ra, nếu giữ lại mô hình sẽ bị rò rỉ đáp án 100%.
   - 'reservation_status_date': Ngày ghi nhận trạng thái cuối cùng, tương tự như trên.
   => Khuyến nghị: BẮT BUỘC LOẠI BỎ khỏi danh sách biến dự báo (features).

2. XỬ LÝ GIÁ TRỊ THIẾU (MISSING VALUES):
   - 'company': Thiếu ~94% (112,593 giá trị).
     + Lý do: Khách cá nhân chiếm đa số, không qua công ty bảo trợ.
     + Đề xuất: KHÔNG xóa cột, tạo đặc trưng nhị phân 'is_company_booking' (1 nếu có công ty, 0 nếu không), hoặc điền giá trị "None/Unknown" để mô hình cây của H2O tự khai thác.
   - 'agent': Thiếu ~13.7% (16,340 giá trị).
     + Lý do: Khách tự đặt trực tiếp không qua đại lý.
     + Đề xuất: Điền mã đại diện 0 ("No_Agent") hoặc ép sang phân loại (enum).
   - 'country': Thiếu 488 giá trị (< 0.5%).
     + Đề xuất: Điền giá trị phân loại "Unknown" hoặc giá trị xuất hiện nhiều nhất ("PRT").
   - 'children': Thiếu 4 giá trị.
     + Đề xuất: Điền 0 (vì phần lớn khách không mang trẻ em).

3. BIẾN PHÂN LOẠI (CATEGORICAL / FACTOR):
   - Các cột: 'hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
     'distribution_channel', 'reserved_room_type', 'assigned_room_type',
     'deposit_type', 'customer_type'.
   - Phương pháp trong H2O: Ép kiểu sang factor (`hf[col] = hf[col].asfactor()`).
     H2O-3 tự động áp dụng One-Hot Encoding hoặc Target Encoding phân tán tối ưu cho từng thuật toán (GBM, DRF, GLM).

4. BIẾN MỤC TIÊU:
   - 'is_canceled': Cần ép kiểu sang factor (`hf['is_canceled'] = hf['is_canceled'].asfactor()`)
     để H2O hiểu đây là bài toán Phân loại nhị phân (Binary Classification), không phải Hồi quy (Regression).
"""
    print(proposal)

def main():
    print("KHOI DONG TIEN TRINH KHAM PHA DU LIEU (EDA) VOI H2O-3")
    # 1. Kiem tra file
    if not step1_check_data_file(DATA_PATH):
        sys.exit(1)
        
    # 2. Khoi tao H2O
    step2_init_h2o()
    
    # 3. Doc du lieu
    hf = step3_load_h2o_frame(DATA_PATH)
    
    # 4. Cau truc du lieu
    step4_data_structure(hf)
    
    # 5. Missing & Duplicates
    step5_check_missing_and_duplicates(hf)
    
    # 6. Phan bo bien muc tieu
    step6_target_distribution(hf)
    
    # 7. Phan tich quan he 2 bien
    step7_bivariate_analysis(hf)
    
    # 8. Tao bieu do
    step8_generate_charts(hf, FIGURES_DIR)
    
    # 9. De xuat tien xu ly
    step9_preprocessing_proposal(hf)
    
    print("\n" + "=" * 70)
    print("HOÀN TẤT GIAI ĐOẠN KHÁM PHÁ DỮ LIỆU (EDA). CHƯA HUẤN LUYỆN MÔ HÌNH.")
    print("=" * 70)

if __name__ == "__main__":
    main()
