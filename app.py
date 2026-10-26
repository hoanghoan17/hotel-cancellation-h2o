# -*- coding: utf-8 -*-
"""
Hotel Booking Cancellation Prediction - Web Application
Môn học: Big Data
Đề tài: H2O – Nền tảng Machine Learning phân tán trên dữ liệu lớn
Framework: Streamlit & H2O-3 AutoML
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Dự đoán Hủy đặt phòng Khách sạn - H2O AutoML",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Đường dẫn thư mục
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "hotel_bookings.csv")
SAMPLE_BATCH_PATH = os.path.join(DATA_DIR, "sample_batch_test.csv")
MODELS_DIR = os.path.join(APP_DIR, "models")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
EVAL_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")
FIGURES_DIR = os.path.join(APP_DIR, "figures")
CONFIG_PATH = os.path.join(APP_DIR, "src", "features_config.json")

# -----------------------------------------------------------------------------
# Caching: Nạp H2O Model và Metadata (Chỉ nạp 1 lần, không huấn luyện lại)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_h2o_model_and_meta():
    """Khởi tạo H2O runtime và nạp mô hình Leader đã huấn luyện sẵn từ models/."""
    import h2o
    try:
        h2o.init(ip="localhost", port=54321, start_h2o=False)
    except Exception:
        h2o.init(max_mem_size="4G", nthreads=-1)
        
    if not os.path.exists(METADATA_PATH):
        return None, None
        
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    model_path = meta.get("binary_model_path")
    if not os.path.exists(model_path):
        return None, meta
        
    model = h2o.load_model(model_path)
    return model, meta

@st.cache_data(show_spinner=False)
def load_preview_data(nrows=1000):
    """Nạp dữ liệu mẫu để xem trước trên giao diện."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, nrows=nrows)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_eval_results():
    """Nạp kết quả đánh giá đã lưu."""
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# -----------------------------------------------------------------------------
# CSS Styling hiện đại cho Demo đồ án
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1e3799;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 15px;
        color: #4b6584;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #3867d6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .danger-box {
        background-color: #ffebee;
        border-left: 5px solid #e74c3c;
        padding: 15px;
        border-radius: 6px;
        color: #c0392b;
        font-weight: 600;
    }
    .success-box {
        background-color: #e8f8f5;
        border-left: 5px solid #2ecc71;
        padding: 15px;
        border-radius: 6px;
        color: #27ae60;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: Thông tin đồ án & Điều hướng
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/hotel-star.png", width=70)
st.sidebar.markdown("### **H2O AutoML Platform**")
st.sidebar.markdown("**Đồ án Big Data**")
st.sidebar.caption("Trường Đại học Công Thương TP.HCM (HUIT)")

st.sidebar.divider()
st.sidebar.markdown("**Thông tin nhóm thực hiện:**")
st.sidebar.markdown("""
- **GVHD:** ThS. Ngô Dương Hà
- **SV 1:** Nguyễn Quốc Huy (2001230301)
- **SV 2:** Hoàng Văn Hoan (2001230258)
""")

st.sidebar.divider()
page = st.sidebar.radio(
    "Chọn chức năng:",
    ["📊 1. Tổng quan dữ liệu", 
     "📈 2. Phân tích khám phá (EDA)", 
     "🏆 3. Đánh giá mô hình", 
     "🔮 4. Dự đoán đơn đặt phòng"],
    index=0
)

st.sidebar.divider()
st.sidebar.info("💡 **Hệ thống:** H2O-3.46.0 Local Cluster (4GB RAM) | Mô hình Leader: Stacked Ensemble")

# -----------------------------------------------------------------------------
# TRANG 1: TỔNG QUAN DỮ LIỆU
# -----------------------------------------------------------------------------
def render_page_overview():
    st.markdown('<div class="main-header">📊 TỔNG QUAN TẬP DỮ LIỆU ĐẶT PHÒNG KHÁCH SẠN</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Đề tài: "H2O - Nền tảng Machine Learning phân tán trên dữ liệu lớn" | Bộ dữ liệu Hotel Booking Demand (119,390 dòng)</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng số đơn đặt phòng", "119,390", help="Toàn bộ bản ghi từ Resort Hotel và City Hotel (2015-2017)")
    with col2:
        st.metric("Check-in thành công (Không hủy)", "75,166", "62.96%", delta_color="normal")
    with col3:
        st.metric("Số đơn bị hủy (is_canceled = 1)", "44,224", "-37.04%", delta_color="inverse")
    with col4:
        st.metric("Đặc trưng huấn luyện", "29", "Đã lọc rò rỉ")

    st.markdown("---")
    st.subheader("1. Phân bố trạng thái đặt phòng & Giá trị thiếu")
    
    col_left, col_right = st.columns(2)
    with col_left:
        fig1 = os.path.join(FIGURES_DIR, "01_cancellation_distribution.png")
        if os.path.exists(fig1):
            st.image(fig1, caption="Hình 1: Tỷ lệ đơn đặt phòng bị hủy vs không bị hủy", use_container_width=True)
        else:
            st.warning("Chưa có biểu đồ 01_cancellation_distribution.png")
            
    with col_right:
        fig5 = os.path.join(FIGURES_DIR, "05_missing_values.png")
        if os.path.exists(fig5):
            st.image(fig5, caption="Hình 2: Tỷ lệ khuyết dữ liệu (Missing Values) ở các cột", use_container_width=True)
        else:
            st.warning("Chưa có biểu đồ 05_missing_values.png")

    st.markdown("---")
    st.subheader("2. Xem trước dữ liệu thực tế (Sample Preview)")
    df_sample = load_preview_data(nrows=500)
    if not df_sample.empty:
        st.dataframe(df_sample.head(20), use_container_width=True)
        st.caption("Hiển thị 20/119,390 dòng dữ liệu đầu tiên từ file `hotel_bookings.csv`.")
    else:
        st.warning("Không tìm thấy file `hotel_bookings.csv` trong thư mục `data/`.")

# -----------------------------------------------------------------------------
# TRANG 2: PHÂN TÍCH KHÁM PHÁ (EDA)
# -----------------------------------------------------------------------------
def render_page_eda():
    st.markdown('<div class="main-header">📈 PHÂN TÍCH CÁC YẾU TỐ LIÊN QUAN ĐẾN HỦY PHÒNG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Khảo sát quan hệ giữa các hành vi đặt phòng với biến mục tiêu is_canceled</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📌 Các phát hiện cốt lõi", "🔍 Bộ lọc khám phá tương tác", "📊 Chi tiết các chỉ số"])
    
    with tab1:
        st.markdown("#### 1. Thời gian đặt trước (Lead Time) & Chính sách cọc (Deposit Type)")
        col_a, col_b = st.columns(2)
        with col_a:
            fig2 = os.path.join(FIGURES_DIR, "02_lead_time_distribution.png")
            if os.path.exists(fig2):
                st.image(fig2, caption="Hình 3: So sánh Lead Time (Số ngày đặt trước)", use_container_width=True)
            st.info("💡 **Nhận xét Lead Time:** Nhóm khách **HỦY PHÒNG** có số ngày đặt trước trung bình là **144.8 ngày** (trung vị 113 ngày), cao gần gấp đôi nhóm **KHÔNG HỦY** (trung bình 80.0 ngày, trung vị 45 ngày). Đặt càng xa ngày đến, nguy cơ hủy càng lớn.")
        
        with col_b:
            fig3 = os.path.join(FIGURES_DIR, "03_cancellation_by_deposit_type.png")
            if os.path.exists(fig3):
                st.image(fig3, caption="Hình 4: Tỷ lệ hủy theo loại tiền cọc (Deposit Type)", use_container_width=True)
            st.info("💡 **Nghịch lý 'Non Refund':** Khách đặt dạng Cọc không hoàn lại (*Non Refund*) có tỷ lệ hủy tới **99.36%**! Đây là nhóm khách đoàn/đại lý tour đặt block phòng sớm với giá ưu đãi chết cọc, chấp nhận mất cọc nếu tour không đủ khách.")

        st.markdown("#### 2. Phân khúc thị trường (Market Segment)")
        fig4 = os.path.join(FIGURES_DIR, "04_cancellation_by_market_segment.png")
        if os.path.exists(fig4):
            st.image(fig4, caption="Hình 5: Tỷ lệ hủy theo phân khúc thị trường (Market Segment)", use_container_width=True)
        st.info("💡 **Nhận xét Phân khúc:** Khách đoàn (*Groups*) có tỷ lệ hủy cao nhất (**61.06%**), tiếp theo là Đại lý trực tuyến *Online TA* (**36.72%**). Khách tự liên hệ trực tiếp (*Direct*) có tỷ lệ hủy thấp nhất (chỉ **15.34%**).")

    with tab2:
        st.markdown("#### Khám phá dữ liệu tương tác theo phân đoạn khách hàng")
        df = load_preview_data(nrows=50000)
        if not df.empty:
            c1, c2, c3 = st.columns(3)
            with c1:
                sel_hotel = st.selectbox("Chọn loại khách sạn:", ["Tất cả"] + list(df["hotel"].unique()))
            with c2:
                sel_deposit = st.selectbox("Chọn loại tiền cọc:", ["Tất cả"] + list(df["deposit_type"].unique()))
            with c3:
                sel_segment = st.selectbox("Chọn phân khúc thị trường:", ["Tất cả"] + list(df["market_segment"].unique()))
                
            filtered_df = df.copy()
            if sel_hotel != "Tất cả":
                filtered_df = filtered_df[filtered_df["hotel"] == sel_hotel]
            if sel_deposit != "Tất cả":
                filtered_df = filtered_df[filtered_df["deposit_type"] == sel_deposit]
            if sel_segment != "Tất cả":
                filtered_df = filtered_df[filtered_df["market_segment"] == sel_segment]
                
            total_f = len(filtered_df)
            if total_f > 0:
                canc_f = filtered_df["is_canceled"].sum()
                rate_f = (canc_f / total_f) * 100
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Số lượng bản ghi thỏa điều kiện", f"{total_f:,}")
                m2.metric("Số đơn bị hủy trong nhóm", f"{canc_f:,}")
                m3.metric("Tỷ lệ hủy trong nhóm", f"{rate_f:.2f}%")
                
                st.dataframe(filtered_df[["hotel", "lead_time", "arrival_date_month", "deposit_type", "market_segment", "adr", "is_canceled"]].head(15), use_container_width=True)
            else:
                st.warning("Không có đơn đặt phòng nào thỏa mãn các điều kiện lọc trên.")

    with tab3:
        st.markdown("#### Bảng thống kê các yếu tố hành vi quan trọng")
        st.markdown("""
        | Yếu tố | Nhóm có nguy cơ hủy cao | Nhóm an toàn (Ít hủy) |
        | :--- | :--- | :--- |
        | **Lead Time** | Đặt trước > 100 ngày (Tỷ lệ hủy ~55%) | Đặt trước < 30 ngày (Tỷ lệ hủy ~20%) |
        | **Lịch sử hủy** | Đã từng hủy >= 1 lần trong quá khứ (**94.4%** hủy lại) | Khách quen chưa từng hủy (**14.2%** hủy) |
        | **Thay đổi phòng** | Không thay đổi thông tin đặt phòng (**40.8%** hủy) | Có 1-2 lần chủ động đổi ngày/phòng (**14%** hủy) |
        | **Yêu cầu đặc biệt** | 0 yêu cầu đặc biệt (**47.7%** hủy) | Có từ 2 yêu cầu trở lên (nôi em bé, tầng cao) (**15.8%** hủy) |
        | **Chỗ đỗ xe** | Không yêu cầu chỗ đỗ xe (**39.5%** hủy) | Có yêu cầu chỗ đỗ xe (**0%** hủy - khách lái xe đến thực tế) |
        """)

# -----------------------------------------------------------------------------
# TRANG 3: ĐÁNH GIÁ MÔ HÌNH H2O AUTOML
# -----------------------------------------------------------------------------
def render_page_model_evaluation():
    st.markdown('<div class="main-header">🏆 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH H2O AUTOML</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Đánh giá mô hình Leader trên tập kiểm tra độc lập (23,948 dòng) | Thời gian chạy AutoML: 60s</div>', unsafe_allow_html=True)
    
    eval_res = load_eval_results()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("ROC-AUC", "0.9589", "Xuất sắc")
    with col2:
        st.metric("Accuracy", "89.04%", "21,324 / 23,948")
    with col3:
        st.metric("Precision", "85.32%", "Độ chuẩn xác")
    with col4:
        st.metric("Recall", "85.25%", "Độ nhạy bắt trúng")
    with col5:
        st.metric("F1-Score", "0.8528", "Cân bằng P & R")
    with col6:
        st.metric("Logloss", "0.2456", "Sai số tự tin thấp")

    st.markdown("---")
    st.subheader("1. Đường cong ROC & Ma trận nhầm lẫn (Confusion Matrix)")
    
    col_l, col_r = st.columns(2)
    with col_l:
        fig6 = os.path.join(FIGURES_DIR, "06_roc_curve.png")
        if os.path.exists(fig6):
            st.image(fig6, caption="Hình 6: Đường cong ROC trên tập Test độc lập (AUC = 0.9589)", use_container_width=True)
        else:
            st.warning("Chưa có biểu đồ 06_roc_curve.png")
            
    with col_r:
        fig7 = os.path.join(FIGURES_DIR, "07_confusion_matrix.png")
        if os.path.exists(fig7):
            st.image(fig7, caption="Hình 7: Ma trận nhầm lẫn trên 23,948 mẫu Test độc lập", use_container_width=True)
        else:
            st.warning("Chưa có biểu đồ 07_confusion_matrix.png")

    st.markdown("---")
    st.subheader("2. Bảng xếp hạng AutoML Leaderboard (22 mô hình được huấn luyện)")
    
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    lb_data = meta.get("leaderboard", [])
    if lb_data:
        df_lb = pd.DataFrame(lb_data)
        # Format các chỉ số
        cols_to_format = ["auc", "logloss", "aucpr", "mean_per_class_error", "rmse", "mse"]
        for c in cols_to_format:
            if c in df_lb.columns:
                df_lb[c] = df_lb[c].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
                
        st.dataframe(df_lb[["model_id", "auc", "logloss", "aucpr", "mean_per_class_error", "rmse"]], use_container_width=True)
        st.caption("Bảng xếp hạng đầy đủ 22 mô hình từ H2O AutoML. Mô hình đứng đầu là StackedEnsemble_AllModels_3.")
        
    st.markdown("---")
    st.subheader("3. Thảo luận kỹ thuật cho Báo cáo Đồ án")
    st.markdown("""
    - **Tại sao Stacked Ensemble dẫn đầu?** Mô hình xếp tầng kết hợp dự đoán từ các cây Gradient Boosting (GBM), Random Forest (DRF) và GLM qua mô hình meta-learner, giúp tận dụng thế mạnh phân vùng của cây và tính khái quát hóa của hồi quy logistic.
    - **Ưu thế của H2O AutoML:** Tự động hoàn toàn từ One-Hot encoding, Cross-validation 5-fold, Grid Search tham số và tạo Stacked Ensemble chỉ trong 60 giây, tiết kiệm thời gian thử nghiệm thủ công.
    """)

# -----------------------------------------------------------------------------
# TRANG 4: DỰ ĐOÁN ĐƠN ĐẶT PHÒNG
# -----------------------------------------------------------------------------
def render_page_prediction():
    st.markdown('<div class="main-header">🔮 DỰ ĐOÁN XÁC SUẤT HỦY ĐẶT PHÒNG KHÁCH SẠN</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ứng dụng mô hình Stacked Ensemble đã huấn luyện sẵn để dự đoán trực tuyến</div>', unsafe_allow_html=True)
    
    with st.spinner("Đang kết nối H2O Cluster và nạp mô hình vào bộ nhớ..."):
        model, meta = load_h2o_model_and_meta()
        
    if model is None:
        st.error("❌ Không thể nạp mô hình H2O! Hãy chắc chắn bạn đã chạy file `train_automl.py` trước.")
        return
        
    pred_tab1, pred_tab2 = st.tabs(["👤 1. Dự đoán cho 1 đơn đặt phòng", "📁 2. Dự đoán hàng loạt từ file CSV"])
    
    with pred_tab1:
        st.markdown("##### Nhập thông tin chi tiết đơn đặt phòng:")
        
        with st.form("single_prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🏨 Thông tin Khách sạn & Thời gian**")
                hotel = st.selectbox("Loại khách sạn:", ["City Hotel", "Resort Hotel"])
                lead_time = st.number_input("Số ngày đặt trước (lead_time):", min_value=0, max_value=800, value=60, help="Số ngày từ lúc đặt tới ngày nhận phòng")
                arrival_date_year = st.selectbox("Năm đến:", [2026, 2027, 2017, 2016, 2015], index=0)
                arrival_date_month = st.selectbox("Tháng đến:", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], index=6)
                arrival_date_week_number = st.slider("Tuần trong năm:", 1, 53, 28)
                arrival_date_day_of_month = st.slider("Ngày trong tháng:", 1, 31, 15)
                stays_in_weekend_nights = st.number_input("Số đêm cuối tuần (T7, CN):", min_value=0, max_value=20, value=2)
                stays_in_week_nights = st.number_input("Số đêm trong tuần (T2-T6):", min_value=0, max_value=50, value=3)

            with col2:
                st.markdown("**👥 Thông tin Khách hàng & Phòng**")
                adults = st.number_input("Số người lớn:", min_value=1, max_value=10, value=2)
                children = st.number_input("Số trẻ em:", min_value=0, max_value=10, value=0)
                babies = st.number_input("Số trẻ sơ sinh:", min_value=0, max_value=5, value=0)
                country = st.selectbox("Quốc gia xuất xứ:", ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "BRA", "NLD", "USA", "VNM", "Unknown"], index=0)
                meal = st.selectbox("Gói bữa ăn (meal):", ["BB", "HB", "FB", "SC", "Undefined"], index=0, help="BB: Bed & Breakfast, HB: Half board, FB: Full board")
                reserved_room_type = st.selectbox("Loại phòng đăng ký:", ["A", "B", "C", "D", "E", "F", "G", "H", "L"], index=0)
                assigned_room_type = st.selectbox("Loại phòng thực xếp:", ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L"], index=0)
                adr = st.number_input("Giá phòng trung bình/ngày (ADR, EUR):", min_value=0.0, max_value=1000.0, value=115.0)

            with col3:
                st.markdown("**💳 Kênh đặt, Tiền cọc & Lịch sử**")
                market_segment = st.selectbox("Phân khúc thị trường:", ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation"], index=0)
                distribution_channel = st.selectbox("Kênh phân phối:", ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"], index=0)
                deposit_type = st.selectbox("Chính sách tiền cọc:", ["No Deposit", "Non Refund", "Refundable"], index=0)
                customer_type = st.selectbox("Loại khách hàng:", ["Transient", "Transient-Party", "Contract", "Group"], index=0)
                is_repeated_guest = st.radio("Khách quay lại?", [0, 1], format_func=lambda x: "Có (Khách quen)" if x == 1 else "Không (Khách mới)", horizontal=True)
                has_company = st.radio("Có công ty bảo trợ?", [0, 1], format_func=lambda x: "Có" if x == 1 else "Không", horizontal=True)
                previous_cancellations = st.number_input("Số lần từng hủy trong quá khứ:", min_value=0, max_value=30, value=0)
                previous_bookings_not_canceled = st.number_input("Số lần từng ở thành công:", min_value=0, max_value=50, value=0)
                booking_changes = st.number_input("Số lần đổi thông tin phòng:", min_value=0, max_value=20, value=0)
                agent = st.number_input("Mã đại lý (agent, 0 nếu không có):", min_value=0, max_value=600, value=9)
                days_in_waiting_list = st.number_input("Số ngày trong hàng đợi:", min_value=0, max_value=400, value=0)
                required_car_parking_spaces = st.number_input("Số chỗ đậu xe yêu cầu:", min_value=0, max_value=5, value=0)
                total_of_special_requests = st.number_input("Số yêu cầu đặc biệt (tầng cao, nôi):", min_value=0, max_value=5, value=1)

            st.markdown("---")
            threshold = st.slider("⚙️ Ngưỡng phân loại hủy phòng (Threshold):", min_value=0.10, max_value=0.90, value=0.45, step=0.01,
                                 help="Mặc định 0.45 tối ưu F1-Score. Đặt ngưỡng thấp nếu khách sạn muốn bắt chặt các rủi ro hủy phòng.")

            submitted = st.form_submit_button("🔍 THỰC HIỆN DỰ ĐOÁN XÁC SUẤT HỦY", use_container_width=True)

        if submitted:
            # Tạo dictionary đúng 29 features
            input_dict = {
                "hotel": [hotel],
                "lead_time": [int(lead_time)],
                "arrival_date_year": [int(arrival_date_year)],
                "arrival_date_month": [arrival_date_month],
                "arrival_date_week_number": [int(arrival_date_week_number)],
                "arrival_date_day_of_month": [int(arrival_date_day_of_month)],
                "stays_in_weekend_nights": [int(stays_in_weekend_nights)],
                "stays_in_week_nights": [int(stays_in_week_nights)],
                "adults": [int(adults)],
                "children": [int(children)],
                "babies": [int(babies)],
                "meal": [meal],
                "country": [country],
                "market_segment": [market_segment],
                "distribution_channel": [distribution_channel],
                "is_repeated_guest": [str(is_repeated_guest)],
                "previous_cancellations": [int(previous_cancellations)],
                "previous_bookings_not_canceled": [int(previous_bookings_not_canceled)],
                "reserved_room_type": [reserved_room_type],
                "assigned_room_type": [assigned_room_type],
                "booking_changes": [int(booking_changes)],
                "deposit_type": [deposit_type],
                "agent": [int(agent)],
                "days_in_waiting_list": [int(days_in_waiting_list)],
                "customer_type": [customer_type],
                "adr": [float(adr)],
                "required_car_parking_spaces": [int(required_car_parking_spaces)],
                "total_of_special_requests": [int(total_of_special_requests)],
                "has_company": [str(has_company)]
            }
            
            # Chuẩn bị H2OFrame
            import h2o
            input_df = pd.DataFrame(input_dict)
            input_hf = h2o.H2OFrame(input_df)
            
            # Ép kiểu danh mục và biến số đồng nhất cho H2O
            for col in meta.get("categorical_features", []):
                if col in input_hf.columns:
                    input_hf[col] = input_hf[col].asfactor()
            for col in meta.get("numerical_features", []):
                if col in input_hf.columns:
                    input_hf[col] = input_hf[col].asnumeric()
                    
            try:
                pred = model.predict(input_hf)
                pred_df = pred.as_data_frame()
                p0 = float(pred_df["p0"].iloc[0])  # Xác suất Không hủy
                p1 = float(pred_df["p1"].iloc[0])  # Xác suất Hủy
                
                # So sánh với ngưỡng
                is_canceled_pred = 1 if p1 >= threshold else 0
                
                st.markdown("### 🎯 KẾT QUẢ DỰ ĐOÁN:")
                
                res_col1, res_col2 = st.columns([1, 1])
                with res_col1:
                    if is_canceled_pred == 1:
                        st.markdown(f"""
                        <div class="danger-box">
                            🚨 CẢNH BÁO: ĐƠN NÀY CÓ NGUY CƠ HỦY PHÒNG CAO (is_canceled = 1)<br>
                            Xác suất hủy phòng dự đoán: <b>{p1*100:.2f}%</b> (Vượt ngưỡng {threshold*100:.0f}%)
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-box">
                            ✅ AN TOÀN: KHÁCH HÀNG SẼ CHECK-IN THÀNH CÔNG (is_canceled = 0)<br>
                            Xác suất ở lại dự đoán: <b>{p0*100:.2f}%</b> (Xác suất hủy: {p1*100:.2f}%)
                        </div>
                        """, unsafe_allow_html=True)
                        
                with res_col2:
                    st.write(f"**Biểu đồ phân phối xác suất dự đoán:**")
                    st.progress(p1, text=f"Nguy cơ hủy: {p1*100:.1f}%")
                    st.write(f"- Xác suất KHÔNG HỦY ($p_0$): **{p0*100:.2f}%**")
                    st.write(f"- Xác suất HỦY PHÒNG ($p_1$): **{p1*100:.2f}%**")

                st.markdown("#### 💡 Khuyến nghị nghiệp vụ cho khách sạn:")
                if is_canceled_pred == 1:
                    st.warning("""
                    - **Hành động 1:** Gửi email/SMS tự động nhắc nhở khách xác nhận chuyến đi trước ngày check-in 3 ngày.
                    - **Hành động 2:** Đề xuất ưu đãi nâng cấp phòng hoặc giảm giá dịch vụ ăn uống nếu khách thanh toán trước 100%.
                    - **Hành động 3:** Cân nhắc chính sách Overbooking (cho phép nhận thêm khách vãng lai) để bù đắp công suất phòng trống.
                    """)
                else:
                    st.success("""
                    - **Hành động:** Giữ phòng theo đúng cam kết tiêu chuẩn, chuẩn bị phòng theo các yêu cầu đặc biệt của khách.
                    """)
                    
            except Exception as e:
                st.error(f"Lỗi khi thực hiện dự đoán: {e}")

    with pred_tab2:
        st.markdown("##### Dự đoán hàng loạt từ file CSV:")
        st.write("Tải lên danh sách các đơn đặt phòng mới để hệ thống tự động chấm điểm rủi ro cho từng đơn.")
        
        # Cung cấp file mẫu
        if os.path.exists(SAMPLE_BATCH_PATH):
            with open(SAMPLE_BATCH_PATH, "rb") as f:
                st.download_button(
                    label="📥 Tải file mẫu kiểm tra (sample_batch_test.csv - 50 dòng)",
                    data=f,
                    file_name="sample_batch_test.csv",
                    mime="text/csv"
                )

        uploaded_file = st.file_uploader("Chọn file CSV:", type=["csv"])
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Đã nạp file: **{batch_df.shape[0]:,} dòng**, **{batch_df.shape[1]} cột**.")
                
                if st.button("🚀 CHẠY DỰ ĐOÁN HÀNG LOẠT", use_container_width=True):
                    with st.spinner("Đang xử lý dự đoán phân tán trên H2O..."):
                        import h2o
                        batch_hf = h2o.H2OFrame(batch_df)
                        for col in meta.get("categorical_features", []):
                            if col in batch_hf.columns:
                                batch_hf[col] = batch_hf[col].asfactor()
                        for col in meta.get("numerical_features", []):
                            if col in batch_hf.columns:
                                batch_hf[col] = batch_hf[col].asnumeric()
                                
                        batch_preds = model.predict(batch_hf)
                        preds_df = batch_preds.as_data_frame()
                        
                        result_df = batch_df.copy()
                        result_df["Xac_Suat_Huy_%"] = (preds_df["p1"] * 100).round(2)
                        result_df["Du_Doan_Huy"] = (preds_df["p1"] >= 0.45).map({True: "HỦY (1)", False: "KHÔNG HỦY (0)"})
                        
                        st.success("✅ Dự đoán hoàn tất!")
                        st.dataframe(result_df[["hotel", "lead_time", "arrival_date_month", "deposit_type", "market_segment", "Xac_Suat_Huy_%", "Du_Doan_Huy"]].head(30), use_container_width=True)
                        
                        csv_data = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Tải kết quả dự đoán (predictions.csv)",
                            data=csv_data,
                            file_name="hotel_cancellation_predictions.csv",
                            mime="text/csv"
                        )
            except Exception as ex:
                st.error(f"Lỗi khi xử lý file CSV: {ex}")

# -----------------------------------------------------------------------------
# MAIN APP ROUTER
# -----------------------------------------------------------------------------
def main():
    if "1." in page:
        render_page_overview()
    elif "2." in page:
        render_page_eda()
    elif "3." in page:
        render_page_model_evaluation()
    elif "4." in page:
        render_page_prediction()

if __name__ == "__main__":
    main()
