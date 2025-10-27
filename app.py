import streamlit as st
from components.navbar import navbar

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Data Mining Dashboard",
    page_icon="📊",
    layout="wide",
)

# ========== GLOBAL CSS ==========
st.markdown("""
<style>
    .main {background-color: #F5F7FA;}
    .block-container {padding-top: 2rem;}
    h1, h2, h3 {color: #0F2940; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

# ========== NAVBAR ==========
navbar()

# ========== HOME CONTENT ==========
st.title("📊 Data Mining Platform")
st.markdown("""
Chào mừng đến với **Data Mining Dashboard** – Nơi bạn có thể:

✅ Khám phá dữ liệu thô và trực quan
✅ Tiền xử lý & làm sạch dữ liệu
✅ Huấn luyện các mô hình Mining / ML
✅ Phân tích thống kê nâng cao
✅ Xuất báo cáo PDF/CSV

---
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📈 Số dataset đang có", 12)
with col2:
    st.metric("⚙️ Số mô hình", 5)
with col3:
    st.metric("⏱️ Lần chạy gần nhất", "2 giờ trước")

st.subheader("✨ Tổng quan hệ thống")
st.write(
    "Ứng dụng này hỗ trợ toàn bộ workflow: *Exploration → Preprocessing → Modeling → Evaluation → Export*."
)

st.info("Đi tới sidebar để chọn chức năng phân tích chi tiết.")
