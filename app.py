import streamlit as st
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from core.inference_engine import AquaExpertSystem

st.set_page_config(
    page_title="AquaExpert AI - Hệ Chuyên Gia Chẩn Đoán Bệnh Thủy Sản",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tải hệ thống chuyên gia
@st.cache_resource
def load_system():
    return AquaExpertSystem()

try:
    expert_system = load_system()
except Exception as e:
    st.error(f"Lỗi khởi tạo hệ thống: {e}")
    st.stop()

st.title("🦐 AquaExpert AI: Hệ Chuyên Gia Chẩn Đoán Bệnh & Giám Sát Hồ Nuôi")
st.caption("Ứng dụng kết hợp Machine Learning phân loại chất lượng nước/NLP và Động cơ suy diễn tiến (Forward Chaining + Certainty Factor)")
st.markdown("---")

col_left, col_right = st.columns([1.05, 1.15], gap="large")

with col_left:
    st.subheader("1. Dữ Liệu Lâm Sàng & Môi Trường")
    
    # 1. Chọn đối tượng nuôi (Đầy đủ 3 loài tôm và 2 loài cá)
    species = st.selectbox(
        "🎯 **Đối tượng thủy sản cần chẩn đoán:**",
        ["Tôm thẻ chân trắng", "Tôm sú", "Tôm càng xanh", "Cá rô phi", "Cá tra"],
        index=0
    )
    
    st.write(f"💡 **Gợi ý chọn nhanh triệu chứng điển hình của {species.lower()}:**")
    
    # Danh sách triệu chứng thay đổi động theo từng nhóm loài
    if species in ["Tôm thẻ chân trắng", "Tôm sú"]:
        quick_tags = [
            "gan teo, nhat mau", "ruot trong", "phan trang",
            "mang den", "mang ban, mang vang", "dong rong",
            "duc co, cong than", "vo mem, op than", "do duoi",
            "phat sang trong bong toi", "dau vang, gan tuy vang nhat",
            "bo an, lo do, tap bo"
        ]
    elif species == "Tôm càng xanh":
        quick_tags = [
            "vo mem, op than", "mang den", "dong rong",
            "duc co, cong than", "gan teo, nhat mau", "ruot trong",
            "bo an, lo do, tap bo"
        ]
    elif species == "Cá rô phi":
        quick_tags = [
            "mat loi", "boi xoay tron", "xuat huyet mang va vay",
            "truong bung", "lo loet than", "tuot nhot", "bo an, lo do"
        ]
    else:  # Cá tra
        quick_tags = [
            "dom trang tren gan than", "trang gan, trang mang",
            "xuat huyet toan than", "xuat huyet goc vay", "tuot nhot",
            "sung hau mon", "bo an, boi lo do"
        ]
    
    selected_tags = st.multiselect("Chọn các biểu hiện quan sát được:", quick_tags)
    
    default_text = ", ".join(selected_tags) if selected_tags else ""
    symptom_text = st.text_area(
        f"Mô tả chi tiết triệu chứng lâm sàng của {species.lower()}:",
        value=default_text,
        placeholder=f"Ví dụ: {species.lower()} giảm ăn, bơi dạt bờ, có dấu hiệu bất thường trên cơ thể...",
        height=85
    )
    
    st.markdown("---")
    st.write("🌦️ **Yếu tố thời tiết & Môi trường nước đo đạc:**")
    
    weather = st.selectbox(
        "Tình trạng thời tiết hiện tại / dự báo ngắn hạn:",
        [
            "Bình thường / Nắng nhẹ (Thời tiết ổn định)",
            "Mưa lớn kéo dài (>50mm)",
            "Nắng nóng gay gắt (>35°C)",
            "Biến động nhiệt độ mạnh (>4°C)",
            "Âm u / Thiếu ánh sáng kéo dài"
        ]
    )
    
    sub1, sub2 = st.columns(2)
    with sub1:
        ph = st.number_input("Độ pH:", min_value=4.0, max_value=11.0, value=7.8, step=0.1)
        temp = st.number_input("Nhiệt độ nước (°C):", min_value=15.0, max_value=40.0, value=29.0, step=0.5)
        do = st.number_input("Oxy hòa tan DO (mg/L):", min_value=0.5, max_value=12.0, value=5.0, step=0.2)
    with sub2:
        salinity = st.number_input("Độ mặn (‰):", min_value=0.0, max_value=40.0, value=15.0, step=1.0)
        nh3 = st.number_input("Khí độc NH3 (mg/L):", min_value=0.0, max_value=2.0, value=0.02, step=0.01)
        no2 = st.number_input("Khí độc NO2 (mg/L):", min_value=0.0, max_value=2.0, value=0.05, step=0.01)
        
    btn_run = st.button("🚀 Bắt Đầu Phân Tích & Chẩn Đoán", type="primary", use_container_width=True)

with col_right:
    st.subheader("2. Kết Quả Chẩn Đoán & Khuyến Nghị Chuyên Gia")
    
    if btn_run:
        if not symptom_text.strip():
            st.warning("⚠️ Vui lòng nhập hoặc chọn ít nhất một vài triệu chứng lâm sàng.")
        else:
            with st.spinner("Đang thực hiện suy diễn tiến kết hợp Certainty Factor..."):
                water_params = {
                    "temperature": temp, "ph": ph, "do": do,
                    "salinity": salinity, "nh3": nh3, "no2": no2
                }
                
                # 1. Đánh giá độ phù hợp nguồn nước theo ngưỡng sinh học
                suitability = expert_system.evaluate_water_suitability(water_params)
                best_fit = max(suitability, key=suitability.get)
                
                # 2. Thực hiện chẩn đoán
                raw_res = expert_system.diagnose(
                    symptom_input=symptom_text,
                    ph=ph, temp=temp, do=do, salinity=salinity, nh3=nh3, no2=no2,
                    weather=weather,
                    species=species
                )
                
                # Xử lý bóc tách an toàn: Hỗ trợ cả list lẫn dict trả về
                if isinstance(raw_res, list):
                    matched_rules = raw_res
                    w_status, w_issues = expert_system.assess_water_status(ph, temp, do, salinity, nh3, no2)
                    top_rule = matched_rules[0] if matched_rules else {}
                    res = {
                        "predicted_disease": top_rule.get("disease_code", "Chưa xác định"),
                        "disease_vi": top_rule.get("disease_vietnamese", "Không phát hiện nguy cơ rõ ràng"),
                        "confidence": top_rule.get("confidence", 0.0),
                        "water_status": w_status,
                        "env_warnings": w_issues,
                        "treatment": top_rule.get("recommendation", "Tiếp tục theo dõi sức khỏe vật nuôi."),
                        "all_matched_rules": matched_rules,
                        "inference_trace": [f"Kích hoạt: {e}" for e in top_rule.get("evidence", [])]
                    }
                else:
                    res = raw_res
                    matched_rules = res.get("all_matched_rules", [])

            # ==============================================================
            # TRỌNG TÂM CỐT LÕI: KẾT QUẢ CHẨN ĐOÁN BỆNH THEO HỆ CHUYÊN GIA
            # ==============================================================
            ten_benh = res.get('disease_vi', 'Chưa phát hiện bệnh rõ ràng')
            ma_benh = res.get('predicted_disease', '')
            conf = res.get('confidence', 0.0)
            
            st.markdown(f"### 🔬 Bệnh Nghi Ngờ: :red[{ten_benh}]")
            st.caption(f"Mã khoa học trong cơ sở tri thức: **{ma_benh}**")
            
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Độ phù hợp tri thức (CF)", f"{conf}%")
            with m_col2:
                risk_lvl = "CAO" if conf >= 70 else ("TRUNG BÌNH" if conf >= 45 else "THẤP")
                st.metric("Mức độ nguy cơ", risk_lvl)

            # Phác đồ điều trị
            treatment = res.get("treatment", "Chưa có khuyến nghị cụ thể.")
            st.error(f"💊 **Phác đồ khuyến nghị & Hướng xử lý khẩn cấp:**\n\n{treatment}")

            # Danh sách các bệnh khác có cùng triệu chứng nếu có nhiều luật thỏa mãn
            if len(matched_rules) > 1:
                with st.expander("📋 Xem thêm các bệnh khác có cùng triệu chứng:"):
                    for extra in matched_rules[1:]:
                        st.write(f"- **{extra.get('disease_vietnamese', extra.get('disease_code'))}**: Nguy cơ **{extra.get('risk_level', 'THẤP')}** (Độ tin cậy: {extra.get('confidence', 0)}%)")

            # Biểu đồ Top 3 Machine Learning nếu có dữ liệu text classifier
            top_preds = res.get("top_predictions", [])
            if top_preds:
                st.write("📊 **Xác suất phân tích triệu chứng từ Machine Learning:**")
                df_chart = pd.DataFrame(top_preds)
                st.bar_chart(df_chart.set_index("Tên bệnh"), color="#ff4b4b")

            # Dấu vết suy diễn Explainable AI
            traces = res.get("inference_trace", [])
            if traces:
                with st.expander("🔍 Xem Dấu Vết Suy Diễn Của Hệ Chuyên Gia (Explainable AI)", expanded=True):
                    st.write("**Các bằng chứng kích hoạt hệ luật:**")
                    for step in traces:
                        st.write(f"- {step}")

            st.markdown("---")

            # ==============================================================
            # ĐÁNH GIÁ CHẤT LƯỢNG NGUỒN NƯỚC & THỜI TIẾT HỖ TRỢ
            # ==============================================================
            st.markdown("#### 🌊 Đánh Giá Chất Lượng Nguồn Nước & Thời Tiết")
            
            # Gợi ý loài phù hợp với nguồn nước đo
            st.info(f"💡 Nguồn nước đo đạc hiện tại phù hợp nhất cho nuôi: **{best_fit}** ({suitability[best_fit]}% tiêu chuẩn).")
            
            s_cols = st.columns(len(suitability))
            for i, (sp, score) in enumerate(suitability.items()):
                with s_cols[i]:
                    status = "✅ Tối ưu" if score >= 85 else ("⚠️ Cần chỉnh" if score >= 70 else "❌ Nguy hại")
                    st.metric(label=sp, value=f"{score}%", delta=status)
            
            w_status = res.get("water_status", "An toàn")
            if w_status == "An toàn":
                st.success(f"🟢 **Trạng thái môi trường ao nuôi ({species}):** {w_status}")
            elif w_status == "Cảnh báo":
                st.warning(f"🟡 **Trạng thái môi trường ao nuôi ({species}):** {w_status}")
            else:
                st.error(f"🔴 **Trạng thái môi trường ao nuôi ({species}):** {w_status}")

            warnings = res.get("env_warnings", [])
            if warnings:
                for w in warnings:
                    st.write(f"- ⚠️ {w}")
                    
            st.caption("ℹ️ *Lưu ý: Kết quả mang tính chất cảnh báo và hỗ trợ ra quyết định lâm sàng, không thay thế hoàn toàn xét nghiệm PCR chuyên môn.*")
    else:
        st.info("Nhập các thông số bên trái và nhấn nút **Bắt Đầu Phân Tích & Chẩn Đoán** để xem kết quả chuyên gia.")