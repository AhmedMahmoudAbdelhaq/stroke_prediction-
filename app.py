import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="NEUROSHIELD AI | Clinical Decision Support System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Cairo', 'Plus Jakarta Sans', sans-serif;
    }
    
    /* خلفية التطبيق الفاتحة والمشرقة */
    .stApp {
        background: #f8fafc;
        color: #0f172a;
    }
    
    /* الشريط الجانبي */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* ترويسة المستشفى الزرقاء الأنيقة */
    .hero-banner {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 50%, #075985 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: #ffffff;
        box-shadow: 0 10px 25px rgba(2, 132, 199, 0.15);
        margin-bottom: 25px;
    }
    .hero-banner h1 {
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        color: #ffffff;
    }
    .hero-banner p {
        margin: 6px 0 0 0;
        font-size: 14px;
        color: #e0f2fe;
    }

    /* كروت إدخال البيانات المضيئة */
    .panel-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 16px;
        font-weight: 700;
        color: #0284c7;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    
    /* كروت عرض النتائج الملونة والواضحة */
    .result-card {
        padding: 22px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    .card-low {
        background: #ecfdf5;
        border: 2px solid #10b981;
        color: #065f46;
    }
    .card-high {
        background: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = joblib.load('stroke_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"تعذر تحميل الموديل: {e}")
    st.stop()

with st.sidebar:
    st.markdown("### 🧬 **NEUROSHIELD AI**")
    st.caption("منظومة الفحص السريري المتقدم")
    st.markdown("---")
    patient_id = st.text_input("معرف المريض (Patient ID)", value="PT-90412")
    physician = st.text_input("الطبيب الفاحص (Physician)", value="Dr. A. Mahmoud")
    department = st.selectbox("القسم الإكلينيكي", ["Vascular Neurology", "Emergency Triage", "Intensive Stroke Unit", "General Outpatient"])
    st.markdown("---")
    clinical_threshold = st.slider(
        "حساسية عتبة القرار (Cutoff Threshold)",
        min_value=0.40,
        max_value=0.85,
        value=0.65,
        step=0.01,
        help="المعيار الطبي الموصى به 0.65"
    )
    st.caption(f"العتبة المختارة حالياً: **{clinical_threshold * 100:.1f}%**")

st.markdown(f"""
<div class="hero-banner">
    <h1>🏥 NEUROSHIELD | المركز التشخيصي الذكي</h1>
    <p>ملف المريض: <b>{patient_id}</b> | القسم: <b>{department}</b> | المشرف: <b>{physician}</b> | التاريخ: <b>{datetime.now().strftime('%Y-%m-%d')}</b></p>
</div>
""", unsafe_allow_html=True)

col_left, col_mid, col_right = st.columns([1, 1, 1], gap="medium")

with col_left:
    st.markdown('<div class="panel-card"><div class="section-title">👤 البيانات الأساسية</div>', unsafe_allow_html=True)
    age = st.slider("العمر (سنة)", min_value=1, max_value=105, value=58)
    gender = st.radio("النوع", ["Male", "Female"], horizontal=True)
    ever_married = st.selectbox("الحالة الاجتماعية", ["Yes", "No"])
    residence = st.selectbox("مكان السكن", ["Urban", "Rural"])
    st.markdown('</div>', unsafe_allow_html=True)

with col_mid:
    st.markdown('<div class="panel-card"><div class="section-title">🩺 الفحوصات والمؤشرات الحيوية</div>', unsafe_allow_html=True)
    avg_glucose = st.number_input("متوسط السكر (mg/dL)", min_value=50.0, max_value=320.0, value=135.0, step=1.0)
    bmi = st.number_input("مؤشر كتلة الجسم (BMI)", min_value=12.0, max_value=65.0, value=29.2, step=0.1)
    st.write("**التاريخ المرضي المزمن:**")
    c1, c2 = st.columns(2)
    with c1:
        hypertension = st.checkbox("ضغط دم مرتفع")
    with c2:
        heart_disease = st.checkbox("مرض بالقلب")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel-card"><div class="section-title">🧬 نمط المعيشة والعمل</div>', unsafe_allow_html=True)
    work_type = st.selectbox("طبيعة العمل", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    smoking = st.selectbox("حالة التدخين", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    if avg_glucose >= 140.0:
        st.warning("⚠️ مستوى السكر أعلى من المعدل الطبيعي")
    if bmi >= 30.0:
        st.info("ℹ️ مؤشر كتلة الجسم يقع ضمن السمنة")
    st.markdown('</div>', unsafe_allow_html=True)

run_eval = st.button("⚡ تحليل البيانات وإصدار النتيجة الطبية", use_container_width=True, type="primary")

if run_eval:
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if ever_married == "Yes" else 0
    residence_val = 1 if residence == "Urban" else 0
    htn_val = 1 if hypertension else 0
    hd_val = 1 if heart_disease else 0
    
    glucose_log = np.log(avg_glucose)
    
    try:
        if scaler.n_features_in_ == 3:
            scaled_res = scaler.transform([[age, bmi, glucose_log]])
            s_age, s_bmi, s_gluc = scaled_res[0][0], scaled_res[0][1], scaled_res[0][2]
        elif scaler.n_features_in_ == 2:
            scaled_res = scaler.transform([[bmi, glucose_log]])
            s_age = age
            s_bmi, s_gluc = scaled_res[0][0], scaled_res[0][1]
        else:
            s_age, s_bmi, s_gluc = age, bmi, glucose_log
    except Exception:
        s_age, s_bmi, s_gluc = age, bmi, glucose_log

    work_never = 1 if work_type == "Never_worked" else 0
    work_private = 1 if work_type == "Private" else 0
    work_self = 1 if work_type == "Self-employed" else 0
    work_children = 1 if work_type == "children" else 0

    smoke_formerly = 1 if smoking == "formerly smoked" else 0
    smoke_never = 1 if smoking == "never smoked" else 0
    smoke_smokes = 1 if smoking == "smokes" else 0

    columns_15 = [
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
        'Residence_type', 'bmi', 'avg_glucose_level_log',
        'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children',
        'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
    ]
    
    values_15 = [
        gender_val, s_age, htn_val, hd_val, married_val,
        residence_val, s_bmi, s_gluc,
        work_never, work_private, work_self, work_children,
        smoke_formerly, smoke_never, smoke_smokes
    ]
    
    input_df = pd.DataFrame([values_15], columns=columns_15)
    stroke_prob = float(model.predict_proba(input_df)[0][1])
    is_high_risk = stroke_prob >= clinical_threshold

    st.markdown("---")
    st.markdown("### 📋 التقرير والنتيجة النهائية")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="result-card {'card-high' if is_high_risk else 'card-low'}">
            <div style="font-size: 15px; font-weight:600;">نسبة الخطورة المقدرة</div>
            <div style="font-size: 44px; font-weight: 800; margin: 6px 0;">{stroke_prob * 100:.1f}%</div>
            <div style="font-size: 13px; font-weight:600;">{'⚠️ تتجاوز عتبة الأمان المحددة' if is_high_risk else '✅ ضمن النطاق الآمن والمستقر'}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        st.markdown(f"""
        <div class="result-card" style="background:#ffffff; border:1.5px solid #cbd5e1;">
            <div style="font-size: 15px; color: #64748b; font-weight:600;">العتبة المعتمدة للقرار</div>
            <div style="font-size: 44px; font-weight: 800; color: #0284c7; margin: 6px 0;">{clinical_threshold * 100:.1f}%</div>
            <div style="font-size: 13px; color: #64748b;">أي نسبة أعلى من هذا الحد تُعتبر خطرة</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m3:
        status_text = "خطر مرتفع (اشتباه)" if is_high_risk else "حالة مستقرة (آمنة)"
        status_color = "#dc2626" if is_high_risk else "#16a34a"
        card_bg = "#fef2f2" if is_high_risk else "#f0fdf4"
        st.markdown(f"""
        <div class="result-card" style="background:{card_bg}; border:1.5px solid {status_color};">
            <div style="font-size: 15px; color: #64748b; font-weight:600;">التصنيف السريري العام</div>
            <div style="font-size: 28px; font-weight: 800; color: {status_color}; margin: 14px 0;">{status_text}</div>
            <div style="font-size: 13px; color: #64748b;">{'يتطلب فحوصات عاجلة' if is_high_risk else 'متابعة دورية اعتيادية'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    col_gauge, col_radar = st.columns([1.2, 1])
    
    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stroke_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "مقياس الخطر الكلي", 'font': {'size': 18, 'color': '#0f172a'}},
            number={'suffix': "%", 'font': {'size': 38, 'color': '#0f172a'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#64748b"},
                'bar': {'color': "#dc2626" if is_high_risk else "#0284c7", 'thickness': 0.3},
                'bgcolor': "#f1f5f9",
                'borderwidth': 1,
                'bordercolor': "#cbd5e1",
                'steps': [
                    {'range': [0, clinical_threshold * 100], 'color': 'rgba(16, 185, 129, 0.15)'},
                    {'range': [clinical_threshold * 100, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                ],
                'threshold': {
                    'line': {'color': "#dc2626", 'width': 4},
                    'thickness': 0.8,
                    'value': clinical_threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_radar:
        categories = ['العمر', 'السكر', 'الوزن (BMI)', 'القلب والضغط', 'التدخين']
        norm_age = min(age / 90.0, 1.0)
        norm_gluc = min(avg_glucose / 250.0, 1.0)
        norm_bmi = min(bmi / 45.0, 1.0)
        norm_vasc = (htn_val + hd_val) / 2.0
        norm_smoke = 1.0 if smoking == 'smokes' else 0.5 if smoking == 'formerly smoked' else 0.1

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[norm_age, norm_gluc, norm_bmi, norm_vasc, norm_smoke],
            theta=categories,
            fill='toself',
            name='بيانات الحالة',
            line_color='#0284c7',
            fillcolor='rgba(2, 132, 199, 0.2)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#e2e8f0"),
                angularaxis=dict(gridcolor="#e2e8f0", color="#0f172a")
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=280,
            margin=dict(l=30, r=30, t=30, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    if is_high_risk:
        st.error("""
        **🚨 خطة التدخل المقترحة للحالة (عالية الخطورة):**
        * توجيه المريض لوحدة الرعاية العصبية والوعائية المتخصصة.
        * إجراء تصوير بالرنين المغناطيسي أو أشعة مقطعية لشرايين المخ (Brain MRI / CT).
        * تخطيط كهربائية القلب وفحص دوبلر لشرايين الرقبة السباتية.
        """)
    else:
        st.success("""
        **✅ التوصيات المقترحة للحالة (مستقرة):**
        * المؤشرات تقع ضمن المعدلات الطبيعية والآمنة.
        * الحفاظ على ممارسة النشاط البدني ومراقبة ضغط الدم ومستوى السكر دورياً.
        """)