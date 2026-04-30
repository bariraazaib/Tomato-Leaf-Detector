import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(
    page_title="Tomato Leaf Detector",
    page_icon="🍅",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;1,400&family=DM+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
.stApp{background:linear-gradient(135deg,#fde8ef 0%,#fdf6ee 45%,#e8f3e6 100%)!important}
#MainMenu,footer,header{visibility:hidden}
.hero-title{font-family:'Playfair Display',serif!important;font-size:2.8rem;color:#3d2a35;text-align:center;line-height:1.2;margin-bottom:6px}
.hero-title em{color:#d4607e}
.hero-sub{text-align:center;font-size:14px;color:#8a6070;font-weight:300;margin-bottom:32px}
.badge-row{display:flex;justify-content:center;margin-bottom:14px}
.badge-pill{background:rgba(255,255,255,.65);border:1px solid rgba(242,167,188,.6);border-radius:20px;padding:5px 18px;font-size:11px;letter-spacing:.1em;color:#d4607e;font-weight:500;text-transform:uppercase}
.chips-row{display:flex;gap:10px;margin-bottom:20px;justify-content:center}
.chip{flex:1;background:rgba(255,255,255,.6);border:1px solid rgba(0,0,0,.06);border-radius:14px;padding:12px 8px;text-align:center}
.chip-icon{font-size:20px;display:block;margin-bottom:4px}
.chip-name{font-size:11px;color:#8a6070;font-weight:500}
.result-header-healthy{background:linear-gradient(135deg,#a8c5a0,#5a8a52);border-radius:20px 20px 0 0;padding:28px;color:white}
.result-header-early{background:linear-gradient(135deg,#f2c4a0,#c8602a);border-radius:20px 20px 0 0;padding:28px;color:white}
.result-header-septoria{background:linear-gradient(135deg,#e8a0b8,#a83060);border-radius:20px 20px 0 0;padding:28px;color:white}
.result-class-name{font-family:'Playfair Display',serif!important;font-size:26px;color:white;margin-bottom:6px}
.result-desc-text{font-size:13px;color:rgba(255,255,255,.88);line-height:1.6}
.conf-pill{display:inline-block;background:rgba(255,255,255,.25);border:1px solid rgba(255,255,255,.4);border-radius:20px;padding:4px 14px;font-size:13px;color:white;font-weight:500;margin-top:12px}
.prob-section{background:rgba(255,255,255,.7);border-radius:0 0 20px 20px;padding:22px}
.prob-label-top{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#8a6070;font-weight:500;margin-bottom:14px}
.prob-row-item{margin-bottom:12px}
.prob-name-row{display:flex;justify-content:space-between;font-size:13px;color:#3d2a35;margin-bottom:4px}
.prob-name-row span{color:#8a6070;font-size:12px}
.prob-track{height:8px;background:rgba(0,0,0,.06);border-radius:10px;overflow:hidden}
.prob-fill-green{height:100%;border-radius:10px;background:linear-gradient(90deg,#a8c5a0,#5a8a52)}
.prob-fill-rose{height:100%;border-radius:10px;background:linear-gradient(90deg,#f2a7bc,#d4607e)}
.prob-fill-purple{height:100%;border-radius:10px;background:linear-gradient(90deg,#d4b8e0,#8e5aad)}
.tip-item{display:flex;align-items:flex-start;gap:10px;padding:10px 12px;border-radius:12px;margin-bottom:8px;background:rgba(255,255,255,.5);border:1px solid rgba(0,0,0,.05);font-size:13px;color:#3d2a35;line-height:1.5}
.tip-dot{width:18px;height:18px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:10px;flex-shrink:0;margin-top:1px}
.stButton>button{background:linear-gradient(135deg,#f2a7bc,#d4607e)!important;color:white!important;border:none!important;border-radius:30px!important;padding:10px 28px!important;font-weight:500!important;box-shadow:0 4px 20px rgba(212,96,126,.3)!important}
[data-testid="stImage"] img{border-radius:20px;border:3px solid rgba(242,167,188,.4)}
hr{border-color:rgba(242,167,188,.25)!important}
.footer-text{text-align:center;font-size:12px;color:#8a6070;margin-top:32px;padding-bottom:16px}
.girly-card{background:rgba(255,255,255,.65);border:1px solid rgba(242,167,188,.3);border-radius:24px;padding:24px;margin-bottom:18px}
</style>
""", unsafe_allow_html=True)

IMG_SIZE    = 224
CLASS_NAMES = ["Healthy", "Early Blight", "Septoria Leaf Spot"]
CLASS_INFO  = {
    "Healthy": {
        "emoji": "🌱",
        "header_class": "result-header-healthy",
        "desc": "Your tomato plant looks healthy and thriving! Keep up the great care.",
        "dots": ["#5a8a52","#d4607e","#8e5aad","#c8602a"],
        "tips": [
            "Keep watering schedule consistent and even.",
            "Monitor regularly for any early warning signs.",
            "Ensure good air circulation around your plants.",
            "Feed with balanced fertiliser every 2-3 weeks."
        ]
    },
    "Early Blight": {
        "emoji": "🍂",
        "header_class": "result-header-early",
        "desc": "Early Blight (Alternaria solani) detected — a common fungal infection.",
        "dots": ["#d4607e","#c8602a","#5a8a52","#8e5aad"],
        "tips": [
            "Remove and destroy all infected leaves immediately.",
            "Apply a copper-based fungicide spray weekly.",
            "Water at the base only — avoid overhead watering.",
            "Rotate your crops in the next growing season."
        ]
    },
    "Septoria Leaf Spot": {
        "emoji": "🔴",
        "header_class": "result-header-septoria",
        "desc": "Septoria Leaf Spot (Septoria lycopersici) found — act quickly!",
        "dots": ["#8e5aad","#d4607e","#c8602a","#5a8a52"],
        "tips": [
            "Remove affected leaves and dispose carefully.",
            "Apply chlorothalonil or mancozeb fungicide.",
            "Water at soil level — never overhead.",
            "Space plants further apart to improve airflow."
        ]
    }
}
FILL_CLASSES = ["prob-fill-green", "prob-fill-rose", "prob-fill-purple"]

@st.cache_resource
def load_model():
    if not os.path.exists("best_model.keras"):
        st.error("Model file not found!")
        return None
    try:
        return tf.keras.models.load_model("best_model.keras")
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

# ✅ FIXED — simple /255.0 (same as training)
def preprocess(img):
    arr = np.array(
        img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
        dtype=np.float32
    ) / 255.0
    return arr

def tta_predict(model, arr):
    augs = [
        arr,
        np.fliplr(arr),
        np.flipud(arr),
        np.fliplr(np.flipud(arr)),
        np.clip(arr + 0.1, 0, 1),
        np.clip(arr - 0.1, 0, 1),
    ]
    return model.predict(np.stack(augs, 0), verbose=0).mean(0)

# Hero
st.markdown('<div class="badge-row"><div class="badge-pill">AI-Powered Plant Care</div></div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Tomato Leaf <em>Disease Detector</em> 🍅</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload a leaf photo — our AI will tell you if your plant needs care 🌿</p>', unsafe_allow_html=True)

st.markdown("""
<div class="chips-row">
  <div class="chip"><span class="chip-icon">🌱</span><span class="chip-name">Healthy</span></div>
  <div class="chip"><span class="chip-icon">🍂</span><span class="chip-name">Early Blight</span></div>
  <div class="chip"><span class="chip-icon">🔴</span><span class="chip-name">Septoria Spot</span></div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    **Model:** ResNet50V2 (Transfer Learning)  
    **Dataset:** PlantVillage · 500 imgs/class  
    **Seed:** 7282  
    **Accuracy:** 97%  
    **Input:** 224×224 px  
    **Augmentation:** 5 standard + Black Patch  
    **TTA:** 6-fold averaging
    """)
    st.divider()
    use_tta = st.toggle("Use TTA (better accuracy)", value=True)
    st.markdown("*Averages predictions over 6 augmentations.*")

model = load_model()

uploaded = st.file_uploader(
    "Drop your tomato leaf photo here",
    type=["jpg", "jpeg", "png"]
)

if uploaded and model:
    img = Image.open(uploaded)
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.image(img, caption="Your leaf", use_container_width=True)
    with col2:
        st.markdown(f"**Ready to analyse!**")
        st.markdown(f"File: `{uploaded.name}`")
        if st.button("Analyse Leaf"):
            with st.spinner("Analysing... 🌿"):
                arr   = preprocess(img)
                probs = tta_predict(model, arr) if use_tta else \
                        model.predict(np.expand_dims(arr, 0), verbose=0)[0]

            idx  = int(np.argmax(probs))
            cls  = CLASS_NAMES[idx]
            conf = float(probs[idx]) * 100
            info = CLASS_INFO[cls]
            tta_l = " · TTA" if use_tta else ""

            st.markdown(f"""
            <div class="{info['header_class']}">
              <div style="font-size:40px;margin-bottom:10px">{info['emoji']}</div>
              <div class="result-class-name">{cls}</div>
              <div class="result-desc-text">{info['desc']}</div>
              <div class="conf-pill">{conf:.1f}% confident{tta_l}</div>
            </div>""", unsafe_allow_html=True)

            bars = '<div class="prob-section"><div class="prob-label-top">Class Probabilities</div>'
            for i, name in enumerate(CLASS_NAMES):
                pct = probs[i] * 100
                bars += f'''<div class="prob-row-item">
                    <div class="prob-name-row">{name}<span>{pct:.1f}%</span></div>
                    <div class="prob-track">
                      <div class="{FILL_CLASSES[i]}" style="width:{pct:.1f}%"></div>
                    </div></div>'''
            bars += '</div>'
            st.markdown(bars, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Care Recommendations**")
            for i, tip in enumerate(info["tips"]):
                c = info["dots"][i % 4]
                st.markdown(
                    f'<div class="tip-item"><div class="tip-dot" '
                    f'style="background:{c}22;color:{c}">✦</div>{tip}</div>',
                    unsafe_allow_html=True
                )

elif not model:
    st.warning("Add best_model.keras to your repo!")
else:
    st.markdown("""
    <div class="girly-card" style="text-align:center;padding:40px">
      <div style="font-size:40px;margin-bottom:12px">📷</div>
      <div style="font-family:'Playfair Display',serif;font-size:18px;color:#3d2a35;margin-bottom:6px">
        Upload a leaf to get started</div>
      <div style="font-size:13px;color:#8a6070">
        Our AI will instantly tell you your plant's health status 🌿</div>
    </div>""", unsafe_allow_html=True)

st.markdown(
    '<div class="footer-text">Built with ResNet50V2 · Transfer Learning · PlantVillage · Seed 7282</div>',
    unsafe_allow_html=True
)
