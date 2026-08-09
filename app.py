"""
app.py
------
CodeAlpha Machine Learning Internship - Task 3: Handwritten Character Recognition
Author: Wareesha Khan

A Streamlit demo where a user draws (or uploads) a handwritten digit or
letter and a CNN trained on EMNIST Balanced predicts what it is.

Run:
    streamlit run app.py
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

from utils import preprocess_canvas_image, top_k_predictions, EMNIST_BALANCED_MAPPING

MODEL_PATH = os.path.join('model', 'handwritten_char_cnn.keras')
MAPPING_PATH = os.path.join('model', 'label_mapping.json')
HISTORY_PATH = 'prediction_history.csv'
HISTORY_COLUMNS = ["Time", "Character", "Confidence (%)", "Source"]

st.set_page_config(
    page_title="Handwritten Character Recognition | Wareesha Khan",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling - a distinct, non-default look (deep indigo + gold accent)
# ---------------------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .brand-title { font-family: 'Space Grotesk', sans-serif !important; }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #1b1035 0%, #0e0a1f 45%, #0a0715 100%);
        color: #EDEBF5;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #17102e 0%, #0d0a1c 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .hero {
        padding: 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(124,58,237,0.25), rgba(236,180,60,0.10));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.4rem;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: #fff; }
    .hero p  { margin: .35rem 0 0 0; color: #C9C4DA; font-size: 0.98rem; }
    .badge {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        background: rgba(236,180,60,0.15); color: #F0C767; font-size: 0.75rem;
        border: 1px solid rgba(236,180,60,0.35); margin-right: 6px;
    }

    .card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }

    .prediction-char {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 5.5rem; font-weight: 700; text-align: center;
        background: linear-gradient(135deg, #EDB84C, #7C3AED);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0.2rem 0;
    }
    .confidence-tag {
        text-align: center; color: #C9C4DA; font-size: 0.95rem; margin-bottom: .6rem;
    }

    .sidebar-name { font-size: 1.15rem; font-weight: 700; color: #fff; margin-bottom: 0; }
    .sidebar-role { font-size: 0.85rem; color: #B7ADD1; margin-top: 0; }

    .footer-credit {
        text-align: center; color: #8B84A3; font-size: 0.8rem;
        margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.06);
    }

    .stButton>button {
        background: linear-gradient(135deg, #7C3AED, #5B21B6);
        color: white; border: none; border-radius: 10px; font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
    .stButton>button:hover { background: linear-gradient(135deg, #8B5CF6, #6D28D9); color: white; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar - author branding + project info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ✍️ Handwritten Character Recognition")
    st.markdown(
        """
        <p class="sidebar-name">Wareesha Khan</p>
        <p class="sidebar-role">Machine Learning Intern @ CodeAlpha</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**Task 3 — CodeAlpha ML Internship**")
    st.markdown(
        """
        - **Objective:** Identify handwritten characters & digits
        - **Dataset:** EMNIST Balanced (47 classes)
        - **Model:** Convolutional Neural Network (CNN)
        - **Framework:** TensorFlow / Keras + Streamlit
        """
    )
    st.markdown("---")
    st.caption("Built as part of the CodeAlpha Machine Learning Internship.")


# ---------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH) as f:
            mapping = {int(k): v for k, v in json.load(f).items()}
    else:
        mapping = EMNIST_BALANCED_MAPPING
    return model, mapping


model, mapping = load_trained_model()

# ---------------------------------------------------------------------------
# Prediction history - persisted to a local CSV so it survives app restarts,
# and mirrored in session_state for instant display without re-reading disk.
# ---------------------------------------------------------------------------
def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_PATH):
        try:
            return pd.read_csv(HISTORY_PATH)
        except Exception:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def log_prediction(character: str, confidence: float, source: str):
    entry = pd.DataFrame([{
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Character": character,
        "Confidence (%)": round(confidence, 1),
        "Source": source,
    }])
    write_header = not os.path.exists(HISTORY_PATH)
    entry.to_csv(HISTORY_PATH, mode='a', header=write_header, index=False)
    st.session_state.history = pd.concat([entry, st.session_state.history], ignore_index=True)


if "history" not in st.session_state:
    st.session_state.history = load_history()

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <span class="badge">CodeAlpha ML Internship</span><span class="badge">Task 3</span>
        <h1>✍️ Handwritten Character Recognition</h1>
        <p>Draw a digit (0-9) or a letter (A-Z / a-z) and let a CNN trained on the
        EMNIST dataset read your handwriting in real time.<br>Developed by
        <b>Wareesha Khan</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.warning(
        "⚠️ No trained model found yet at `model/handwritten_char_cnn.keras`.\n\n"
        "Run **`python train_model.py`** first (see the README for setup notes — "
        "the EMNIST download needs an unrestricted internet connection, so Google "
        "Colab is the easiest place to train if your local network blocks it). "
        "Once training finishes, refresh this page."
    )

tab_draw, tab_upload, tab_history = st.tabs(
    ["🖌️ Draw a Character", "📁 Upload an Image", "📜 History"]
)

image_array = None
predict_clicked = False
prediction_source = None

with tab_draw:
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Draw here")
        stroke_width = st.slider("Pen thickness", 8, 25, 16)
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=stroke_width,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        predict_clicked = st.button("🔍 Predict from Drawing", use_container_width=True)

    with right:
        st.markdown("#### Prediction")
        st.caption("Draw on the left, then click **Predict from Drawing**.")

    if predict_clicked and canvas_result.image_data is not None:
        image_array = canvas_result.image_data.astype('uint8')
        prediction_source = "Drawing"

with tab_upload:
    st.markdown("#### Upload a photo of a single handwritten character")
    uploaded = st.file_uploader("PNG or JPG", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        from PIL import Image
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="Uploaded image", width=200)
        if st.button("🔍 Predict from Upload"):
            image_array = np.array(pil_img)
            prediction_source = "Upload"


def render_prediction(img_array, source: str):
    if model is None:
        st.error("Train the model first — see the warning above.")
        return
    processed = preprocess_canvas_image(img_array)
    probs = model.predict(processed, verbose=0)[0]
    results = top_k_predictions(probs, k=5)
    top_char, top_conf = results[0]

    st.markdown(f'<div class="prediction-char">{top_char}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence-tag">Confidence: {top_conf:.1f}%</div>', unsafe_allow_html=True)

    df = pd.DataFrame(results, columns=["Character", "Confidence (%)"]).set_index("Character")
    st.bar_chart(df)

    log_prediction(top_char, top_conf, source)


if image_array is not None:
    with tab_draw if prediction_source == "Drawing" else tab_upload:
        st.markdown("---")
        st.markdown("### Result")
        render_prediction(image_array, prediction_source)

with tab_history:
    st.markdown("#### Prediction History")
    hist_df = st.session_state.history

    if hist_df.empty:
        st.markdown(
            '<div class="card">No predictions yet. Draw or upload a character to get started — '
            'every prediction is logged here automatically.</div>',
            unsafe_allow_html=True,
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total predictions", len(hist_df))
        c2.metric("Most predicted", hist_df["Character"].mode().iloc[0])
        c3.metric("Avg. confidence", f'{hist_df["Confidence (%)"].mean():.1f}%')

        st.dataframe(hist_df, use_container_width=True, height=280, hide_index=True)

        dl_col, clear_col = st.columns([1, 1])
        dl_col.download_button(
            "⬇️ Download full history (CSV)",
            hist_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if clear_col.button("🗑️ Clear history", use_container_width=True):
            st.session_state.history = pd.DataFrame(columns=HISTORY_COLUMNS)
            if os.path.exists(HISTORY_PATH):
                os.remove(HISTORY_PATH)
            st.rerun()

st.markdown(
    """
    <div class="footer-credit">
        © 2026 Wareesha Khan · Handwritten Character Recognition · CodeAlpha Machine Learning Internship
    </div>
    """,
    unsafe_allow_html=True,
)
