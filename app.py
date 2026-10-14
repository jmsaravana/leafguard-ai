# =============================================================================
# LeafGuard AI — Streamlit Web Application
# =============================================================================
# This file is the main Streamlit app. Run it with:
#   streamlit run app.py
#
# If a trained model (leafguard_model.keras) is found it is loaded and used for
# real predictions.  If the model file is missing, the app automatically falls
# back to DEMO MODE which shows clearly-labelled SIMULATED predictions.
# =============================================================================

import os
import random
import numpy as np
import streamlit as st
from PIL import Image

# ── Page configuration (must be the very first Streamlit call) ─────────────────
st.set_page_config(
    page_title="LeafGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Disease classes ────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Healthy",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Potato Early Blight",
    "Potato Late Blight",
]

# ── Precautions / advice per class ────────────────────────────────────────────
PRECAUTIONS = {
    "Healthy": {
        "status": "✅ Plant appears healthy!",
        "color": "#22c55e",
        "tips": [
            "Continue regular watering — avoid wetting foliage.",
            "Maintain proper plant spacing for good air circulation.",
            "Apply balanced fertiliser during the growing season.",
            "Monitor leaves weekly for early signs of disease.",
            "Remove weeds that compete for nutrients.",
        ],
    },
    "Tomato Early Blight": {
        "status": "⚠️ Tomato Early Blight Detected",
        "color": "#f59e0b",
        "tips": [
            "Remove and destroy infected leaves immediately.",
            "Apply copper-based fungicide as directed on the label.",
            "Water at the base; avoid overhead irrigation.",
            "Improve air circulation by pruning crowded stems.",
            "Rotate crops — do not plant tomatoes in the same spot next year.",
        ],
    },
    "Tomato Late Blight": {
        "status": "🔴 Tomato Late Blight Detected",
        "color": "#ef4444",
        "tips": [
            "Act immediately — late blight spreads very rapidly.",
            "Remove ALL infected plant material and dispose away from garden.",
            "Apply a systemic fungicide (e.g., mancozeb or chlorothalonil).",
            "Avoid working among wet plants to limit spread.",
            "Consider destroying severely infected plants to protect neighbours.",
        ],
    },
    "Tomato Leaf Mold": {
        "status": "⚠️ Tomato Leaf Mold Detected",
        "color": "#f59e0b",
        "tips": [
            "Reduce humidity inside greenhouses with better ventilation.",
            "Remove and dispose of affected leaves.",
            "Apply approved fungicide for leaf mold control.",
            "Space plants to improve airflow.",
            "Avoid excess nitrogen fertilisation.",
        ],
    },
    "Potato Early Blight": {
        "status": "⚠️ Potato Early Blight Detected",
        "color": "#f59e0b",
        "tips": [
            "Remove infected foliage as soon as spotted.",
            "Apply mancozeb or chlorothalonil fungicide preventively.",
            "Ensure adequate potassium nutrition.",
            "Avoid excessive nitrogen which promotes lush, susceptible growth.",
            "Rotate potato crops annually.",
        ],
    },
    "Potato Late Blight": {
        "status": "🔴 Potato Late Blight Detected",
        "color": "#ef4444",
        "tips": [
            "This is the same pathogen that caused the Irish Famine — act fast!",
            "Apply systemic fungicide (metalaxyl + mancozeb) immediately.",
            "Destroy infected haulms before harvest.",
            "Harvest tubers during dry weather and inspect carefully.",
            "Plant certified disease-free seed potatoes next season.",
        ],
    },
}

# ── Custom CSS (green-themed, modern look) ────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        /* ── Google Font ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ── Background ── */
        .stApp {
            background: linear-gradient(135deg, #0a1f0a 0%, #0f2d1a 40%, #0d2411 100%);
            min-height: 100vh;
        }

        /* ── Hero banner ── */
        .hero-banner {
            background: linear-gradient(135deg, #065f46 0%, #14532d 50%, #052e16 100%);
            border-radius: 20px;
            padding: 2.5rem 3rem;
            margin-bottom: 2rem;
            border: 1px solid #16a34a33;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        }
        .hero-banner h1 {
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(90deg, #4ade80, #86efac, #bbf7d0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 0.5rem 0;
        }
        .hero-banner p {
            color: #86efac;
            font-size: 1.1rem;
            margin: 0;
            line-height: 1.6;
        }

        /* ── Metric card ── */
        .metric-card {
            background: rgba(20, 83, 45, 0.5);
            border: 1px solid #16a34a55;
            border-radius: 14px;
            padding: 1.2rem 1.5rem;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .metric-card .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #4ade80;
        }
        .metric-card .metric-label {
            font-size: 0.8rem;
            color: #86efac;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        /* ── Prediction card ── */
        .pred-card {
            background: rgba(5, 46, 22, 0.8);
            border: 1px solid #16a34a44;
            border-radius: 16px;
            padding: 1.8rem;
            margin-top: 1rem;
            backdrop-filter: blur(12px);
        }

        /* ── Confidence bar wrapper ── */
        .conf-bar-wrap {
            background: rgba(0,0,0,0.3);
            border-radius: 999px;
            height: 12px;
            width: 100%;
            margin: 0.5rem 0 1rem;
        }
        .conf-bar-fill {
            height: 12px;
            border-radius: 999px;
            background: linear-gradient(90deg, #16a34a, #4ade80);
            transition: width 0.6s ease;
        }

        /* ── Warning banner ── */
        .warning-banner {
            background: rgba(120, 53, 15, 0.4);
            border: 1px solid #d97706;
            border-radius: 12px;
            padding: 1rem 1.4rem;
            color: #fde68a;
            font-size: 0.9rem;
            margin-top: 1.5rem;
        }

        /* ── Demo mode banner ── */
        .demo-banner {
            background: rgba(67, 20, 7, 0.5);
            border: 2px dashed #ef4444;
            border-radius: 12px;
            padding: 1rem 1.4rem;
            color: #fca5a5;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #052e16 0%, #0a1f0a 100%);
            border-right: 1px solid #16a34a33;
        }
        section[data-testid="stSidebar"] * {
            color: #d1fae5 !important;
        }

        /* ── Upload area ── */
        [data-testid="stFileUploader"] {
            border: 2px dashed #16a34a !important;
            border-radius: 12px !important;
            background: rgba(5, 46, 22, 0.4) !important;
        }

        /* ── Streamlit elements override ── */
        h2, h3, h4 { color: #4ade80 !important; }
        .stMarkdown p { color: #d1fae5; }

        /* ── Precaution list ── */
        .precaution-item {
            background: rgba(20, 83, 45, 0.4);
            border-left: 3px solid #16a34a;
            border-radius: 0 8px 8px 0;
            padding: 0.6rem 1rem;
            margin: 0.4rem 0;
            color: #d1fae5;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Try to load the trained Keras model.
    Returns (model, True) on success, (None, False) if model file is missing.
    """
    model_path = "leafguard_model.keras"
    if not os.path.exists(model_path):
        return None, False
    try:
        import tensorflow as tf  # import deferred so app starts without TF
        model = tf.keras.models.load_model(model_path)
        return model, True
    except Exception as exc:
        st.error(f"Model file found but could not be loaded: {exc}")
        return None, False


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize to 224×224, normalise pixel values to [0, 1], add batch dimension.
    """
    img = image.convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)


def predict_real(model, img_array: np.ndarray):
    """Run genuine model inference and return (class_name, confidence)."""
    preds = model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx])


def predict_demo(img_array: np.ndarray):
    """
    Generate a clearly SIMULATED prediction for demo mode.
    Uses a seeded hash of the image so the same image always gives the same demo
    result — but this is NOT real model inference.
    """
    seed = int(np.sum(img_array * 255)) % (2 ** 31)
    rng = random.Random(seed)
    idx = rng.randint(0, len(CLASS_NAMES) - 1)
    confidence = rng.uniform(0.55, 0.95)
    return CLASS_NAMES[idx], confidence


def render_confidence_bar(confidence: float):
    pct = int(confidence * 100)
    bar_color = "#22c55e" if pct >= 70 else "#f59e0b" if pct >= 50 else "#ef4444"
    st.markdown(
        f"""
        <div class="conf-bar-wrap">
          <div class="conf-bar-fill" style="width:{pct}%; background: linear-gradient(90deg,{bar_color},{bar_color}aa);"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction(class_name: str, confidence: float, demo_mode: bool):
    info = PRECAUTIONS[class_name]
    label_suffix = " ⚠️ [SIMULATED — Demo Mode]" if demo_mode else ""

    st.markdown(
        f"""
        <div class="pred-card">
          <div style="font-size:1.5rem; font-weight:700; color:{info['color']};">
            {info['status']}{label_suffix}
          </div>
          <div style="margin-top:0.3rem; color:#86efac; font-size:0.95rem;">
            Predicted class: <strong style="color:#d1fae5;">{class_name}</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📊 Model Confidence")
    render_confidence_bar(confidence)
    st.markdown(
        f"<span style='color:#4ade80; font-size:1.3rem; font-weight:600;'>{confidence*100:.1f}%</span>",
        unsafe_allow_html=True,
    )

    if demo_mode:
        st.markdown(
            """
            <div class="demo-banner">
              🚧 <strong>DEMO MODE — No Trained Model Found</strong><br>
              The confidence score and class above are <em>randomly simulated</em> for
              demonstration purposes only.  They do <strong>not</strong> reflect any real
              analysis of your image.  Train the model first (see the sidebar for
              instructions) to obtain genuine predictions.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### 🛡️ Recommended Precautions")
    for tip in info["tips"]:
        st.markdown(f'<div class="precaution-item">• {tip}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="warning-banner">
          ⚠️ <strong>Disclaimer:</strong> Predictions made by this tool are
          <strong>experimental</strong> and intended for educational purposes only.
          They do <strong>not</strong> replace professional agricultural advice.
          Always consult a qualified agronomist or plant pathologist before taking
          action on your crops.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main application ───────────────────────────────────────────────────────────

def main():
    inject_css()

    # ── Load (or skip) model ─────────────────────────────────────────────────
    model, model_loaded = load_model()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🌿 LeafGuard AI")
        st.markdown("---")

        if model_loaded:
            st.success("✅ Trained model loaded")
        else:
            st.warning("⚠️ No model found — running in **Demo Mode**")
            st.markdown(
                """
                **To enable real predictions:**
                1. Prepare your dataset in `dataset/` with one folder per class.
                2. Run the training script:
                   ```
                   python train_model.py
                   ```
                3. Restart this app after training.
                """
            )

        st.markdown("---")
        st.markdown("### 📋 Supported Classes")
        for cls in CLASS_NAMES:
            icon = "✅" if cls == "Healthy" else "🍃"
            st.markdown(f"{icon} {cls}")

        st.markdown("---")
        st.markdown("### ℹ️ How to Use")
        st.markdown(
            """
            1. Upload a clear photo of a **single leaf**.
            2. Make sure the leaf is the main subject.
            3. Click **Analyse Leaf** to get a prediction.
            4. Read the precautions carefully.
            """
        )

        st.markdown("---")
        st.markdown(
            "<small style='color:#6b7280;'>LeafGuard AI • Educational Project</small>",
            unsafe_allow_html=True,
        )

    # ── Hero banner ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-banner">
          <h1>🌿 LeafGuard AI</h1>
          <p>Early Plant Disease Detection Using Deep Learning<br>
          <span style="font-size:0.95rem; color:#6ee7b7;">
            Upload a leaf image to detect diseases in tomato and potato plants.
          </span>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Stats row ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">6</div>'
            '<div class="metric-label">Disease Classes</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">224px</div>'
            '<div class="metric-label">Input Resolution</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">CNN</div>'
            '<div class="metric-label">Model Architecture</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        mode_label = "Live Model" if model_loaded else "Demo Mode"
        mode_icon = "🟢" if model_loaded else "🟡"
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{mode_icon}</div>'
            f'<div class="metric-label">{mode_label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Upload & inference ────────────────────────────────────────────────────
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown("### 📤 Upload Leaf Image")
        uploaded_file = st.file_uploader(
            label="Choose a JPG, JPEG, or PNG image of a plant leaf",
            type=["jpg", "jpeg", "png"],
            help="For best results use a clear, well-lit photo with the leaf as the main subject.",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
            st.markdown(
                f"<small style='color:#6ee7b7;'>📐 Original size: {image.width} × {image.height} px → resized to 224 × 224 for inference</small>",
                unsafe_allow_html=True,
            )

            analyse_btn = st.button(
                "🔍 Analyse Leaf",
                type="primary",
                use_container_width=True,
            )
        else:
            st.info("👆 Please upload a leaf image to get started.")
            analyse_btn = False

    with right_col:
        st.markdown("### 🧠 Analysis Results")

        if uploaded_file is not None and analyse_btn:
            with st.spinner("Preprocessing image and running inference …"):
                img_array = preprocess_image(image)

                if model_loaded:
                    class_name, confidence = predict_real(model, img_array)
                    demo_mode = False
                else:
                    class_name, confidence = predict_demo(img_array)
                    demo_mode = True

            render_prediction(class_name, confidence, demo_mode)

        elif uploaded_file is not None:
            st.markdown(
                """
                <div style="color:#86efac; padding:2rem; text-align:center;
                            border:1px dashed #16a34a; border-radius:12px;">
                  👈 Click <strong>Analyse Leaf</strong> to run the prediction.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="color:#6b7280; padding:2rem; text-align:center;
                            border:1px dashed #374151; border-radius:12px;">
                  Results will appear here after you upload an image.
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
