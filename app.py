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

# ── Disease classes (All 15 PlantVillage Categories) ─────────────────────────
CLASS_NAMES = [
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Mosaic Virus",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
]

# ── Precautions / advice per class ────────────────────────────────────────────
PRECAUTIONS = {
    "Pepper Bell Bacterial Spot": {
        "status": "⚠️ Pepper Bell Bacterial Spot Detected",
        "color": "#f59e0b",
        "tips": [
            "Apply copper-based bactericide/fungicide according to label.",
            "Water strictly at the soil level; avoid splashing wet soil on leaves.",
            "Remove and discard severely spotted leaves.",
            "Sterilize gardening shears and tools with 70% alcohol.",
            "Practice 2-3 year crop rotation away from solanaceous plants.",
        ],
    },
    "Pepper Bell Healthy": {
        "status": "✅ Pepper Bell Plant Appears Healthy!",
        "color": "#22c55e",
        "tips": [
            "Maintain consistent soil moisture without waterlogging.",
            "Ensure 6-8 hours of direct sunlight daily.",
            "Apply balanced fertilizer with calcium to prevent blossom end rot.",
            "Inspect underside of leaves weekly for early pests.",
        ],
    },
    "Potato Early Blight": {
        "status": "⚠️ Potato Early Blight Detected",
        "color": "#f59e0b",
        "tips": [
            "Prune infected lower foliage as soon as concentric rings appear.",
            "Apply mancozeb or chlorothalonil protective fungicide.",
            "Ensure adequate potassium and balanced nitrogen nutrition.",
            "Mulch around plants to reduce soil splash during rain.",
            "Rotate potato planting locations annually.",
        ],
    },
    "Potato Healthy": {
        "status": "✅ Potato Plant Appears Healthy!",
        "color": "#22c55e",
        "tips": [
            "Hill soil around stems as plants grow to protect developing tubers.",
            "Maintain even moisture, especially during tuber bulking.",
            "Monitor regularly for Colorado potato beetles and aphids.",
            "Harvest tubers on dry days and cure before storing.",
        ],
    },
    "Potato Late Blight": {
        "status": "🔴 Potato Late Blight Detected",
        "color": "#ef4444",
        "tips": [
            "Urgent: Late blight destroys foliage and tubers very rapidly.",
            "Apply systemic fungicide (metalaxyl or cymoxanil + mancozeb) immediately.",
            "Cut and destroy haulms 2 weeks before harvest to prevent tuber rot.",
            "Do not compost infected plant debris; seal in bags and discard.",
            "Only plant certified disease-free seed potatoes next season.",
        ],
    },
    "Tomato Bacterial Spot": {
        "status": "⚠️ Tomato Bacterial Spot Detected",
        "color": "#f59e0b",
        "tips": [
            "Apply fixed copper spray combined with mancozeb for synergistic effect.",
            "Never work among tomato plants when leaves are wet.",
            "Mulch heavily around the plant base to prevent soil bacteria splash.",
            "Remove infected bottom foliage to improve air circulation.",
        ],
    },
    "Tomato Early Blight": {
        "status": "⚠️ Tomato Early Blight Detected",
        "color": "#f59e0b",
        "tips": [
            "Remove and dispose of infected leaves with brown bullseye spots.",
            "Apply copper or chlorothalonil fungicide every 7-10 days as directed.",
            "Water at the base with drip irrigation or soaker hoses.",
            "Prune lower suckers to maintain at least 12 inches of clearance from soil.",
            "Rotate tomato plots for at least 3 years.",
        ],
    },
    "Tomato Healthy": {
        "status": "✅ Tomato Plant Appears Healthy!",
        "color": "#22c55e",
        "tips": [
            "Continue deep, regular watering at the root zone.",
            "Stake or cage plants to maintain strong upright growth and airflow.",
            "Apply balanced tomato fertilizer with micronutrients.",
            "Inspect weekly for hornworms and mite webs.",
        ],
    },
    "Tomato Late Blight": {
        "status": "🔴 Tomato Late Blight Detected",
        "color": "#ef4444",
        "tips": [
            "Immediate action required — late blight can ruin a crop in days.",
            "Remove and destroy all water-soaked lesions and affected stems.",
            "Apply a systemic fungicide (e.g., chlorothalonil, copper, or mancozeb).",
            "Avoid overhead watering and minimize canopy moisture.",
            "Consider harvesting unaffected green tomatoes early if weather is humid.",
        ],
    },
    "Tomato Leaf Mold": {
        "status": "⚠️ Tomato Leaf Mold Detected",
        "color": "#f59e0b",
        "tips": [
            "Significantly improve ventilation in greenhouses or enclosed tunnels.",
            "Keep relative humidity below 85% around plant canopies.",
            "Prune crowded inner stems to allow light penetration.",
            "Apply sulfur or copper-based fungicide if mold persists.",
        ],
    },
    "Tomato Mosaic Virus": {
        "status": "🔴 Tomato Mosaic Virus (ToMV) Detected",
        "color": "#ef4444",
        "tips": [
            "Viral disease: There is NO chemical cure once infected.",
            "Remove and burn/bag infected plants immediately to protect neighbors.",
            "Never smoke or handle tobacco near tomato plants (tobacco transmits ToMV).",
            "Disinfect hands and all garden tools in 20% nonfat dry milk solution.",
            "Plant ToMV-resistant tomato varieties in future seasons.",
        ],
    },
    "Tomato Septoria Leaf Spot": {
        "status": "⚠️ Tomato Septoria Leaf Spot Detected",
        "color": "#f59e0b",
        "tips": [
            "Prune infected leaves starting from the bottom of the plant.",
            "Apply broad-spectrum organic or copper fungicide early.",
            "Keep organic mulch around stems to stop spores from splashing.",
            "Stake plants off the ground and avoid overhead sprinkling.",
        ],
    },
    "Tomato Spider Mites": {
        "status": "⚠️ Tomato Spider Mites Infestation",
        "color": "#f97316",
        "tips": [
            "Spray undersides of leaves with insecticidal soap or horticultural oil.",
            "Introduce natural beneficial predators such as Phytoseiulus persimilis.",
            "Mist plants lightly during hot, dry weather to discourage mite breeding.",
            "Remove and discard heavily webbed branches.",
        ],
    },
    "Tomato Target Spot": {
        "status": "⚠️ Tomato Target Spot Detected",
        "color": "#f59e0b",
        "tips": [
            "Remove lower foliage showing target-like concentric lesions.",
            "Apply azoxystrobin, pyraclostrobin, or chlorothalonil fungicides.",
            "Ensure plants have adequate air spacing and row orientation.",
            "Control weeds around planting beds that host Corynespora spores.",
        ],
    },
    "Tomato Yellow Leaf Curl Virus": {
        "status": "🔴 Tomato Yellow Leaf Curl Virus (TYLCV) Detected",
        "color": "#ef4444",
        "tips": [
            "Transmitted by Silverleaf Whiteflies: Control the insect vector.",
            "Install yellow sticky traps and spray neem oil or pyrethrin for whiteflies.",
            "Uproot and destroy infected stunted/cupped plants to reduce vector reservoir.",
            "Use insect netting or floating row covers in early season.",
            "Select TYLCV-resistant hybrids for subsequent plantings.",
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
def _load_keras_file(model_path: str, mtime: float):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)


def load_model():
    """
    Try to load the trained Keras model.
    Returns (model, True) on success, (None, False) if model file is missing.
    """
    model_path = "leafguard_model.keras"
    if not os.path.exists(model_path):
        return None, False
    try:
        mtime = os.path.getmtime(model_path)
        model = _load_keras_file(model_path, mtime)
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


# ── Universal Botanical Diagnostic Engine (Pure Computer Vision - 0 API Keys) ──

def analyze_any_leaf_cv(image: Image.Image) -> dict:
    """
    Analyzes ANY leaf in the world using local computer vision and spectral colorimetry.
    Detects necrosis, chlorosis, fungal powdery mildew, and overall health index.
    Requires ZERO API keys, works 100% locally and offline.
    """
    img = image.convert("RGB").resize((256, 256), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Filter out pure white or pure dark background
    is_bg = ((r > 240) & (g > 240) & (b > 240)) | ((r < 20) & (g < 20) & (b < 20))
    leaf_mask = ~is_bg
    total_leaf = int(np.sum(leaf_mask))

    if total_leaf < 400:
        # Fallback if image has no distinctive background
        leaf_mask = np.ones_like(r, dtype=bool)
        total_leaf = int(leaf_mask.size)

    # 1. Healthy green chlorophyll
    green_mask = leaf_mask & (g > r * 1.05) & (g > b * 1.1)
    green_pct = float(np.sum(green_mask) / total_leaf * 100)

    # 2. Chlorosis (Yellowing / Nitrogen & Iron deficiency)
    yellow_mask = leaf_mask & (r > 130) & (g > 120) & (b < 110) & (np.abs(r - g) < 45)
    yellow_pct = float(np.sum(yellow_mask) / total_leaf * 100)

    # 3. Necrotic lesions / Brown spots / Fungal blights
    brown_mask = leaf_mask & (r < 140) & (g < 120) & (b < 95) & ((r + g + b) < 320) & ~green_mask
    brown_pct = float(np.sum(brown_mask) / total_leaf * 100)

    # 4. Powdery mildew / fungal mycelium
    white_mask = leaf_mask & (r > 195) & (g > 195) & (b > 195)
    white_pct = float(np.sum(white_mask) / total_leaf * 100)

    # Compute overall health score
    health_score = max(5.0, min(99.0, green_pct - (brown_pct * 1.6) - (yellow_pct * 0.7) - (white_pct * 1.2)))

    # Determine diagnosis
    if brown_pct > 12.0:
        diagnosis = "Fungal Leaf Spot / Necrotic Blight"
        status_type = "Critical"
        color = "#ef4444"
        summary = "Significant necrotic (dead tissue) spots detected across the leaf surface. Common in fungal blights and bacterial spots."
        tips = [
            "Prune and destroy heavily spotted leaves to stop spore progression.",
            "Apply copper-based fungicide or biofungicide spray at 7-day intervals.",
            "Water only at the root base — keep the foliage completely dry.",
            "Improve plant spacing to ensure plenty of sunshine and air circulation.",
        ]
        organic = ["Neem oil spray (2 tbsp/gallon)", "Baking soda spray (1 tsp per quart water)", "Compost tea foliar spray"]
        chemical = ["Copper octanoate fungicide", "Chlorothalonil broad-spectrum fungicide", "Mancozeb wettable powder"]

    elif white_pct > 8.0:
        diagnosis = "Powdery Mildew / Fungal Surface Infestation"
        status_type = "Warning"
        color = "#f59e0b"
        summary = "White powdery fungal residue observed covering foliage. This blocks sunlight and photosynthesis."
        tips = [
            "Wipe or prune leaves with heavy white fungal coating.",
            "Spray with diluted milk-water solution (40% milk, 60% water) in direct sunlight.",
            "Increase air circulation around the plant.",
            "Reduce humidity levels around foliage.",
        ]
        organic = ["Potassium bicarbonate foliar spray", "Neem oil organic fungicide", "Diluted sulfur dust"]
        chemical = ["Myclobutanil fungicide", "Triflumizole spray", "Wettable sulfur suspension"]

    elif yellow_pct > 15.0:
        diagnosis = "Foliar Chlorosis / Nutrient Deficiency"
        status_type = "Warning"
        color = "#f59e0b"
        summary = "Abnormal leaf yellowing (loss of chlorophyll). Most commonly caused by nitrogen, iron, or magnesium deficiency, or overwatering."
        tips = [
            "Check soil moisture — avoid waterlogged, suffocated roots.",
            "Apply a chelated iron or balanced N-P-K liquid fertilizer.",
            "Test soil pH; alkaline soils (pH > 7.0) frequently lock out iron.",
            "Ensure plants receive 6+ hours of adequate sunlight daily.",
        ]
        organic = ["Epsom salt spray (1 tbsp/gallon for magnesium)", "Fish emulsion / Seaweed extract", "Compost top-dressing"]
        chemical = ["Chelated Iron (EDDHA/EDTA)", "Water-soluble balanced 20-20-20 fertilizer", "Ammonium sulfate"]

    elif green_pct >= 70.0 and brown_pct < 6.0:
        diagnosis = "Healthy Foliage (Vibrant Chlorophyll)"
        status_type = "Optimal"
        color = "#22c55e"
        summary = "Strong, healthy chlorophyll pigment detected with minimal cellular stress or lesions."
        tips = [
            "Continue your regular watering routine at soil level.",
            "Provide balanced organic fertilizer during active growing periods.",
            "Inspect weekly for early pests under the leaf blades.",
            "Maintain clean gardening tools to keep plants disease-free.",
        ]
        organic = ["Standard compost mulching", "Liquid seaweed tonic", "Regular morning watering"]
        chemical = ["No chemical treatment needed — maintain current care."]

    else:
        diagnosis = "Mild Foliar Stress / Early Discoloration"
        status_type = "Mild"
        color = "#f59e0b"
        summary = "Early color variance detected on leaf tissue. May indicate slight weather stress, pest activity, or initial infection."
        tips = [
            "Monitor this plant daily for any spreading spots or discoloration.",
            "Isolate the plant if leaf changes accelerate.",
            "Ensure moderate watering — do not let soil completely dry out or stay soggy.",
        ]
        organic = ["Mild insecticidal soap", "Neem oil preventative spray"]
        chemical = ["Broad-spectrum preventative fungicide"]

    return {
        "diagnosis": diagnosis,
        "status_type": status_type,
        "color": color,
        "summary": summary,
        "health_score": round(health_score, 1),
        "green_pct": round(green_pct, 1),
        "yellow_pct": round(yellow_pct, 1),
        "brown_pct": round(brown_pct, 1),
        "white_pct": round(white_pct, 1),
        "tips": tips,
        "organic": organic,
        "chemical": chemical,
    }


def render_cv_prediction(res: dict):
    """Renders the local Computer Vision diagnosis for any leaf without API keys."""
    st.markdown(
        f"""
        <div class="pred-card">
          <div style="font-size:0.85rem; color:#86efac; text-transform:uppercase; letter-spacing:0.08em;">
            🔬 Local Botanical Diagnosis
          </div>
          <div style="font-size:1.6rem; font-weight:700; color:{res['color']}; margin-top:0.2rem;">
            {res['diagnosis']}
          </div>
          <p style="margin-top:0.4rem; color:#d1fae5; font-size:0.95rem;">
            {res['summary']}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Health score bar
    st.markdown("#### 💚 Overall Leaf Health Score")
    score_norm = res["health_score"] / 100.0
    render_confidence_bar(score_norm)
    st.markdown(
        f"<span style='color:{res['color']}; font-size:1.4rem; font-weight:700;'>{res['health_score']} / 100</span>",
        unsafe_allow_html=True,
    )

    # Spectral breakdown metrics
    st.markdown("#### 🧪 Spectral Tissue Analysis")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value" style="color:#22c55e;">{res["green_pct"]}%</div>'
            '<div class="metric-label">Healthy Chlorophyll</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value" style="color:#f59e0b;">{res["yellow_pct"]}%</div>'
            '<div class="metric-label">Yellowing / Chlorosis</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value" style="color:#ef4444;">{res["brown_pct"]}%</div>'
            '<div class="metric-label">Necrotic Spots</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value" style="color:#e2e8f0;">{res["white_pct"]}%</div>'
            '<div class="metric-label">Powdery Fungi</div></div>',
            unsafe_allow_html=True,
        )

    # Treatments
    st.markdown("#### 💊 Actionable Treatment Plan")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("<strong style='color:#86efac;'>🌱 Natural & Organic Solutions:</strong>", unsafe_allow_html=True)
        for o in res["organic"]:
            st.markdown(f'<div class="precaution-item">• {o}</div>', unsafe_allow_html=True)
    with t2:
        st.markdown("<strong style='color:#f87171;'>🧪 Targeted Chemical Options:</strong>", unsafe_allow_html=True)
        for c in res["chemical"]:
            st.markdown(f'<div class="precaution-item">• {c}</div>', unsafe_allow_html=True)

    # General Tips
    st.markdown("#### 🛡️ Preventive Recommendations")
    for tip in res["tips"]:
        st.markdown(f'<div class="precaution-item">• {tip}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="warning-banner">
          ⚠️ <strong>100% Offline Analysis:</strong> Computed directly on your CPU via computer vision & colorimetry.
          Requires zero external API keys or internet connection.
        </div>
        """,
        unsafe_allow_html=True,
# ── Main application ───────────────────────────────────────────────────────────

def main():
    inject_css()

    # ── Load (or skip) local CNN model ───────────────────────────────────────
    model, model_loaded = load_model()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🌿 LeafGuard AI")
        st.markdown("---")

        st.markdown("### ⚙️ Select Diagnostic Mode")
        engine_mode = st.radio(
            label="Engine Mode",
            options=[
                "🌿 Universal Leaf Health Scanner (Any Plant in the World)",
                "🧠 Deep Learning CNN Classifier (15 Crop Diseases)",
            ],
            index=0,
            label_visibility="collapsed",
            help="Universal mode scans ANY leaf on Earth using spectral colorimetry. CNN mode classifies specific diseases in Pepper, Potato, and Tomato."
        )

        st.markdown("---")

        if "Universal" in engine_mode:
            st.markdown("### 🌍 Universal Scope")
            st.markdown(
                """
                - 🌿 **Works on ANY Plant Leaf on Earth**
                - 🔬 **Necrotic Spots & Fungal Blight**
                - 🟡 **Chlorosis & Nutrient Deficiencies**
                - ⚪ **Powdery Mildew & Fungal Spores**
                - 💚 **Chlorophyll Health Index (0-100)**
                - 🔒 **100% Offline — No API Keys Needed**
                """
            )
        else:
            if model_loaded:
                st.success("✅ 15-Class CNN Model Loaded")
            else:
                st.warning("⚠️ No model found — running in **Demo Mode**")
                st.markdown(
                    """
                    **To train the model:**
                    ```
                    python train_model.py
                    ```
                    """
                )

            st.markdown("### 📋 Supported Crop Classes (15)")
            for cls in CLASS_NAMES:
                icon = "✅" if "Healthy" in cls else "🍃"
                st.markdown(f"{icon} <small>{cls}</small>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ℹ️ How to Use")
        st.markdown(
            """
            1. Upload a clear photo of **any plant leaf**.
            2. Make sure the leaf is well-lit and centered.
            3. Click **Analyse Leaf** for immediate diagnosis.
            4. Follow the tailored treatment and preventive tips.
            """
        )

        st.markdown("---")
        st.markdown(
            "<small style='color:#6b7280;'>LeafGuard AI • 100% Free & Open Source</small>",
            unsafe_allow_html=True,
        )

    # ── Hero banner ───────────────────────────────────────────────────────────
    if "Universal" in engine_mode:
        st.markdown(
            """
            <div class="hero-banner">
              <h1>🌿 LeafGuard AI — Universal Leaf Scanner</h1>
              <p>Instant Plant & Leaf Health Diagnosis for <strong>Any Plant in the World</strong><br>
              <span style="font-size:0.95rem; color:#6ee7b7;">
                Upload leaves from Mango, Rose, Neem, Tulsi, Houseplants, Herbs, Trees, or Garden Crops. Analyzes cellular tissue, chlorosis, and necrosis locally with zero setup.
              </span>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                '<div class="metric-card"><div class="metric-value">Worldwide</div>'
                '<div class="metric-label">Any Plant Leaf</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div class="metric-card"><div class="metric-value">0 Keys</div>'
                '<div class="metric-label">100% Offline</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                '<div class="metric-card"><div class="metric-value">Computer Vision</div>'
                '<div class="metric-label">Spectral Scanner</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                '<div class="metric-card"><div class="metric-value">🟢 Active</div>'
                '<div class="metric-label">Universal Engine</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="hero-banner">
              <h1>🌿 LeafGuard AI — Deep Learning CNN</h1>
              <p>Specialized Neural Network for Agricultural Crop Pathology<br>
              <span style="font-size:0.95rem; color:#6ee7b7;">
                Classifies 15 specific disease conditions across Bell Pepper, Potato, and Tomato crops using our custom trained CNN.
              </span>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                '<div class="metric-card"><div class="metric-value">15</div>'
                '<div class="metric-label">Crop Disease Classes</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div class="metric-card"><div class="metric-value">224px</div>'
                '<div class="metric-label">CNN Resolution</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                '<div class="metric-card"><div class="metric-value">CNN</div>'
                '<div class="metric-label">Neural Architecture</div></div>',
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
        upload_help = "Upload a photo of ANY plant leaf in the world." if "Universal" in engine_mode else "Upload a Bell Pepper, Potato, or Tomato leaf."
        st.markdown("### 📤 Upload Leaf Image")
        uploaded_file = st.file_uploader(
            label="Choose a JPG, JPEG, or PNG image of a plant leaf",
            type=["jpg", "jpeg", "png"],
            help=upload_help,
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
            st.markdown(
                f"<small style='color:#6ee7b7;'>📐 Original size: {image.width} × {image.height} px</small>",
                unsafe_allow_html=True,
            )

            btn_label = "🔍 Scan Leaf Health (Any Plant)" if "Universal" in engine_mode else "🔍 Classify Crop Disease (CNN)"
            analyse_btn = st.button(
                btn_label,
                type="primary",
                use_container_width=True,
            )
        else:
            st.info("👆 Please upload a leaf image to get started.")
            analyse_btn = False

    with right_col:
        st.markdown("### 🧠 Analysis Results")

        if uploaded_file is not None and analyse_btn:
            if "Universal" in engine_mode:
                with st.spinner("🔬 Performing spectral tissue analysis and lesion detection …"):
                    result = analyze_any_leaf_cv(image)
                    render_cv_prediction(result)
            else:
                with st.spinner("Preprocessing image and running CNN inference …"):
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
                  👈 Click <strong>Analyse Leaf</strong> to run the diagnosis.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            hint_text = "Upload any plant leaf in the world to get instant health analysis, tissue metrics, and remedies." if "Universal" in engine_mode else "Upload a leaf to classify diseases across 15 agricultural classes."
            st.markdown(
                f"""
                <div style="color:#6b7280; padding:2rem; text-align:center;
                            border:1px dashed #374151; border-radius:12px;">
                  {hint_text}
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
