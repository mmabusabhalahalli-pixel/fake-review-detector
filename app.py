import streamlit as st
import joblib
import re

# ---------- Page Setup ----------
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🕵️",
    layout="centered"
)

# ---------- Global 3D / Attractive Styling ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f0f4f8 0%, #d9e6f5 100%);
}

.platform-card {
    border-radius: 16px;
    padding: 22px 10px;
    text-align: center;
    background: linear-gradient(145deg, #ffffff, #eef2f7);
    box-shadow: 6px 6px 14px rgba(0,0,0,0.15), -4px -4px 10px rgba(255,255,255,0.7);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    border: 1px solid rgba(0,0,0,0.05);
}
.platform-card:hover {
    transform: translateY(-6px) scale(1.03);
    box-shadow: 10px 10px 20px rgba(0,0,0,0.2), -6px -6px 14px rgba(255,255,255,0.8);
}
.platform-icon { font-size: 34px; }
.platform-name { font-weight: 700; color: #1F4E79; margin-top: 8px; font-size: 15px; }

.info-box {
    background: linear-gradient(145deg, #ffffff, #eef2f7);
    border-radius: 18px;
    padding: 24px;
    box-shadow: 8px 8px 18px rgba(0,0,0,0.12), -6px -6px 14px rgba(255,255,255,0.7);
    margin-bottom: 20px;
}

.hero-title {
    background: linear-gradient(90deg, #1F4E79, #2E86C1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

div.stButton > button {
    border-radius: 12px;
    box-shadow: 4px 4px 10px rgba(0,0,0,0.18), -3px -3px 8px rgba(255,255,255,0.6);
    transition: transform 0.15s ease;
    font-weight: 600;
}
div.stButton > button:hover {
    transform: translateY(-2px);
}

.stTextArea textarea, .stTextInput input {
    border-radius: 10px !important;
    box-shadow: inset 2px 2px 6px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ---------- Demo Login Credentials ----------
VALID_USERNAME = "admin"
VALID_PASSWORD = "admin123"

# ---------- Session State Setup ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------- Load Model & Vectorizer ----------
@st.cache_resource
def load_model():
    model = joblib.load("fake_review_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def predict_review(review_text, model, vectorizer):
    cleaned = clean_text(review_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    confidence = max(probability) * 100
    label = "Fake (Computer Generated)" if prediction == 1 else "Real (Genuine)"
    return label, confidence, prediction


def platform_card(name, icon, url):
    st.markdown(
        f"""
        <a href="{url}" target="_blank" style="text-decoration:none;">
            <div class="platform-card">
                <div class="platform-icon">{icon}</div>
                <div class="platform-name">{name}</div>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOGIN PAGE
# ============================================================
def login_page():
    st.markdown('<h1 class="hero-title">🔐 Login</h1>', unsafe_allow_html=True)
    st.write("Please log in to access the Fake Review Detection System.")

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login", type="primary"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.logged_in = True
            st.session_state.page = "Home"
            st.rerun()
        else:
            st.error("❌ Invalid username or password. Please try again.")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ Demo credentials"):
        st.write(f"**Username:** `{VALID_USERNAME}`")
        st.write(f"**Password:** `{VALID_PASSWORD}`")


# ============================================================
# HOME PAGE
# ============================================================
def home_page():
    st.markdown('<h1 class="hero-title">🕵️ Fake Review Detection System</h1>', unsafe_allow_html=True)
    st.write("A Machine Learning based system to detect fake product reviews.")

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.subheader("📖 About This Project")
    st.write("""
    Online reviews strongly influence buying decisions on e-commerce platforms.
    However, many reviews are **fake** — written to unfairly boost or damage a
    product's reputation. This project uses **Machine Learning (TF-IDF + Logistic
    Regression)** trained on 40,000+ labeled reviews to automatically classify a
    review as **Real** or **Fake**, with an accuracy of about **89.9%**.

    - **Model:** Logistic Regression
    - **Feature Extraction:** TF-IDF (Term Frequency–Inverse Document Frequency)
    - **Dataset:** 40,432 labeled product reviews (Kaggle)
    - **Built by:** Group 1, BCA — Shri Gavisiddeswar Arts, Commerce & Science
      College, Koppal
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("🛒 Check Reviews From Your Favorite Platforms")
    st.write("Click a platform below to open it, copy any review, and paste it into the Review Checker page.")

    platforms = [
        ("Amazon", "🛍️", "https://www.amazon.in"),
        ("Flipkart", "🟦", "https://www.flipkart.com"),
        ("Meesho", "🟪", "https://www.meesho.com"),
        ("Myntra", "👗", "https://www.myntra.com"),
        ("Google Reviews", "⭐", "https://www.google.com/search?q=google+reviews"),
    ]

    cols = st.columns(len(platforms))
    for col, (name, icon, url) in zip(cols, platforms):
        with col:
            platform_card(name, icon, url)

    st.caption("Note: these links open the official site in a new tab (these platforms don't allow being displayed inside other websites).")

    st.markdown("---")
    if st.button("➡️ Go to Review Checker", type="primary"):
        st.session_state.page = "Review Checker"
        st.rerun()


# ============================================================
# REVIEW CHECKER PAGE
# ============================================================
def review_checker_page():
    st.markdown('<h1 class="hero-title">🔍 Review Checker</h1>', unsafe_allow_html=True)
    st.write("Paste a product review below and the model will predict whether it looks **Real** or **Fake**.")

    model, vectorizer = load_model()

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    review_input = st.text_area(
        "✍️ Enter a product review:",
        height=150,
        placeholder="e.g. This product is amazing! Best purchase ever, highly recommend!"
    )

    if st.button("🔍 Check Review", type="primary"):
        if review_input.strip() == "":
            st.warning("⚠️ Please enter a review first.")
        else:
            label, confidence, prediction = predict_review(review_input, model, vectorizer)
            st.markdown("---")
            if prediction == 1:
                st.error(f"### ❌ {label}")
            else:
                st.success(f"### ✅ {label}")
            st.metric("Model Confidence", f"{confidence:.1f}%")
            st.progress(confidence / 100)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("💡 Try these sample reviews"):
        st.write("**Sample (often Real):**")
        st.code("Bought this two weeks ago. Battery lasts about 5 hours and sound quality is decent for the price.")
        st.write("**Sample (often Fake):**")
        st.code("Excellent product! Best quality ever! Highly recommend! Five stars! Amazing!")

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"
        st.rerun()


# ============================================================
# MAIN APP FLOW
# ============================================================
if not st.session_state.logged_in:
    login_page()
else:
    st.sidebar.title("📂 Navigation")
    nav_choice = st.sidebar.radio("Go to:", ["Home", "Review Checker"],
                                   index=0 if st.session_state.page == "Home" else 1)
    st.session_state.page = nav_choice

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "Home"
        st.rerun()

    if st.session_state.page == "Home":
        home_page()
    else:
        review_checker_page()
