import streamlit as st
import joblib
import re

# ---------- Page Setup ----------
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🕵️",
    layout="centered"
)

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


# ============================================================
# LOGIN PAGE
# ============================================================
def login_page():
    st.title("🔐 Login")
    st.write("Please log in to access the Fake Review Detection System.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login", type="primary"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.logged_in = True
            st.session_state.page = "Home"
            st.rerun()
        else:
            st.error("❌ Invalid username or password. Please try again.")

    with st.expander("ℹ️ Demo credentials"):
        st.write(f"**Username:** `{VALID_USERNAME}`")
        st.write(f"**Password:** `{VALID_PASSWORD}`")


# ============================================================
# HOME PAGE
# ============================================================
def home_page():
    st.title("🕵️ Fake Review Detection System")
    st.write("A Machine Learning based system to detect fake product reviews.")

    st.markdown("---")

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

    st.markdown("---")

    st.subheader("🛒 Check Reviews From Your Favorite Shopping Sites")
    st.write("Click a platform below to open it, copy any review, and paste it into the Review Checker page.")

    platforms = [
        ("Amazon", "🛍️", "https://www.amazon.in"),
        ("Flipkart", "🟦", "https://www.flipkart.com"),
        ("Meesho", "🟪", "https://www.meesho.com"),
        ("Myntra", "👗", "https://www.myntra.com"),
    ]

    cols = st.columns(len(platforms))
    for col, (name, icon, url) in zip(cols, platforms):
        with col:
            st.markdown(
                f"""
                <a href="{url}" target="_blank" style="text-decoration:none;">
                    <div style="
                        border:1px solid #ddd;
                        border-radius:10px;
                        padding:20px 10px;
                        text-align:center;
                        background-color:#f9f9f9;">
                        <div style="font-size:32px;">{icon}</div>
                        <div style="font-weight:600; color:#333; margin-top:8px;">{name}</div>
                    </div>
                </a>
                """,
                unsafe_allow_html=True,
            )

    st.caption("Note: these links open the official site in a new tab (shopping sites don't allow being displayed inside other websites).")

    st.markdown("---")
    if st.button("➡️ Go to Review Checker", type="primary"):
        st.session_state.page = "Review Checker"
        st.rerun()


# ============================================================
# REVIEW CHECKER PAGE
# ============================================================
def review_checker_page():
    st.title("🔍 Review Checker")
    st.write("Paste a product review below and the model will predict whether it looks **Real** or **Fake**.")

    st.markdown("---")

    model, vectorizer = load_model()

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

    st.markdown("---")
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
    # Sidebar navigation
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
