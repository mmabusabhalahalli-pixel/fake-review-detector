import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
from datetime import datetime

# ---------- Page Setup ----------
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🕵️",
    layout="centered"
)

# ---------- Global 3D / Attractive Styling ----------
st.markdown("""
<style>
.stApp { background: #EAF1F8; }
.platform-card {
    border-radius: 16px; padding: 22px 10px; text-align: center;
    background: linear-gradient(145deg, #ffffff, #eef2f7);
    box-shadow: 6px 6px 14px rgba(0,0,0,0.15), -4px -4px 10px rgba(255,255,255,0.7);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    border: 1px solid rgba(0,0,0,0.05);
}
.platform-card:hover { transform: translateY(-6px) scale(1.03); box-shadow: 10px 10px 20px rgba(0,0,0,0.2), -6px -6px 14px rgba(255,255,255,0.8); }
.platform-icon { font-size: 34px; }
.platform-name { font-weight: 700; color: #1F4E79; margin-top: 8px; font-size: 15px; }
.info-box {
    background: linear-gradient(145deg, #ffffff, #eef2f7);
    border-radius: 18px; padding: 24px;
    box-shadow: 8px 8px 18px rgba(0,0,0,0.12), -6px -6px 14px rgba(255,255,255,0.7);
    margin-bottom: 20px;
}
.hero-title {
    background: linear-gradient(90deg, #1F4E79, #2E86C1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;
}
div.stButton > button {
    border-radius: 12px;
    box-shadow: 4px 4px 10px rgba(0,0,0,0.18), -3px -3px 8px rgba(255,255,255,0.6);
    transition: transform 0.15s ease; font-weight: 600;
}
div.stButton > button:hover { transform: translateY(-2px); }
.stTextArea textarea, .stTextInput input { border-radius: 10px !important; box-shadow: inset 2px 2px 6px rgba(0,0,0,0.08); }

/* ---------- Figma Login Page ---------- */
.login-title { color:#17365D; font-size:42px; font-weight:800; text-align:center; margin:4px 0 8px; }
.login-subtitle { color:#6B7280; font-size:16px; text-align:center; margin:0 auto 24px; }
.login-lock { text-align:center; font-size:44px; margin-top:8px; margin-bottom:2px; }
.login-demo { text-align:center; color:#5A5A5A; font-size:14px; margin-top:12px; }
body:has(.login-marker) [data-testid="stForm"] {
    background: rgba(255,255,255,0.97) !important;
    border: 1px solid rgba(31,78,121,0.08) !important;
    border-radius: 20px !important;
    padding: 28px 34px 24px !important;
    box-shadow: 0 12px 30px rgba(31,78,121,0.12), 0 2px 8px rgba(0,0,0,0.05) !important;
}
body:has(.login-marker) [data-testid="stForm"] label {
    color:#25364A !important; font-weight:650 !important; font-size:15px !important;
}
body:has(.login-marker) [data-testid="stForm"] input {
    background:#FFFFFF !important; border:1px solid #D8DEE7 !important;
    border-radius:10px !important; min-height:46px !important;
}
body:has(.login-marker) [data-testid="stForm"] input:focus {
    border:1.5px solid #2E86C1 !important; box-shadow:0 0 0 2px rgba(46,134,193,.12) !important;
}
body:has(.login-marker) [data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
    background:#1F4E79 !important; color:white !important; border:0 !important;
    border-radius:10px !important; min-height:48px !important; font-weight:700 !important;
    box-shadow:none !important; width:100% !important;
}
body:has(.login-marker) [data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover {
    background:#173B5C !important; transform:none !important;
}
body:has(.login-marker) [data-testid="stCheckbox"] label { color:#6B7280 !important; font-size:14px !important; }
.login-divider { display:flex; align-items:center; gap:12px; color:#9CA3AF; font-size:13px; margin:18px 0 10px; }
.login-divider:before,.login-divider:after { content:""; height:1px; background:#E2E6EB; flex:1; }
.login-google { width:100%; box-sizing:border-box; border:1px solid #D8DEE7; border-radius:10px; background:#fff; color:#374151; text-align:center; padding:12px; font-size:15px; }
.login-signup { text-align:center; color:#6B7280; font-size:14px; margin-top:16px; }
.login-signup span { color:#1F4E79; font-weight:700; }

/* ---------- Figma Create Account / Security Portal ---------- */
.signup-portal { text-align:center; color:#17365D; font-size:12px; font-weight:800; letter-spacing:2px; margin-top:8px; }
.signup-title { color:#17365D; font-size:38px; font-weight:800; text-align:center; margin:4px 0 8px; }
.signup-subtitle { color:#6B7280; font-size:15px; text-align:center; line-height:1.5; margin:0 auto 24px; max-width:560px; }
.security-note { text-align:center; color:#667085; font-size:12px; margin-top:16px; line-height:1.5; }
.security-footer { text-align:center; color:#7A8494; font-size:11px; line-height:1.7; margin-top:18px; }
.security-footer b { color:#526173; letter-spacing:.5px; }
.strength-good { color:#27AE60; font-weight:700; }
.strength-medium { color:#D68910; font-weight:700; }
.strength-weak { color:#C0392B; font-weight:700; }
body:has(.signup-marker) [data-testid="stForm"] { background:rgba(255,255,255,.97)!important; border:1px solid rgba(31,78,121,.08)!important; border-radius:20px!important; padding:28px 34px 24px!important; box-shadow:0 12px 30px rgba(31,78,121,.12),0 2px 8px rgba(0,0,0,.05)!important; }
body:has(.signup-marker) [data-testid="stForm"] label { color:#25364A!important; font-weight:650!important; font-size:15px!important; }
body:has(.signup-marker) [data-testid="stForm"] input { background:#fff!important; border:1px solid #D8DEE7!important; border-radius:10px!important; min-height:46px!important; }
body:has(.signup-marker) [data-testid="stForm"] input:focus { border:1.5px solid #2E86C1!important; box-shadow:0 0 0 2px rgba(46,134,193,.12)!important; }
body:has(.signup-marker) [data-testid="stForm"] [data-testid="stFormSubmitButton"] button { background:#1F4E79!important; color:white!important; border:0!important; border-radius:10px!important; min-height:48px!important; font-weight:700!important; width:100%!important; }
body:has(.signup-marker) [data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover { background:#173B5C!important; transform:none!important; }
.signup-google { width:100%; box-sizing:border-box; border:1px solid #D8DEE7; border-radius:10px; background:#fff; color:#374151; text-align:center; padding:12px; font-size:15px; margin-bottom:12px; }
.signup-divider { display:flex; align-items:center; gap:12px; color:#9CA3AF; font-size:13px; margin:18px 0 12px; }
.signup-divider:before,.signup-divider:after { content:""; height:1px; background:#E2E6EB; flex:1; }
.signup-login { text-align:center; color:#6B7280; font-size:14px; margin-top:14px; }

/* ---------- Figma Credential Recovery / Reset Password ---------- */
.recovery-marker { text-align:center; }
.recovery-portal { text-align:center; color:#17365D; font-size:12px; font-weight:800; letter-spacing:2px; margin-top:8px; }
.recovery-title { color:#17365D; font-size:38px; font-weight:800; text-align:center; margin:4px 0 8px; }
.recovery-subtitle { color:#6B7280; font-size:15px; text-align:center; line-height:1.5; margin:0 auto 24px; max-width:560px; }
.recovery-note { text-align:center; color:#667085; font-size:12px; margin-top:16px; line-height:1.5; }
.recovery-footer { text-align:center; color:#7A8494; font-size:11px; line-height:1.7; margin-top:18px; }
.recovery-footer b { color:#526173; letter-spacing:.5px; }
.recovery-security { text-align:center; color:#526173; font-size:12px; font-weight:800; letter-spacing:.5px; margin-top:16px; }
.recovery-node { text-align:center; color:#7A8494; font-size:11px; margin-top:10px; }
body:has(.recovery-marker) [data-testid="stForm"] { background:rgba(255,255,255,.97)!important; border:1px solid rgba(31,78,121,.08)!important; border-radius:20px!important; padding:28px 34px 24px!important; box-shadow:0 12px 30px rgba(31,78,121,.12),0 2px 8px rgba(0,0,0,.05)!important; }
body:has(.recovery-marker) [data-testid="stForm"] label { color:#25364A!important; font-weight:650!important; font-size:15px!important; }
body:has(.recovery-marker) [data-testid="stForm"] input { background:#fff!important; border:1px solid #D8DEE7!important; border-radius:10px!important; min-height:46px!important; }
body:has(.recovery-marker) [data-testid="stForm"] input:focus { border:1.5px solid #2E86C1!important; box-shadow:0 0 0 2px rgba(46,134,193,.12)!important; }
body:has(.recovery-marker) [data-testid="stForm"] [data-testid="stFormSubmitButton"] button { background:#1F4E79!important; color:white!important; border:0!important; border-radius:10px!important; min-height:48px!important; font-weight:700!important; width:100%!important; }
body:has(.recovery-marker) [data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover { background:#173B5C!important; transform:none!important; }
</style>
""", unsafe_allow_html=True)

VALID_USERNAME = "admin"
VALID_PASSWORD = "admin123"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "auth_screen" not in st.session_state:
    st.session_state.auth_screen = "login"
if "created_accounts" not in st.session_state:
    st.session_state.created_accounts = {}

# ---------- Google Sheets Logging (via Apps Script Web App) ----------
def get_apps_script_url():
    try:
        return st.secrets["APPS_SCRIPT_URL"]
    except Exception:
        return None

def log_to_sheet(review, prediction, confidence):
    """Send one row to the Google Sheet. Never breaks the app if it fails."""
    url = get_apps_script_url()
    if not url:
        return
    try:
        requests.post(
            url,
            json={"review": review, "prediction": prediction, "confidence": confidence},
            timeout=5,
        )
    except Exception:
        pass  # logging failure should never break the user's experience

@st.cache_data(ttl=30)
def get_dashboard_data():
    """Fetch all logged rows from the Google Sheet. Returns a DataFrame or None."""
    url = get_apps_script_url()
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=8)
        rows = resp.json()
        if not rows or len(rows) < 2:
            return pd.DataFrame(columns=["timestamp", "review", "prediction", "confidence"])
        df = pd.DataFrame(rows[1:], columns=["timestamp", "review", "prediction", "confidence"])
        return df
    except Exception:
        return None


@st.cache_resource
def load_model():
    model = joblib.load("fake_review_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

def clean_text(text):
    text = str(text).lower()
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

def explain_review(review_text, model, vectorizer, top_n=6):
    cleaned = clean_text(review_text)
    vec = vectorizer.transform([cleaned])
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    nonzero_idx = vec.nonzero()[1]
    if len(nonzero_idx) == 0:
        return [], []

    contributions = vec.toarray()[0][nonzero_idx] * coefs[nonzero_idx]
    words = feature_names[nonzero_idx]

    order = np.argsort(contributions)
    fake_words = [(words[i], contributions[i]) for i in order[::-1] if contributions[i] > 0][:top_n]
    real_words = [(words[i], contributions[i]) for i in order if contributions[i] < 0][:top_n]
    return fake_words, real_words

def platform_card(name, icon, url):
    st.markdown(
        f"""<a href="{url}" target="_blank" style="text-decoration:none;">
        <div class="platform-card"><div class="platform-icon">{icon}</div>
        <div class="platform-name">{name}</div></div></a>""",
        unsafe_allow_html=True,
    )


# ============================================================
# LOGIN PAGE
# ============================================================
def login_page():
    st.markdown('<div class="login-marker"></div>', unsafe_allow_html=True)
    st.markdown('<div class="login-lock">🔒</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-title">Login</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Please log in to access the Fake Review Detection System.</div>', unsafe_allow_html=True)

    with st.form("figma_login_form", clear_on_submit=False):
        username = st.text_input("Username", placeholder="Username")
        email = st.text_input("Email", placeholder="Enter your Email")
        password = st.text_input("Password", type="password", placeholder="Password")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.checkbox("Remember me")
        with c2:
            forgot_clicked = st.form_submit_button("Forgot password?")

        clicked = st.form_submit_button("Login")

        if forgot_clicked:
            st.session_state.auth_screen = "recovery"
            st.rerun()

        if clicked:
            ok = username == VALID_USERNAME and password == VALID_PASSWORD
            if username in st.session_state.created_accounts:
                ok = password == st.session_state.created_accounts[username]["password"]
            if ok:
                st.session_state.logged_in = True
                st.session_state.page = "Home"
                st.rerun()
            else:
                st.error("❌ Invalid username or password. Please try again.")

    st.markdown('<div class="login-divider"><span>OR CONTINUE WITH</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="login-google">🌐 &nbsp; Sign in with Google</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-signup">Don\'t have an account?</div>', unsafe_allow_html=True)
    if st.button("Sign up", key="open_signup"):
        st.session_state.auth_screen = "signup"
        st.rerun()
    st.markdown(f'<div class="login-demo">• &nbsp; Demo credentials: &nbsp; <b>{VALID_USERNAME}</b> / <b>{VALID_PASSWORD}</b></div>', unsafe_allow_html=True)


# ============================================================
# CREDENTIAL RECOVERY / RESET PASSWORD PAGE
# ============================================================
def recovery_page():
    st.markdown('<div class="recovery-marker"></div>', unsafe_allow_html=True)
    st.markdown('<div class="recovery-portal">SECURITY PORTAL</div>', unsafe_allow_html=True)
    st.markdown('<div class="recovery-title">CREDENTIAL RECOVERY</div>', unsafe_allow_html=True)
    st.markdown('<div class="recovery-subtitle"><b>Reset Password</b><br>Enter your registered email address and we\'ll send you<br>verification instructions to reset your account password.</div>', unsafe_allow_html=True)

    with st.form("recovery_form", clear_on_submit=False):
        email = st.text_input(
            "Email Address *",
            placeholder="Enter your registered email (e.g. name@company.com)"
        )
        st.markdown(
            '<div class="recovery-note">Recovery links remain active for precisely 15 minutes.</div>',
            unsafe_allow_html=True
        )
        send_clicked = st.form_submit_button("Send Reset Instructions")

        if send_clicked:
            email_clean = email.strip().lower()
            if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email_clean):
                st.error("Please enter a valid registered email address.")
            else:
                st.success("✅ Reset instructions have been requested for this email address.")
                st.info("For the live password-reset email, we will connect this page to the authentication service next.")

    st.markdown(
        '<div class="recovery-note"><b>Need immediate access?</b> Contact system administrator<br>at <b>security@veritas-shield.io</b></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="recovery-security">🔒 256-BIT SSL SAFEGUARDED &nbsp;&nbsp; | &nbsp;&nbsp; SOC2 Type II</div>',
        unsafe_allow_html=True
    )

    if st.button("⬅️ Back to Login", key="recovery_back_login"):
        st.session_state.auth_screen = "login"
        st.rerun()

    st.markdown(
        '<div class="recovery-node">Verified Identity Node #804-F</div>',
        unsafe_allow_html=True
    )
    footer = """<div class="recovery-footer">© 2025 Veritas Shield Threat Intelligence. Cryptographically Secured.<br><b>Privacy Protocol</b> &nbsp;&nbsp; <b>Terms of Verification</b> &nbsp;&nbsp; <b>Audit Telemetry</b> &nbsp;&nbsp; <b>Security Architecture</b></div>"""
    st.markdown(footer, unsafe_allow_html=True)


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
    - **Built by:** Group 1, BCA — Shri Gavisiddeswar Arts, Commerce & Science College, Koppal
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
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🔍 Review Checker", type="primary"):
            st.session_state.page = "Review Checker"; st.rerun()
    with c2:
        if st.button("📂 Batch Checker"):
            st.session_state.page = "Batch Checker"; st.rerun()
    with c3:
        if st.button("📊 Insights"):
            st.session_state.page = "Insights"; st.rerun()
    with c4:
        if st.button("📈 Dashboard"):
            st.session_state.page = "Dashboard"; st.rerun()


# ============================================================
# REVIEW CHECKER PAGE (with explanation + logging)
# ============================================================
def review_checker_page():
    st.markdown('<h1 class="hero-title">🔍 Review Checker</h1>', unsafe_allow_html=True)
    st.write("Paste a product review below and the model will predict whether it looks **Real** or **Fake**.")

    model, vectorizer = load_model()

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    review_input = st.text_area(
        "✍️ Enter a product review:", height=150,
        placeholder="e.g. This product is amazing! Best purchase ever, highly recommend!"
    )

    if st.button("🔍 Check Review", type="primary"):
        if review_input.strip() == "":
            st.warning("⚠️ Please enter a review first.")
        else:
            label, confidence, prediction = predict_review(review_input, model, vectorizer)
            log_to_sheet(review_input, label, round(confidence, 1))

            st.markdown("---")
            if prediction == 1:
                st.error(f"### ❌ {label}")
            else:
                st.success(f"### ✅ {label}")
            st.metric("Model Confidence", f"{confidence:.1f}%")
            st.progress(confidence / 100)

            fake_words, real_words = explain_review(review_input, model, vectorizer)
            st.markdown("#### 🧠 Why did the model decide this?")
            wc1, wc2 = st.columns(2)
            with wc1:
                st.write("**Words pushing toward FAKE:**")
                if fake_words:
                    for w, score in fake_words:
                        st.write(f"🔴 `{w}`  (+{score:.3f})")
                else:
                    st.write("_None found_")
            with wc2:
                st.write("**Words pushing toward REAL:**")
                if real_words:
                    for w, score in real_words:
                        st.write(f"🟢 `{w}`  ({score:.3f})")
                else:
                    st.write("_None found_")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("💡 Try these sample reviews"):
        st.write("**Sample (often Real):**")
        st.code("Bought this two weeks ago. Battery lasts about 5 hours and sound quality is decent for the price.")
        st.write("**Sample (often Fake):**")
        st.code("Excellent product! Best quality ever! Highly recommend! Five stars! Amazing!")

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"; st.rerun()


# ============================================================
# BATCH CHECKER PAGE (with logging)
# ============================================================
def run_batch_predictions(review_list, model, vectorizer):
    results = []
    progress = st.progress(0)
    total = len(review_list)
    for i, text in enumerate(review_list):
        label, confidence, _ = predict_review(text, model, vectorizer)
        log_to_sheet(text, label, round(confidence, 1))
        results.append({"review": text, "prediction": label, "confidence (%)": round(confidence, 1)})
        progress.progress((i + 1) / total)
    return pd.DataFrame(results)


def show_batch_results(result_df):
    total = len(result_df)
    st.success(f"✅ Checked {total} reviews!")
    st.dataframe(result_df)

    fake_count = (result_df["prediction"].str.contains("Fake")).sum()
    real_count = total - fake_count
    c1, c2 = st.columns(2)
    c1.metric("Real Reviews", real_count)
    c2.metric("Fake Reviews", fake_count)

    csv_out = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results as CSV", csv_out, "review_results.csv", "text/csv")


def batch_checker_page():
    st.markdown('<h1 class="hero-title">📂 Batch Review Checker</h1>', unsafe_allow_html=True)
    st.write("Check many reviews at once — either upload a CSV file or paste multiple reviews directly.")

    model, vectorizer = load_model()

    tab1, tab2 = st.tabs(["📋 Paste Multiple Reviews", "📁 Upload CSV File"])

    with tab1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write("Copy reviews from Amazon, Flipkart, Meesho, etc. and paste them below — **one review per line**.")
        pasted_text = st.text_area(
            "Paste reviews here (one per line):",
            height=220,
            placeholder="Bought this two weeks ago, works great!\nExcellent product! Best quality ever! Amazing!\nDecent for the price, delivery was on time."
        )

        if st.button("🔍 Check Pasted Reviews", type="primary"):
            lines = [line.strip() for line in pasted_text.split("\n") if line.strip()]
            if not lines:
                st.warning("⚠️ Please paste at least one review.")
            else:
                result_df = run_batch_predictions(lines, model, vectorizer)
                show_batch_results(result_df)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded file:")
            st.dataframe(df.head())

            col_name = st.selectbox("Select the column that contains the review text:", df.columns)

            if st.button("🔍 Check All Reviews", type="primary"):
                result_df = run_batch_predictions(df[col_name].astype(str).tolist(), model, vectorizer)
                show_batch_results(result_df)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"; st.rerun()


# ============================================================
# INSIGHTS PAGE
# ============================================================
def insights_page():
    st.markdown('<h1 class="hero-title">📊 Model Insights</h1>', unsafe_allow_html=True)
    st.write("These word clouds show which words the model has learned are most associated with **Fake** and **Real** reviews, based on the trained model's internal weights.")

    model, vectorizer = load_model()
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    top_n = 60
    fake_idx = np.argsort(coefs)[::-1][:top_n]
    real_idx = np.argsort(coefs)[:top_n]

    fake_freq = {feature_names[i]: coefs[i] for i in fake_idx}
    real_freq = {feature_names[i]: abs(coefs[i]) for i in real_idx}

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.subheader("🔴 Words Most Associated with FAKE Reviews")
    wc_fake = WordCloud(width=800, height=350, background_color="white", colormap="Reds").generate_from_frequencies(fake_freq)
    fig1, ax1 = plt.subplots(figsize=(8, 3.5))
    ax1.imshow(wc_fake, interpolation="bilinear")
    ax1.axis("off")
    st.pyplot(fig1)

    st.subheader("🟢 Words Most Associated with REAL Reviews")
    wc_real = WordCloud(width=800, height=350, background_color="white", colormap="Greens").generate_from_frequencies(real_freq)
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.imshow(wc_real, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"; st.rerun()


# ============================================================
# DASHBOARD PAGE (reads from Google Sheet)
# ============================================================
def dashboard_page():
    st.markdown('<h1 class="hero-title">📈 Admin Dashboard</h1>', unsafe_allow_html=True)
    st.write("All-time statistics logged from every review checked on this system.")

    if not get_apps_script_url():
        st.warning(
            "⚠️ Dashboard logging is not connected yet. Add your Google Apps Script Web App "
            "URL to Streamlit secrets as `APPS_SCRIPT_URL` to enable this page."
        )
        if st.button("⬅️ Back to Home"):
            st.session_state.page = "Home"; st.rerun()
        return

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

    df = get_dashboard_data()

    if df is None:
        st.error("❌ Could not connect to the Google Sheet. Check the Apps Script URL and try again.")
    elif len(df) == 0:
        st.info("No reviews have been logged yet. Go check a few reviews first!")
    else:
        total = len(df)
        fake_count = df["prediction"].astype(str).str.contains("Fake").sum()
        real_count = total - fake_count

        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Reviews Checked", total)
        c2.metric("Fake Detected", int(fake_count))
        c3.metric("Real Detected", int(real_count))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Fake vs Real Distribution")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            [fake_count, real_count],
            labels=["Fake", "Real"],
            autopct="%1.1f%%",
            colors=["#C0392B", "#27AE60"],
            startangle=90,
        )
        ax.axis("equal")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Recent Activity")
        st.dataframe(df.sort_values("timestamp", ascending=False).head(15))
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"; st.rerun()


# ============================================================
# MAIN APP FLOW
# ============================================================
if not st.session_state.logged_in:
    if st.session_state.auth_screen == "signup":
        create_account_page()
    elif st.session_state.auth_screen == "recovery":
        recovery_page()
    else:
        login_page()
else:
    st.sidebar.title("📂 Navigation")
    pages = ["Home", "Review Checker", "Batch Checker", "Insights", "Dashboard"]
    nav_choice = st.sidebar.radio("Go to:", pages, index=pages.index(st.session_state.page))
    st.session_state.page = nav_choice

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "Home"
        st.rerun()

    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Review Checker":
        review_checker_page()
    elif st.session_state.page == "Batch Checker":
        batch_checker_page()
    elif st.session_state.page == "Insights":
        insights_page()
    elif st.session_state.page == "Dashboard":
        dashboard_page()
