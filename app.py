import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests

# ---------- Page Setup ----------
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🕵️",
    layout="centered"
)

# ---------- Global 3D / Attractive Styling ----------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f0f4f8 0%, #d9e6f5 100%); }
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
.logo-box {
    width: 70px; height: 70px; border-radius: 18px;
    background: linear-gradient(145deg, #ffffff, #eef2f7);
    box-shadow: 4px 4px 12px rgba(0,0,0,0.1), -3px -3px 8px rgba(255,255,255,0.8);
    display: flex; align-items: center; justify-content: center;
    font-size: 32px; margin: 0 auto 14px auto;
}
.hint-box {
    background: #EAF1F8; border-radius: 10px; padding: 12px 14px;
    font-size: 12.5px; color: #5A5A5A; margin-top: 10px;
}

/* ---------- Sidebar styled like the Figma design ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1F4E79, #163a5c);
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-weight: 600;
    padding: 6px 4px;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.2);
}
[data-testid="stSidebar"] div.stButton > button {
    background: rgba(255,255,255,0.08);
    color: #ffb3b3 !important;
    border: 1px solid rgba(255,255,255,0.25);
    box-shadow: none;
}
</style>
""", unsafe_allow_html=True)

VALID_USERNAME = "admin"
VALID_PASSWORD = "admin123"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Home"

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
    user_email = st.session_state.get("user_email", "unknown")
    try:
        requests.post(
            url,
            json={
                "review": review,
                "prediction": prediction,
                "confidence": confidence,
                "user": user_email,
            },
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
            return pd.DataFrame(columns=["timestamp", "review", "prediction", "confidence", "user"])
        header = rows[0]
        cols = ["timestamp", "review", "prediction", "confidence", "user"][:len(header)]
        df = pd.DataFrame(rows[1:], columns=cols)
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
# LOGIN PAGE (Google Sign-In + Demo Admin fallback)
# ============================================================
def google_auth_available():
    """Check whether [auth] is configured in secrets.toml."""
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def call_apps_script(payload):
    """POST a JSON payload to the Apps Script backend and return the parsed response.
    Returns None if not connected yet or on any network error — callers must handle that."""
    url = get_apps_script_url()
    if not url:
        return None
    try:
        resp = requests.post(url, json=payload, timeout=8)
        return resp.json()
    except Exception:
        return None


def login_page():
    # If Streamlit's native Google login already succeeded, pick it up here.
    if google_auth_available() and getattr(st.user, "is_logged_in", False):
        st.session_state.logged_in = True
        st.session_state.user_name = st.user.name
        st.session_state.user_email = st.user.email
        st.session_state.page = "Home"
        st.rerun()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="logo-box">🕵️</div>', unsafe_allow_html=True)
        st.markdown(
            '<h2 class="hero-title" style="text-align:center; font-size:26px;">Fake Review Detector</h2>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="text-align:center; color:#5A5A5A; font-size:13.5px; margin-bottom:20px;">'
            'Sign in to access the Fake Review Detection System.</p>',
            unsafe_allow_html=True,
        )

        tab_login, tab_signup, tab_forgot = st.tabs(["🔑 Login", "📝 Sign Up", "❓ Forgot Password"])

        # ================= LOGIN TAB =================
        with tab_login:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)

            if not get_apps_script_url():
                st.caption("ℹ️ Account database not connected yet — only the demo account works for now.")

            username = st.text_input("👤 Username or Email", placeholder="Enter your username or email", key="login_user")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="login_pass")

            if st.button("LOGIN  →", type="primary", use_container_width=True):
                if username == VALID_USERNAME and password == VALID_PASSWORD:
                    st.session_state.logged_in = True
                    st.session_state.user_name = "Admin (Demo)"
                    st.session_state.user_email = "admin-demo@local"
                    st.session_state.page = "Home"
                    st.rerun()
                else:
                    result = call_apps_script({"type": "login", "username": username, "password": password})
                    if result is None:
                        st.error("❌ Invalid username or password. Please try again.")
                    elif result.get("status") == "ok":
                        st.session_state.logged_in = True
                        st.session_state.user_name = username
                        st.session_state.user_email = username
                        st.session_state.page = "Home"
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password. Please try again.")

            st.markdown(
                f'<div class="hint-box">ℹ️ <b>Demo credentials</b> — '
                f'Username: <b>{VALID_USERNAME}</b> &nbsp;|&nbsp; Password: <b>{VALID_PASSWORD}</b></div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # ================= SIGN UP TAB =================
        with tab_signup:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write("Create a new account to access the system.")

            new_user = st.text_input("👤 Choose a Username", key="signup_user")
            new_email = st.text_input("📧 Email", placeholder="Enter your email", key="signup_email")
            new_pass = st.text_input("🔒 Choose a Password", type="password", key="signup_pass")
            confirm_pass = st.text_input("🔒 Confirm Password", type="password", key="signup_confirm")

            if st.button("CREATE ACCOUNT  →", type="primary", use_container_width=True):
                if not new_user or not new_email or not new_pass or not confirm_pass:
                    st.warning("⚠️ Please fill in all fields.")
                elif new_pass != confirm_pass:
                    st.error("❌ Passwords do not match.")
                elif not get_apps_script_url():
                    st.info("ℹ️ Account database is not connected yet. This will work once it's set up.")
                else:
                    result = call_apps_script({
                        "type": "register",
                        "username": new_user,
                        "email": new_email,
                        "password": new_pass,
                    })
                    if result is None:
                        st.error("❌ Could not reach the account database. Try again later.")
                    elif result.get("status") == "ok":
                        st.success("✅ Account created successfully! Please login.")
                    elif result.get("status") == "username_exists":
                        st.error("❌ Username already exists.")
                    elif result.get("status") == "email_exists":
                        st.error("❌ Email already exists.")
                    else:
                        st.error(result.get("message", "❌ Registration failed."))
            st.markdown('</div>', unsafe_allow_html=True)

        # ================= FORGOT PASSWORD TAB =================
        with tab_forgot:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write("Enter your username and choose a new password.")

            fp_user = st.text_input("👤 Username or Email", key="forgot_user")
            fp_new_pass = st.text_input("🔒 New Password", type="password", key="forgot_new_pass")
            fp_confirm_pass = st.text_input("🔒 Confirm New Password", type="password", key="forgot_confirm_pass")

            if st.button("RESET PASSWORD  →", type="primary", use_container_width=True):
                if not fp_user or not fp_new_pass:
                    st.warning("⚠️ Please fill in your username and a new password.")
                elif fp_new_pass != fp_confirm_pass:
                    st.error("❌ Passwords do not match.")
                elif not get_apps_script_url():
                    st.info("ℹ️ Account database is not connected yet. This will work once it's set up.")
                else:
                    result = call_apps_script({"type": "reset_password", "username": fp_user, "password": fp_new_pass})
                    if result is None:
                        st.error("❌ Could not reach the account database. Try again later.")
                    elif result.get("status") == "ok":
                        st.success("✅ Password updated! You can now log in with your new password.")
                    elif result.get("status") == "not_found":
                        st.error("❌ No account found with that username.")
                    else:
                        st.error("❌ Something went wrong. Please try again.")
            st.markdown('</div>', unsafe_allow_html=True)


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
    login_page()
else:
    st.sidebar.markdown(
        f"👋 **{st.session_state.get('user_name', 'User')}**<br>"
        f"<span style='font-size:12px; opacity:0.8;'>{st.session_state.get('user_email', '')}</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.title("📂 Navigation")
    pages = ["Home", "Review Checker", "Batch Checker", "Insights", "Dashboard"]
    nav_choice = st.sidebar.radio("Go to:", pages, index=pages.index(st.session_state.page))
    st.session_state.page = nav_choice

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "Home"
        if google_auth_available() and getattr(st.user, "is_logged_in", False):
            st.logout()
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
