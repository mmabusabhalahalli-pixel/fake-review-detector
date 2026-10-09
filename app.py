import streamlit as st
import joblib
import re
import html
from collections import Counter
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
.highlight-box {
    background: #ffffff; border-radius: 12px; padding: 16px 18px;
    line-height: 2.0; font-size: 15.5px; color: #222222;
    border: 1px solid rgba(0,0,0,0.08);
}
.legend-chip {
    display: inline-block; padding: 2px 10px; border-radius: 6px;
    font-size: 12.5px; margin-right: 8px; color: #222222;
}
.flag-box {
    background: #FFF4E5; border-left: 5px solid #E67E22; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 8px; color: #5D3A00; font-size: 14px;
}
.score-card {
    border-radius: 20px; padding: 26px 20px; text-align: center; color: #ffffff;
    box-shadow: 8px 8px 18px rgba(0,0,0,0.18), -6px -6px 14px rgba(255,255,255,0.6);
    margin-bottom: 18px;
}
.score-number { font-size: 64px; font-weight: 800; line-height: 1.1; }
.score-label { font-size: 20px; font-weight: 700; margin-top: 4px; }
.score-sub { font-size: 13px; opacity: 0.92; margin-top: 6px; }

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

# ---------- Validation helpers (security) ----------
EMAIL_RE = re.compile(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
MIN_LEN = 8


def validate_email(email):
    return bool(EMAIL_RE.match(email.strip()))


def password_problem(pw):
    """Returns an error message, or None if the password is OK."""
    if len(pw) < MIN_LEN:
        return f"Password must be at least {MIN_LEN} characters."
    if not re.search(r"[A-Za-z]", pw):
        return "Password must contain at least one letter."
    if not re.search(r"[0-9]", pw):
        return "Password must contain at least one number."
    return None


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "role" not in st.session_state:
    st.session_state.role = "user"   # "admin" or "user"


def is_admin():
    return st.session_state.get("role") == "admin"

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


# ============================================================
# PHASE 1 HELPERS: red flags, highlighting, trust score
# ============================================================
STOPWORDS = {
    "the", "and", "this", "that", "with", "have", "for", "was", "are", "but", "not",
    "you", "all", "can", "had", "her", "his", "from", "they", "been", "were", "will",
    "would", "there", "their", "what", "about", "which", "when", "your", "very", "just",
}
PRAISE_WORDS = {
    "best", "amazing", "excellent", "perfect", "awesome", "fantastic", "wonderful",
    "superb", "outstanding", "incredible", "great", "love", "loved",
}


def find_red_flags(text):
    """Simple rule-based checks that work alongside the ML model.
    Returns a list of human-readable warnings (empty list = no red flags)."""
    flags = []
    t = str(text).strip()
    words = re.findall(r"[A-Za-z']+", t)
    n = len(words)

    if n < 5:
        flags.append("Very short review (less than 5 words) — gives no real detail.")

    excl = t.count("!")
    if excl >= 3:
        flags.append(f"Too many exclamation marks ({excl}).")

    caps = [w for w in words if len(w) >= 3 and w.isupper()]
    if len(caps) >= 3:
        flags.append(f"Many ALL-CAPS words ({len(caps)}).")

    counts = Counter(w.lower() for w in words if len(w) > 3 and w.lower() not in STOPWORDS)
    repeated = [w for w, c in counts.items() if c >= 3]
    if repeated:
        flags.append("Same word repeated 3+ times: " + ", ".join(repeated[:4]) + ".")

    praise = [w for w in words if w.lower() in PRAISE_WORDS]
    if len(praise) >= 3 and n <= 25:
        flags.append("Lots of over-the-top praise words but very little detail.")

    return flags


def highlight_review_html(review_text, model, vectorizer):
    """Return HTML where each word the model knows is shaded red (fake-leaning)
    or green (real-leaning). Darker shade = stronger influence."""
    coefs = model.coef_[0]
    vocab = vectorizer.vocabulary_

    tokens = str(review_text).split()
    weights = []
    for tok in tokens:
        key = clean_text(tok).strip()
        idx = vocab.get(key) if key else None
        weights.append(float(coefs[idx]) if idx is not None else 0.0)

    max_abs = max((abs(w) for w in weights), default=0.0)
    parts = []
    for tok, w in zip(tokens, weights):
        safe = html.escape(tok)
        if max_abs == 0 or abs(w) / max_abs < 0.2:
            parts.append(safe)
            continue
        strength = abs(w) / max_abs
        alpha = 0.18 + 0.55 * strength
        if w > 0:
            color = f"rgba(192,57,43,{alpha:.2f})"
            tip = "pushes toward FAKE"
        else:
            color = f"rgba(39,174,96,{alpha:.2f})"
            tip = "pushes toward REAL"
        parts.append(
            f'<span title="{tip}" style="background:{color}; border-radius:5px; padding:2px 4px;">{safe}</span>'
        )
    return " ".join(parts)


def predict_many(review_list, model, vectorizer):
    """Fast batch prediction. Returns (fake_probabilities, is_fake_flags)."""
    cleaned = [clean_text(r) for r in review_list]
    X = vectorizer.transform(cleaned)
    probs = model.predict_proba(X)
    fake_col = list(model.classes_).index(1)
    fake_prob = probs[:, fake_col]
    return fake_prob, fake_prob >= 0.5


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
        st.session_state.role = "user"
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
                    st.session_state.role = "admin"
                    st.session_state.page = "Home"
                    st.rerun()
                else:
                    result = call_apps_script({"type": "login", "username": username, "password": password})
                    if result is None:
                        st.error("❌ Invalid username or password. Please try again.")
                    elif result.get("status") == "ok":
                        # Use the canonical username/email returned by the database so that
                        # history stays consistent even if the user logs in with email one day
                        # and username the next.
                        real_username = result.get("username") or username
                        real_email = result.get("email") or real_username
                        st.session_state.logged_in = True
                        st.session_state.user_name = real_username
                        st.session_state.user_email = real_email
                        st.session_state.role = result.get("role") or "user"
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
            st.caption("Min 8 characters, with at least one letter and one number.")
            confirm_pass = st.text_input("🔒 Confirm Password", type="password", key="signup_confirm")

            if st.button("CREATE ACCOUNT  →", type="primary", use_container_width=True):
                pw_err = password_problem(new_pass) if new_pass else None
                if not new_user or not new_email or not new_pass or not confirm_pass:
                    st.warning("⚠️ Please fill in all fields.")
                elif not validate_email(new_email):
                    st.error("❌ Please enter a valid email (e.g. name@example.com).")
                elif pw_err:
                    st.error(f"❌ {pw_err}")
                elif new_pass != confirm_pass:
                    st.error("❌ Passwords do not match.")
                elif not get_apps_script_url():
                    st.info("ℹ️ Account database is not connected yet. This will work once it's set up.")
                else:
                    result = call_apps_script({
                        "type": "register",
                        "username": new_user.strip(),
                        "email": new_email.strip(),
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
                    elif result.get("status") == "invalid_email":
                        st.error("❌ Invalid email format.")
                    elif result.get("status") == "weak_password":
                        st.error("❌ Password is too weak.")
                    else:
                        st.error(result.get("message", "❌ Registration failed."))
            st.markdown('</div>', unsafe_allow_html=True)

        # ================= FORGOT PASSWORD TAB =================
        with tab_forgot:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write("Enter your username and choose a new password.")

            fp_user = st.text_input("👤 Username or Email", key="forgot_user")
            fp_new_pass = st.text_input("🔒 New Password", type="password", key="forgot_new_pass")
            st.caption("Min 8 characters, with at least one letter and one number.")
            fp_confirm_pass = st.text_input("🔒 Confirm New Password", type="password", key="forgot_confirm_pass")

            if st.button("RESET PASSWORD  →", type="primary", use_container_width=True):
                pw_err = password_problem(fp_new_pass) if fp_new_pass else None
                if not fp_user or not fp_new_pass:
                    st.warning("⚠️ Please fill in your username and a new password.")
                elif pw_err:
                    st.error(f"❌ {pw_err}")
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
                    elif result.get("status") == "weak_password":
                        st.error("❌ Password is too weak.")
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
    # Buttons shown depend on role: only admins get the Dashboard button.
    quick_links = [
        ("🔍 Review Checker", "Review Checker", True),
        ("🛡️ Trust Score", "Product Trust Score", True),
        ("📂 Batch Checker", "Batch Checker", False),
        ("📊 Insights", "Insights", False),
        ("🕘 My History", "My History", False),
    ]
    if is_admin():
        quick_links.append(("📈 Dashboard", "Dashboard", False))

    # Show the buttons in rows of 3 so labels never get squeezed.
    for start in range(0, len(quick_links), 3):
        row = quick_links[start:start + 3]
        cols = st.columns(3)
        for col, (label, target, primary) in zip(cols, row):
            with col:
                if st.button(label, type="primary" if primary else "secondary",
                             key=f"home_btn_{target}", use_container_width=True):
                    st.session_state.page = target
                    st.rerun()


# ============================================================
# REVIEW CHECKER PAGE (explanation + highlighting + red flags + logging)
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

            # ---- Highlighted review ----
            st.markdown("#### 🖍️ Your review, highlighted")
            st.markdown(
                '<span class="legend-chip" style="background:rgba(192,57,43,0.45);">Fake-sounding</span>'
                '<span class="legend-chip" style="background:rgba(39,174,96,0.45);">Real-sounding</span>'
                '<span style="font-size:12px; color:#777;">Darker shade = stronger influence</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="highlight-box">{highlight_review_html(review_input, model, vectorizer)}</div>',
                unsafe_allow_html=True,
            )

            # ---- Red flags (rule-based) ----
            flags = find_red_flags(review_input)
            st.markdown("#### 🚩 Red-flag check")
            if flags:
                for f in flags:
                    st.markdown(f'<div class="flag-box">🚩 {html.escape(f)}</div>', unsafe_allow_html=True)
                st.caption("These are simple rule-based warnings. They work alongside the ML model, they don't replace it.")
            else:
                st.success("✅ No suspicious writing patterns found.")

            # ---- Word-level explanation ----
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
# PRODUCT TRUST SCORE PAGE (new in Phase 1)
# ============================================================
def analyze_product(reviews, product_name, model, vectorizer):
    fake_prob, is_fake = predict_many(reviews, model, vectorizer)
    flags_list = [find_red_flags(r) for r in reviews]

    total = len(reviews)
    fake_count = int(is_fake.sum())
    real_count = total - fake_count
    flagged = sum(1 for f in flags_list if f)

    # Trust Score = 100 minus the average chance that a review is fake.
    trust = int(round((1 - float(fake_prob.mean())) * 100))

    if trust >= 75:
        verdict, icon = "Trustworthy", "✅"
        bg = "linear-gradient(145deg, #27AE60, #1E8449)"
        advice = "Most reviews look genuine. Still read a few before buying."
    elif trust >= 50:
        verdict, icon = "Mixed — Read Carefully", "⚠️"
        bg = "linear-gradient(145deg, #E67E22, #CA6F1E)"
        advice = "A noticeable share of reviews look fake. Don't rely on the star rating alone."
    else:
        verdict, icon = "Suspicious", "❌"
        bg = "linear-gradient(145deg, #C0392B, #922B21)"
        advice = "Many reviews look fake. Be very careful with this product's rating."

    title = html.escape(product_name.strip()) if product_name.strip() else "This product"
    st.markdown(
        f"""<div class="score-card" style="background:{bg};">
        <div style="font-size:15px; opacity:0.9;">{title}</div>
        <div class="score-number">{trust}<span style="font-size:26px;">/100</span></div>
        <div class="score-label">{icon} {verdict}</div>
        <div class="score-sub">{advice}</div></div>""",
        unsafe_allow_html=True,
    )
    st.progress(trust / 100)

    if total < 5:
        st.warning("⚠️ Only a few reviews were checked, so this score may not be reliable. Add 10+ reviews for a better result.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", total)
    c2.metric("Fake", fake_count)
    c3.metric("Real", real_count)
    c4.metric("Red-flagged", flagged)

    st.markdown("#### Fake vs Real")
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    if fake_count == 0 or real_count == 0:
        ax.pie([1], labels=["Fake" if fake_count else "Real"],
               colors=["#C0392B" if fake_count else "#27AE60"], autopct=lambda p: f"{total}")
    else:
        ax.pie([fake_count, real_count], labels=["Fake", "Real"], autopct="%1.1f%%",
               colors=["#C0392B", "#27AE60"], startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    result_df = pd.DataFrame({
        "review": reviews,
        "prediction": ["Fake" if x else "Real" for x in is_fake],
        "fake probability (%)": np.round(fake_prob * 100, 1),
        "red flags": [len(f) for f in flags_list],
    })

    st.markdown("#### 🔎 Most suspicious reviews")
    top = result_df.sort_values("fake probability (%)", ascending=False).head(3)
    for _, row in top.iterrows():
        st.markdown(
            f'<div class="flag-box">🚩 <b>{row["fake probability (%)"]}% fake</b> — {html.escape(str(row["review"])[:200])}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("📋 See every review with its result"):
        st.dataframe(result_df)

    csv_out = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Report (CSV)", csv_out, "trust_score_report.csv", "text/csv")


def trust_score_page():
    st.markdown('<h1 class="hero-title">🛡️ Product Trust Score</h1>', unsafe_allow_html=True)
    st.write(
        "Should you trust this product's reviews? Paste several reviews of **one product** and the system "
        "will give an overall **Trust Score out of 100**."
    )

    model, vectorizer = load_model()

    with st.expander("ℹ️ How is the score calculated?"):
        st.write(
            "The model gives every review a probability of being fake. "
            "**Trust Score = 100 − average fake probability.** "
            "75 or more is trustworthy, 50–74 is mixed, below 50 is suspicious."
        )

    product_name = st.text_input("🏷️ Product name (optional)", placeholder="e.g. boAt Airdopes 141")

    tab1, tab2 = st.tabs(["📋 Paste Reviews", "📁 Upload CSV"])

    with tab1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write("Copy 10–20 reviews of the product from Amazon, Flipkart, etc. — **one review per line**.")
        pasted = st.text_area(
            "Paste reviews here (one per line):", height=220, key="trust_paste",
            placeholder="Battery lasts a full day, mic is average.\nBest product ever! Amazing! Buy now!!!\nStopped working after 3 weeks."
        )
        if st.button("🛡️ Calculate Trust Score", type="primary", key="trust_btn_paste"):
            lines = [l.strip() for l in pasted.split("\n") if l.strip()]
            if not lines:
                st.warning("⚠️ Please paste at least one review.")
            else:
                analyze_product(lines, product_name, model, vectorizer)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        up = st.file_uploader("Upload CSV file", type=["csv"], key="trust_csv")
        if up is not None:
            df = pd.read_csv(up)
            st.dataframe(df.head())
            col_name = st.selectbox("Column with the review text:", df.columns, key="trust_col")
            if st.button("🛡️ Calculate Trust Score", type="primary", key="trust_btn_csv"):
                reviews = [r.strip() for r in df[col_name].astype(str).tolist() if r.strip() and r.strip().lower() != "nan"]
                if not reviews:
                    st.warning("⚠️ No review text found in that column.")
                else:
                    analyze_product(reviews, product_name, model, vectorizer)
        st.markdown('</div>', unsafe_allow_html=True)

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
        results.append({
            "review": text,
            "prediction": label,
            "confidence (%)": round(confidence, 1),
            "red flags": len(find_red_flags(text)),
        })
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
# MY HISTORY PAGE (only the logged-in user's own reviews)
# ============================================================
def my_history_page():
    st.markdown('<h1 class="hero-title">🕘 My History</h1>', unsafe_allow_html=True)
    st.write("All the reviews **you** have checked on this system.")

    if not get_apps_script_url():
        st.warning("⚠️ History is not available because the database is not connected yet.")
        if st.button("⬅️ Back to Home"):
            st.session_state.page = "Home"; st.rerun()
        return

    if st.button("🔄 Refresh"):
        st.cache_data.clear()

    df = get_dashboard_data()
    my_id = st.session_state.get("user_email", "")

    if df is None:
        st.error("❌ Could not load history right now. Please try again.")
    elif "user" not in df.columns:
        st.info("No personal history yet — start checking some reviews!")
    else:
        mine = df[df["user"].astype(str) == str(my_id)]

        if len(mine) == 0:
            st.info("You haven't checked any reviews yet. Go to the Review Checker and try one!")
        else:
            total = len(mine)
            fake_count = int(mine["prediction"].astype(str).str.contains("Fake").sum())
            real_count = total - fake_count

            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Reviews You Checked", total)
            c2.metric("Fake Found", fake_count)
            c3.metric("Real Found", real_count)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.subheader("Your Recent Reviews")
            shown = mine.sort_values("timestamp", ascending=False)
            st.dataframe(shown[["timestamp", "review", "prediction", "confidence"]])
            csv_out = shown.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download My History (CSV)", csv_out, "my_history.csv", "text/csv")
            st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"; st.rerun()


# ============================================================
# DASHBOARD PAGE (admin only, reads from Google Sheet)
# ============================================================
def dashboard_page():
    st.markdown('<h1 class="hero-title">📈 Admin Dashboard</h1>', unsafe_allow_html=True)

    # Role-based access: normal users are blocked even if they reach this page somehow.
    if not is_admin():
        st.error("🚫 Access denied. The Dashboard is available to administrators only.")
        if st.button("⬅️ Back to Home"):
            st.session_state.page = "Home"; st.rerun()
        return

    st.write("All-time statistics logged from every review checked on this system.")

    if not get_apps_script_url():
        st.warning(
            "⚠️ Dashboard logging is not connected yet. Add your Google Apps Script Web App "
            "URL to Streamlit secrets as `APPS_SCRIPT_URL` to enable this page."
        )
        if st.button("⬅️ Back to Home"):
            st.session_state.page = "Home"; st.rerun()
        return

    c_refresh, c_clear = st.columns([1, 1])
    with c_refresh:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()

    with c_clear:
        with st.popover("🗑️ Clear All Data"):
            st.warning("⚠️ This will permanently delete all logged reviews. This cannot be undone.")
            confirm = st.checkbox("Yes, I'm sure — delete everything")
            if st.button("Confirm Delete", type="primary", disabled=not confirm):
                result = call_apps_script({"type": "clear_logs"})
                if result and result.get("status") == "ok":
                    st.cache_data.clear()
                    st.success("✅ All data cleared!")
                    st.rerun()
                else:
                    st.error("❌ Could not clear data. Try again.")

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
    role_label = "Administrator" if is_admin() else "User"
    st.sidebar.markdown(
        f"👋 **{st.session_state.get('user_name', 'User')}**<br>"
        f"<span style='font-size:12px; opacity:0.8;'>{st.session_state.get('user_email', '')}</span><br>"
        f"<span style='font-size:11px; opacity:0.8;'>Role: {role_label}</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.title("📂 Navigation")

    # Pages visible depend on role: only admins see the Dashboard.
    pages = ["Home", "Review Checker", "Product Trust Score", "Batch Checker", "Insights", "My History"]
    if is_admin():
        pages.append("Dashboard")

    # If the stored page is not allowed for this role, send the user home.
    if st.session_state.page not in pages:
        st.session_state.page = "Home"

    nav_choice = st.sidebar.radio("Go to:", pages, index=pages.index(st.session_state.page))
    st.session_state.page = nav_choice

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.role = "user"
        st.session_state.page = "Home"
        if google_auth_available() and getattr(st.user, "is_logged_in", False):
            st.logout()
        st.rerun()

    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Review Checker":
        review_checker_page()
    elif st.session_state.page == "Product Trust Score":
        trust_score_page()
    elif st.session_state.page == "Batch Checker":
        batch_checker_page()
    elif st.session_state.page == "Insights":
        insights_page()
    elif st.session_state.page == "My History":
        my_history_page()
    elif st.session_state.page == "Dashboard":
        dashboard_page()
