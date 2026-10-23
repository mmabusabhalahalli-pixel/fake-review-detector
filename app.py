import streamlit as st
import joblib
import re
import os
import html
from datetime import datetime, timedelta
import io
import csv
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
import requests

# ---------- Page Setup ----------
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🕵️",
    layout="wide"
)

# ---------- Themes (colours of the whole app) ----------
# "brand_dark" and "accent" are used behind/under white or dark text, so they are dark enough to read easily.
THEMES = {
    "Premium Trust": dict(bg1="#F8FAFC", bg2="#E8F5F2", soft="#F0FDFA", ink="#0B1220",
                          brand="#14B8A6", brand_dark="#0F766E", accent="#14B8A6", accent_dark="#0F766E",
                          side1="#0B1220", side2="#123B46"),
    "Royal Blue": dict(bg1="#F0F4F8", bg2="#D9E6F5", soft="#EAF1F8", ink="#1F4E79",
                       brand="#2E86C1", brand_dark="#1F4E79", accent="#C0392B", accent_dark="#922B21",
                       side1="#1F4E79", side2="#163A5C"),
    "Violet Glow": dict(bg1="#F7F3FF", bg2="#E9DDFB", soft="#EFE7FD", ink="#2E1065",
                        brand="#8B5CF6", brand_dark="#5B21B6", accent="#BE185D", accent_dark="#9D174D",
                        side1="#2E1065", side2="#4C1D95"),
}
DEFAULT_THEME = "Premium Trust"

THEME_TEMPLATE = r"""
.stApp { background: linear-gradient(135deg, {{bg1}} 0%, {{bg2}} 100%); }

/* ----- cards ----- */
.platform-card {
    border-radius: 16px; padding: 22px 10px; text-align: center;
    background: linear-gradient(145deg, #ffffff, {{soft}});
    box-shadow: 6px 6px 14px rgba(0,0,0,0.12), -4px -4px 10px rgba(255,255,255,0.7);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    border: 1px solid rgba(0,0,0,0.05);
}
.platform-card:hover { transform: translateY(-6px) scale(1.03); box-shadow: 10px 10px 20px rgba(0,0,0,0.18), -6px -6px 14px rgba(255,255,255,0.8); }
.platform-icon { font-size: 34px; }
.platform-name { font-weight: 700; color: {{ink}}; margin-top: 8px; font-size: 15px; }
.info-box {
    background: linear-gradient(145deg, #ffffff, {{soft}});
    border-radius: 18px; padding: 24px; border-top: 4px solid {{brand}};
    box-shadow: 8px 8px 18px rgba(0,0,0,0.10), -6px -6px 14px rgba(255,255,255,0.7);
    margin-bottom: 20px;
}
.hero-title {
    background: linear-gradient(90deg, {{ink}}, {{brand_dark}});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;
}
.logo-box {
    width: 70px; height: 70px; border-radius: 18px;
    background: linear-gradient(145deg, #ffffff, {{soft}});
    box-shadow: 4px 4px 12px rgba(0,0,0,0.1), -3px -3px 8px rgba(255,255,255,0.8);
    display: flex; align-items: center; justify-content: center;
    font-size: 32px; margin: 0 auto 14px auto;
}
.hint-box {
    background: {{soft}}; border-left: 4px solid {{brand}}; border-radius: 10px; padding: 12px 14px;
    font-size: 12.5px; color: #444444; margin-top: 10px;
}

/* ----- home banner ----- */
.hero-banner {
    background: linear-gradient(120deg, {{side1}}, {{side2}} 62%, {{brand_dark}});
    border-radius: 22px; padding: 28px 26px; margin-bottom: 22px;
    box-shadow: 8px 8px 20px rgba(0,0,0,0.18);
}
.hero-banner-title { color: #ffffff; font-size: 30px; font-weight: 800; line-height: 1.2; margin-bottom: 6px; }
.hero-banner-sub { color: rgba(255,255,255,0.93); font-size: 15px; margin-bottom: 14px; }
.chip {
    display: inline-block; background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.4);
    color: #ffffff; border-radius: 999px; padding: 4px 12px; font-size: 12.5px; margin: 0 6px 6px 0;
}

/* ----- review highlight, flags, score cards ----- */
.highlight-box {
    background: #ffffff; border-radius: 12px; padding: 16px 18px;
    line-height: 2.0; font-size: 15.5px; color: #222222; border: 1px solid rgba(0,0,0,0.08);
}
.legend-chip { display: inline-block; padding: 2px 10px; border-radius: 6px; font-size: 12.5px; margin-right: 8px; color: #222222; }
.flag-box {
    background: #FFF4E5; border-left: 5px solid #E67E22; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 8px; color: #5D3A00; font-size: 14px;
}
.score-card {
    border-radius: 20px; padding: 26px 20px; text-align: center; color: #ffffff;
    box-shadow: 8px 8px 18px rgba(0,0,0,0.18), -6px -6px 14px rgba(255,255,255,0.6); margin-bottom: 18px;
}
.score-number { font-size: 64px; font-weight: 800; line-height: 1.1; }
.score-label { font-size: 20px; font-weight: 700; margin-top: 4px; }
.score-sub { font-size: 13px; opacity: 0.95; margin-top: 6px; }
.deal-badge {
    display: inline-block; color: #ffffff; border-radius: 12px; padding: 6px 16px;
    font-size: 18px; font-weight: 700; margin-bottom: 10px;
}

/* ----- buttons ----- */
div.stButton > button, div.stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
    border-radius: 12px; font-weight: 600; transition: transform 0.15s ease, filter 0.15s ease;
    box-shadow: 4px 4px 10px rgba(0,0,0,0.14), -3px -3px 8px rgba(255,255,255,0.6);
}
div.stButton > button:hover, div.stDownloadButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px); filter: brightness(1.06);
}
div.stButton > button[kind="primary"], div.stButton > button[data-testid="stBaseButton-primary"],
div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"],
div[data-testid="stFormSubmitButton"] > button[data-testid="stBaseButton-primaryFormSubmit"] {
    background: linear-gradient(135deg, {{accent}}, {{accent_dark}}); color: #ffffff; border: none;
}
div.stButton > button[kind="primary"] p, div.stButton > button[data-testid="stBaseButton-primary"] p { color: #ffffff; }
div.stButton > button[kind="secondary"], div.stButton > button[data-testid="stBaseButton-secondary"],
div.stDownloadButton > button {
    background: #ffffff; color: {{brand_dark}}; border: 1.5px solid {{brand}};
}

/* ----- inputs, tabs, progress, metrics ----- */
.stTextArea textarea, .stTextInput input, .stNumberInput input { border-radius: 10px !important; box-shadow: inset 2px 2px 6px rgba(0,0,0,0.07); }
.stTextArea textarea:focus, .stTextInput input:focus, .stNumberInput input:focus {
    border-color: {{brand}} !important; box-shadow: 0 0 0 2px {{brand}}44 !important;
}
button[data-baseweb="tab"] { font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] { color: {{brand_dark}}; }
div[data-baseweb="tab-highlight"] { background-color: {{accent}} !important; }
div[data-testid="stProgress"] div[role="progressbar"] > div, .stProgress > div > div > div > div {
    background: linear-gradient(90deg, {{brand}}, {{brand_dark}}) !important;
}
[data-testid="stMetricValue"] { color: {{ink}}; font-weight: 700; }
a { color: {{brand_dark}}; }

/* ----- sidebar (kept last so it wins) ----- */
[data-testid="stSidebar"] { background: linear-gradient(180deg, {{side1}}, {{side2}}); }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stRadio label { font-weight: 600; padding: 6px 4px; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }
[data-testid="stSidebar"] div.stButton > button {
    background: rgba(255,255,255,0.10) !important; color: #ffd0d0 !important;
    border: 1px solid rgba(255,255,255,0.3) !important; box-shadow: none !important;
}
"""


def theme_css(name):
    """The <style> block for a theme name. Unknown names fall back to the default theme."""
    palette = THEMES.get(name) or THEMES[DEFAULT_THEME]
    css = THEME_TEMPLATE
    for key, value in palette.items():
        css = css.replace("{{" + key + "}}", value)
    return "<style>" + css + "</style>"


if st.session_state.get("theme") not in THEMES:
    st.session_state["theme"] = DEFAULT_THEME
st.markdown(theme_css(st.session_state["theme"]), unsafe_allow_html=True)

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

def log_many_to_sheet(rows):
    """Save many review checks with ONE request per 200 rows (much faster than one by one).
    rows = list of dicts: {"review", "prediction", "confidence"}"""
    url = get_apps_script_url()
    if not url or not rows:
        return
    user_email = st.session_state.get("user_email", "unknown")
    for i in range(0, len(rows), 200):
        try:
            requests.post(url, json={"type": "log_many", "rows": rows[i:i + 200], "user": user_email}, timeout=20)
        except Exception:
            pass


def send_feedback(review, prediction_label, confidence, correct):
    """Save the user's thumbs up / thumbs down. Returns True if it was saved."""
    result = call_apps_script({
        "type": "feedback",
        "review": review,
        "prediction": prediction_label,
        "confidence": confidence,
        "correct": bool(correct),
        "user": st.session_state.get("user_email", "unknown"),
    })
    return bool(result and result.get("status") == "ok")


@st.cache_data(ttl=30)
def get_feedback_data(admin_key):
    """Admin only: all feedback rows as a DataFrame (or None on error)."""
    res = call_apps_script({"type": "list_feedback", "admin_key": admin_key})
    if not res or res.get("status") != "ok":
        return None
    cols = ["timestamp", "review", "model_said", "actual_label", "verdict", "user"]
    rows = res.get("rows", [])
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([r[:6] for r in rows], columns=cols[:len(rows[0][:6])])


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


BASE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "app.py")))


@st.cache_resource
def _load_model_files():
    # Looked up next to app.py, so it works no matter which folder the terminal is in
    model = joblib.load(os.path.join(BASE_DIR, "fake_review_model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
    return model, vectorizer


def load_model():
    """Loads the trained model. If the files are missing, shows what to do instead of a long error."""
    try:
        return _load_model_files()
    except FileNotFoundError:
        st.error("🧠 The model files were not found: `fake_review_model.pkl` and `tfidf_vectorizer.pkl`.")
        st.info(
            "**How to create them (one time):**\n\n"
            "1. Download the Kaggle *Fake Reviews Dataset* (.csv) and put it in the same folder as `app.py`.\n"
            "2. In the terminal run:  `python train_model.py fake_reviews_dataset.csv`\n"
            "3. Run the app again.\n\n"
            "Or just double-click `START_HERE.bat`, it does these steps for you."
        )
        st.stop()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def predict_review(review_text, model, vectorizer):
    """Single-review prediction. Uses predict_many() so every page gives the SAME result."""
    fake_prob, _ = predict_many([review_text], model, vectorizer)
    fp = float(fake_prob[0])
    prediction = 1 if fp >= 0.5 else 0
    confidence = max(fp, 1 - fp) * 100
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


def normalize_reviews(raw_list):
    """Clean a list of reviews the SAME way for paste, CSV and TXT input.
    Returns (clean_reviews, skipped_count)."""
    out, skipped = [], 0
    for r in raw_list:
        s = "" if r is None else str(r)
        s = s.replace("\u00a0", " ")
        s = re.sub(r"\s+", " ", s).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            skipped += 1
            continue
        out.append(s)
    return out, skipped


def read_review_file(up, key):
    """Read an uploaded CSV/TXT the same way every time. Returns a DataFrame."""
    raw = up.getvalue()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    is_txt = up.name.lower().endswith(".txt")
    has_header = st.checkbox(
        "My file has a header row (first row = column names)",
        value=True, key=f"{key}_hdr", disabled=is_txt,
        help="Untick this if the very first row of your file is already a review.",
    )

    clean_csv = False
    if not is_txt:
        try:
            # A proper CSV has the same number of fields on every row.
            # If rows have different counts, reviews contain unquoted commas
            # and splitting on commas would CUT the reviews short.
            rows = [r for r in csv.reader(io.StringIO(text)) if r]
            clean_csv = len({len(r) for r in rows}) == 1
        except Exception:
            clean_csv = False

        if clean_csv:
            try:
                df = pd.read_csv(io.StringIO(text), header=0 if has_header else None,
                                 dtype=str, keep_default_na=False)
                st.caption(f"📄 Rows read from file: **{len(df)}**"
                           + (" (first row used as column names)" if has_header else ""))
                return df
            except Exception:
                clean_csv = False

        st.warning("⚠️ This file is not a clean CSV (reviews contain commas). "
                   "Reading it as one full review per line instead, so nothing gets cut.")

    lines = text.splitlines()
    if has_header and not is_txt and lines:
        lines = lines[1:]          # first line is the header
    df = pd.DataFrame({"review": lines})
    st.caption(f"📄 Lines read from file: **{len(df)}**")
    return df


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


def _remember_api_error(msg):
    """Keeps the last technical error so the screen can show it (helps find the real problem)."""
    try:
        st.session_state["last_api_error"] = None if msg is None else str(msg).replace("`", "'")[:200]
    except Exception:
        pass


def api_error_detail():
    msg = st.session_state.get("last_api_error")
    return f"\n\nTechnical detail: `{msg}`" if msg else ""


def call_apps_script(payload, timeout=25):
    """POST a JSON payload to the Apps Script backend and return the parsed response.
    Returns None if not connected yet or on any network error — callers must handle that.
    Password hashing in Google Apps Script can be slow, so we wait up to 25 seconds."""
    url = get_apps_script_url()
    if not url:
        return None
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        try:
            data = resp.json()
        except ValueError:
            _remember_api_error(f"Backend did not return JSON (HTTP {resp.status_code}): {resp.text[:80]}")
            return None
        _remember_api_error(None)
        return data
    except Exception as e:
        _remember_api_error(f"{type(e).__name__}: {e}")
        return None


def get_admin_key():
    """Secret key that proves to the backend that a request comes from this app's admin."""
    try:
        return st.secrets["ADMIN_KEY"]
    except Exception:
        return None


def call_admin_script(payload):
    """Same as call_apps_script but adds the secret ADMIN_KEY."""
    key = get_admin_key()
    if not key:
        return {"status": "no_key"}
    return call_apps_script({**payload, "admin_key": key})


def otp_error_message(result):
    """Friendly message for a failed 'send code' request."""
    if result is None:
        return "❌ Could not reach the account database. Try again later." + api_error_detail()
    status = result.get("status")
    if status == "wait":
        return f"⏳ Please wait {result.get('seconds', 60)} seconds before asking for a new code."
    if status == "mail_quota":
        return "❌ The daily email limit was reached. Please try again tomorrow."
    if status == "mail_failed":
        return "❌ Could not send the email. Check that this account has a valid email address."
    return "❌ Something went wrong. Please try again."


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
                    with st.spinner("Checking your login... (can take up to 25 seconds)"):
                        result = call_apps_script({"type": "login", "username": username, "password": password})
                    if result is None:
                        if not get_apps_script_url():
                            st.error("❌ Invalid username or password. Please try again.")
                        else:
                            st.error(
                                "⏳ The server did not answer. This is NOT a wrong-password error. "
                                "Wait a few seconds and try once more."
                                + api_error_detail()
                            )
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
                    elif result.get("status") == "locked":
                        mins = result.get("minutes_left", 5)
                        st.error(f"🔒 Too many wrong attempts. This account is locked for {mins} more minute(s).")
                    else:
                        left = result.get("attempts_left")
                        msg = "❌ Invalid username or password."
                        if left is not None:
                            msg += f" {left} attempt(s) left before the account is locked."
                        st.error(msg)

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

        # ================= FORGOT PASSWORD TAB (6-digit code by email) =================
        with tab_forgot:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            stage = st.session_state.get("fp_stage", 1)

            if st.session_state.get("fp_flash"):
                st.success(st.session_state.pop("fp_flash"))

            if not get_apps_script_url():
                st.info("ℹ️ Account database is not connected yet. This will work once it's set up.")

            elif stage == 1:
                st.write("Enter your username or email. We will send a 6-digit code to your registered email address.")
                fp_user = st.text_input("👤 Username or Email", key="forgot_user")

                if st.button("SEND CODE  →", type="primary", use_container_width=True):
                    if not fp_user.strip():
                        st.warning("⚠️ Please enter your username or email.")
                    else:
                        result = call_apps_script({"type": "request_otp", "username": fp_user.strip()})
                        if result and result.get("status") == "otp_sent":
                            st.session_state.fp_stage = 2
                            st.session_state.fp_account = fp_user.strip()
                            st.rerun()
                        else:
                            st.error(otp_error_message(result))

            else:
                account = st.session_state.get("fp_account", "")
                st.info("📧 If this account exists, a 6-digit code was sent to its registered email. "
                        "It is valid for 10 minutes. Check your spam folder too.")
                st.caption(f"Account: {account}")

                otp = st.text_input("🔢 6-digit code", max_chars=6, key="forgot_otp")
                fp_new_pass = st.text_input("🔒 New Password", type="password", key="forgot_new_pass")
                st.caption("Min 8 characters, with at least one letter and one number.")
                fp_confirm_pass = st.text_input("🔒 Confirm New Password", type="password", key="forgot_confirm_pass")

                if st.button("RESET PASSWORD  →", type="primary", use_container_width=True):
                    pw_err = password_problem(fp_new_pass) if fp_new_pass else None
                    if not re.fullmatch(r"\d{6}", otp.strip()):
                        st.warning("⚠️ Enter the 6-digit code from your email.")
                    elif not fp_new_pass:
                        st.warning("⚠️ Please enter a new password.")
                    elif pw_err:
                        st.error(f"❌ {pw_err}")
                    elif fp_new_pass != fp_confirm_pass:
                        st.error("❌ Passwords do not match.")
                    else:
                        with st.spinner("Updating your password... (can take up to 25 seconds)"):
                            result = call_apps_script({
                                "type": "reset_with_otp", "username": account,
                                "otp": otp.strip(), "password": fp_new_pass,
                            })
                        status = result.get("status") if result else None
                        if status == "ok":
                            st.session_state.fp_stage = 1
                            st.session_state.fp_flash = "✅ Password updated! You can now log in with your new password."
                            st.rerun()
                        elif status == "otp_invalid":
                            left = result.get("tries_left")
                            msg = "❌ That code is not correct."
                            if left is not None:
                                msg += f" {left} attempt(s) left for this code."
                            st.error(msg)
                        elif status == "otp_expired":
                            st.error("⌛ This code has expired. Click 'Send a new code' below.")
                        elif status == "otp_locked":
                            st.error("🔒 Too many wrong codes. Wait a minute, then click 'Send a new code'.")
                        elif status == "weak_password":
                            st.error("❌ Password is too weak.")
                        elif result is None:
                            st.error(
                                "⏳ The server did not answer in time. Your password may already have been changed. "
                                "**Open the Login tab and try your NEW password first.** "
                                "If that fails, click 'Send a new code' below and try again."
                                + api_error_detail()
                            )
                        else:
                            st.error("❌ Something went wrong. Please try again.")

                c_resend, c_back = st.columns(2)
                with c_resend:
                    if st.button("🔁 Send a new code", use_container_width=True):
                        result = call_apps_script({"type": "request_otp", "username": account})
                        if result and result.get("status") == "otp_sent":
                            st.success("📧 A new code was sent.")
                        else:
                            st.warning(otp_error_message(result))
                with c_back:
                    if st.button("⬅️ Use another account", use_container_width=True):
                        st.session_state.fp_stage = 1
                        st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# HOME PAGE
# ============================================================
def home_page():
    st.markdown(
        """<div class="hero-banner">
        <div class="hero-banner-title">🕵️ Shop smarter. Spot fake reviews.</div>
        <div class="hero-banner-sub">Check if reviews are real, see a product's Trust Score, and find out
        if today is a good day to buy.</div>
        <span class="chip">🧠 AI review check</span><span class="chip">🛡️ Trust Score</span>
        <span class="chip">💰 Price forecast</span><span class="chip">🚩 Red-flag alerts</span></div>""",
        unsafe_allow_html=True,
    )

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
        ("💰 Price Prediction", "Price Prediction", True),
        ("📂 Batch Checker", "Batch Checker", False),
        ("📊 Insights", "Insights", False),
        ("🕘 My History", "My History", False),
    ]
    if is_admin():
        quick_links.append(("📈 Dashboard", "Dashboard", False))
        quick_links.append(("👥 Manage Users", "Manage Users", False))

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
def render_review_result(res, model, vectorizer):
    """Shows a saved result. Lives in session_state so the feedback buttons don't make it vanish."""
    review_input = res["review"]
    label, confidence, prediction = res["label"], res["confidence"], res["prediction"]

    st.markdown("---")
    if prediction == 1:
        st.error(f"### ❌ {label}")
    else:
        st.success(f"### ✅ {label}")
    st.metric("Model Confidence", f"{confidence:.1f}%")
    st.progress(confidence / 100)

    # ---- Feedback ----
    st.markdown("#### 💬 Was this prediction correct?")
    if res.get("fb") is not None:
        if res["fb"] == "correct":
            st.success("🙏 Thanks! Glad the model got it right. Your feedback was saved.")
        else:
            st.info("🙏 Thanks! The correction was saved and will help improve the model.")
    elif not get_apps_script_url():
        st.caption("Feedback is available once the database is connected.")
    else:
        b1, b2 = st.columns(2)
        with b1:
            if st.button("👍 Yes, correct", key="fb_yes", use_container_width=True):
                if send_feedback(review_input, label, round(confidence, 1), True):
                    res["fb"] = "correct"
                    st.rerun()
                else:
                    st.error("❌ Could not save feedback. Try again.")
        with b2:
            if st.button("👎 No, wrong", key="fb_no", use_container_width=True):
                if send_feedback(review_input, label, round(confidence, 1), False):
                    res["fb"] = "wrong"
                    st.rerun()
                else:
                    st.error("❌ Could not save feedback. Try again.")

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
            st.session_state.pop("rc_result", None)
        else:
            label, confidence, prediction = predict_review(review_input, model, vectorizer)
            log_to_sheet(review_input, label, round(confidence, 1))
            st.session_state.rc_result = {
                "review": review_input, "label": label,
                "confidence": confidence, "prediction": int(prediction), "fb": None,
            }

    res = st.session_state.get("rc_result")
    if res:
        render_review_result(res, model, vectorizer)
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
def analyze_product(reviews, product_name, model, vectorizer, skipped=0):
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
    st.caption(f"🔢 Reviews analysed: **{len(reviews)}**"
               + (f" · empty lines skipped: {skipped}" if skipped else ""))

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

    # Save these checks to the history/dashboard too (one fast request)
    log_many_to_sheet([
        {"review": r,
         "prediction": "Fake (Computer Generated)" if f else "Real (Genuine)",
         "confidence": round(float(max(pr, 1 - pr)) * 100, 1)}
        for r, f, pr in zip(reviews, is_fake, fake_prob)
    ])

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
            lines, skipped = normalize_reviews(pasted.split("\n"))
            if not lines:
                st.warning("⚠️ Please paste at least one review.")
            else:
                analyze_product(lines, product_name, model, vectorizer, skipped)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        up = st.file_uploader("Upload CSV or TXT file", type=["csv", "txt"], key="trust_csv")
        if up is not None:
            df = read_review_file(up, "trust")
            st.dataframe(df.head())
            col_name = st.selectbox("Column with the review text:", df.columns,
                                    format_func=str, key="trust_col")
            if st.button("🛡️ Calculate Trust Score", type="primary", key="trust_btn_csv"):
                reviews, skipped = normalize_reviews(df[col_name].tolist())
                if not reviews:
                    st.warning("⚠️ No review text found in that column.")
                else:
                    analyze_product(reviews, product_name, model, vectorizer, skipped)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"; st.rerun()


# ============================================================
# PRICE PREDICTION PAGE (new)
# ============================================================
# The page works on a price HISTORY given by the user (pasted or CSV).
# It cannot read live prices from shopping sites.

DATE_LINE_RE = re.compile(r"^\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})(.*)$")

# Typical sale periods in India (a general pattern, the exact dates change every year)
SALE_SEASONS = [
    ("End-of-season sale (fashion & lifestyle)", (1, 1), (1, 31)),
    ("Republic Day sales", (1, 15), (1, 26)),
    ("Summer sale", (4, 15), (5, 31)),
    ("End-of-season sale (fashion & lifestyle)", (6, 20), (7, 31)),
    ("Mid-year mega sale (e.g. Prime Day)", (7, 10), (7, 25)),
    ("Independence Day sales", (8, 1), (8, 15)),
    ("Festive & Diwali season (Big Billion Days, Great Indian Festival)", (9, 20), (11, 15)),
    ("Black Friday / Cyber Monday", (11, 20), (12, 2)),
    ("Year-end sale", (12, 15), (12, 31)),
]


def parse_price(text):
    """'₹1,499.00' -> 1499.0, 'Rs. 999' -> 999.0, nonsense -> None."""
    if re.match(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b", str(text)):
        return None                       # looks like a date, not a price
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(text))
    if not m:
        return None
    try:
        value = float(m.group(0).replace(",", ""))
    except ValueError:
        return None
    return value if value > 0 else None


def parse_date_text(text):
    """Reads 2026-09-15, 15-09-2026, 15/09/2026, 15.09.26 (day first, like in India)."""
    t = re.sub(r"[/.]", "-", str(text).strip())
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%m-%y"):
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            continue
    return None


def today_midnight():
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


def dates_ending_today(n):
    t = today_midnight()
    return [t - timedelta(days=n - 1 - i) for i in range(n)]


def parse_price_history(text, price_only=False):
    """Pasted text -> (DataFrame[date, price], list of lines that could not be read)."""
    rows, bad = [], []
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        if price_only:
            price = parse_price(line)
            if price is None:
                bad.append(line)
            else:
                rows.append((None, price))
            continue
        m = DATE_LINE_RE.match(line)
        date = parse_date_text(m.group(1)) if m else None
        price = parse_price(m.group(2).lstrip(" ,;|:\t")) if m else None
        if date is None or price is None:
            bad.append(line)
        else:
            rows.append((date, price))
    if price_only and rows:
        dates = dates_ending_today(len(rows))
        rows = [(d, p) for d, (_, p) in zip(dates, rows)]
    return pd.DataFrame(rows, columns=["date", "price"]), bad


def history_from_table(df, date_col, price_col):
    """Uploaded CSV -> (DataFrame[date, price], count of unreadable rows)."""
    prices = [parse_price(v) for v in df[price_col].tolist()]
    if date_col is None:
        keep = [p for p in prices if p is not None]
        out = pd.DataFrame({"date": dates_ending_today(len(keep)), "price": keep})
        return out, len(prices) - len(keep)
    rows, bad = [], 0
    for raw_date, price in zip(df[date_col].tolist(), prices):
        d = parse_date_text(str(raw_date).split(" ")[0].split("T")[0])
        if d is None:
            parsed = pd.to_datetime(raw_date, dayfirst=True, errors="coerce")
            d = None
            if not pd.isna(parsed) and 2000 <= parsed.year <= datetime.now().year + 1:
                d = parsed.to_pydatetime().replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
        if d is None or price is None:
            bad += 1
        else:
            rows.append((d, price))
    return pd.DataFrame(rows, columns=["date", "price"]), bad


def sale_season_info(today):
    """Which typical sale season is on now, and which one comes next."""
    active, upcoming = [], []
    for name, (sm, sd), (em, ed) in SALE_SEASONS:
        for year in (today.year, today.year + 1):
            start, end = datetime(year, sm, sd), datetime(year, em, ed)
            if start <= today <= end:
                if name not in [a for a, _ in active]:
                    active.append((name, end))
            elif start > today:
                upcoming.append((start, name))
    upcoming.sort()
    return active, upcoming


def analyze_prices(df):
    """All the numbers the page shows. df has columns date, price (any order)."""
    d = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    prices = d["price"].astype(float).values
    n = len(d)
    cur = float(prices[-1])
    mn, mx, avg = float(prices.min()), float(prices.max()), float(prices.mean())
    last_date = d["date"].iloc[-1]
    span_days = (last_date - d["date"].iloc[0]).days
    pct = float((prices <= cur).mean() * 100)

    res = {
        "d": d, "n": n, "current": cur, "min": mn, "max": mx, "avg": avg,
        "min_date": d.loc[d["price"].idxmin(), "date"], "max_date": d.loc[d["price"].idxmax(), "date"],
        "last_date": last_date, "span_days": span_days, "percentile": pct,
        "range_pos": (cur - mn) / (mx - mn) * 100 if mx > mn else 50.0,
        "above_min": (cur - mn) / mn * 100 if mn else 0.0,
        "vs_avg": (cur - avg) / avg * 100 if avg else 0.0,
        "can_forecast": n >= 5 and span_days >= 7,
        "proj": {}, "trend30": None, "sigma": None, "slope_per_day": None,
        "stable": mn > 0 and (mx - mn) / mn < 0.03,
    }

    ago = d[d["date"] <= last_date - timedelta(days=30)]
    res["price_30_ago"] = (float(ago["price"].iloc[-1]), ago["date"].iloc[-1]) if len(ago) else None

    if n >= 2:
        change = np.diff(prices) / prices[:-1] * 100
        drops = change[change <= -8]
        res["n_drops"] = int(len(drops))
        res["avg_drop"] = float(-drops.mean()) if len(drops) else 0.0
    else:
        res["n_drops"], res["avg_drop"] = 0, 0.0

    if res["can_forecast"]:
        # Trend from the most recent prices: last 14 days, or 30 / 60 days / everything if there are
        # fewer than 5 points. Uses the MEDIAN of all pairwise slopes (Theil-Sen), so one big sale
        # drop in the past does not make a price that is now flat look like it is "falling".
        w = d
        for days in (14, 30, 60):
            w = d[d["date"] >= last_date - timedelta(days=days)]
            if len(w) >= 5:
                break
        else:
            w = d
        w = w.tail(60)
        x = (w["date"] - w["date"].iloc[0]).dt.total_seconds().values / 86400.0
        y = w["price"].values.astype(float)
        pair_slopes = [(y[j] - y[i]) / (x[j] - x[i])
                       for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]]
        slope = float(np.median(pair_slopes)) if pair_slopes else 0.0
        intercept = float(np.median(y - slope * x))
        resid = y - (slope * x + intercept)
        sigma = float(resid.std(ddof=1)) if len(w) > 1 else 0.0
        res["slope_per_day"] = float(slope)
        res["sigma"] = sigma
        for h in (7, 15, 30):
            est = cur + slope * h                       # start from today's real price
            est = float(min(max(est, mn * 0.9), mx * 1.1))
            band = max(1.28 * sigma * (1 + h / 60.0), 0.02 * cur)   # ~80% range, wider further away
            res["proj"][h] = (est, max(est - band, 0.0), est + band)
        res["trend30"] = (res["proj"][30][0] - cur) / cur * 100

    res["verdict"] = price_verdict(res)
    return res


def price_verdict(res):
    cur, mn, mx, t30 = res["current"], res["min"], res["max"], res["trend30"]
    pos = res["range_pos"]                                  # 0 = at the lowest, 100 = at the highest
    near_low = cur <= mn * 1.03
    stable = mn > 0 and (mx - mn) / mn < 0.03        # moved less than 3% in the whole period
    GREEN = "linear-gradient(145deg, #27AE60, #1E8449)"
    LIGHT = "linear-gradient(145deg, #52BE80, #229954)"
    ORANGE = "linear-gradient(145deg, #E67E22, #CA6F1E)"
    RED = "linear-gradient(145deg, #C0392B, #922B21)"
    note = "" if t30 is not None else " (based only on past prices, there is not enough data for a forecast yet)"
    rng = f"₹{mn:,.0f} – ₹{mx:,.0f}"

    if stable:
        return {"label": "Fair price", "icon": "⚖️", "bg": ORANGE, "code": "fair",
                "reason": f"The price has hardly changed in this period, so there is no clear best moment.{note}"}
    if near_low and (t30 is None or t30 > -5):
        return {"label": "Buy now", "icon": "✅", "bg": GREEN, "code": "buy",
                "reason": f"Today's price is within 3% of the lowest price seen (₹{mn:,.0f}).{note}"}
    if t30 is not None and t30 <= -5:
        return {"label": "Wait", "icon": "⏳", "bg": RED, "code": "wait",
                "reason": f"The price has been falling and may drop about {abs(t30):.0f}% in the next 30 days."}
    if pos >= 70:
        return {"label": "Wait", "icon": "⏳", "bg": RED, "code": "wait",
                "reason": f"Today's price is near the high end of its range ({rng}). It was cheaper before.{note}"}
    if pos <= 30:
        return {"label": "Good time to buy", "icon": "👍", "bg": LIGHT, "code": "good",
                "reason": f"Today's price is in the lower part of its range ({rng}) and is not expected to fall much more.{note}"}
    return {"label": "Fair price", "icon": "⚖️", "bg": ORANGE, "code": "fair",
            "reason": f"The price is around the middle of its range ({rng}). No strong reason to rush or to wait.{note}"}


def forecast_confidence(res):
    """How much to trust the forecast: High / Medium / Low, with the reason. None if there is no forecast."""
    if not res["can_forecast"]:
        return None
    n, span = res["n"], res["span_days"]
    pts = 2 if n >= 20 else 1 if n >= 10 else 0
    days = 2 if span >= 30 else 1 if span >= 14 else 0
    noise = (res["sigma"] or 0.0) / res["current"] if res["current"] else 1.0
    calm = 2 if noise <= 0.03 else 1 if noise <= 0.08 else 0
    total = pts + days + calm
    level = "High" if total >= 5 else "Medium" if total >= 3 else "Low"
    moves = {2: "small", 1: "moderate", 0: "large"}[calm]
    return {"level": level, "points": total,
            "why": f"based on {n} prices over {span} days; the price moves are {moves}."}


def deal_score(res, trust_score=None):
    """One 0-100 number: how good is today's price? Parts: price level, price direction, review trust."""
    price_part = 50.0 if res["stable"] else 100.0 - res["range_pos"]          # lower in its range = better
    t30 = res["trend30"]
    dir_part = 50.0 if t30 is None else float(min(max(50 + 5 * t30, 0), 100))   # falling = wait, rising = buy sooner
    if trust_score is None:
        weights = (0.55, 0.45, 0.0)
    else:
        weights = (0.45, 0.30, 0.25)
    trust_part = float(trust_score) if trust_score is not None else 0.0
    score = price_part * weights[0] + dir_part * weights[1] + trust_part * weights[2]
    score = int(round(min(max(score, 0), 100)))
    if score >= 75:
        label, color = "Great deal", "#1E8449"
    elif score >= 60:
        label, color = "Good deal", "#2E7D32"
    elif score >= 45:
        label, color = "Okay", "#B45309"
    else:
        label, color = "Poor timing", "#B3261E"
    return {"score": score, "label": label, "color": color,
            "parts": [("Price level (lower in its range is better)", price_part, weights[0]),
                      ("Price direction (next 30 days)", dir_part, weights[1]),
                      ("Review trust", trust_part, weights[2])]}


def target_analysis(res, target):
    """How realistic is the price the user wants to pay?"""
    cur, mn, n = res["current"], res["min"], res["n"]
    d = res["d"]
    if target >= cur:
        return {"code": "reached", "headline": "🎯 Today's price is already at or below your target.",
                "details": [f"You can buy now at ₹{cur:,.0f}, which is ₹{target - cur:,.0f} under your target."]}

    gap = cur - target
    details = [f"Your target is ₹{gap:,.0f} ({gap / cur * 100:.1f}%) below today's price."]
    hits = d[d["price"] <= target]
    code = "possible"
    if len(hits):
        details.append(f"The price was at or below ₹{target:,.0f} on {len(hits)} of {n} recorded days "
                       f"(last on {hits['date'].iloc[-1].strftime('%d %b %Y')}), so it has been this low before.")
    else:
        code = "unlikely"
        details.append(f"It is below the lowest price seen so far (₹{mn:,.0f}), so it would need a bigger discount than before.")

    if res["can_forecast"]:
        slope = res["slope_per_day"]
        if slope < 0 and target >= mn * 0.9:
            days = gap / -slope
            if days <= 60:
                eta = res["last_date"] + timedelta(days=days)
                details.append(f"If the recent fall continues, it could reach your target in about {days:.0f} days "
                               f"(around {eta.strftime('%d %b')}). This is a rough guess.")
                if days <= 30 and code != "unlikely":
                    code = "likely"
            else:
                details.append("The price is falling slowly, so it may take more than 2 months to reach your target.")
        elif slope >= 0:
            details.append("The price is not falling right now, so it may not reach your target soon without a sale.")

    _, upcoming = sale_season_info(today_midnight())
    if upcoming and code != "unlikely":
        start, name = upcoming[0]
        days_to = (start - today_midnight()).days
        if days_to <= 45:
            details.append(f"A usual sale ({name}) starts in about {days_to} day(s). Prices often drop during sales.")

    headline = {"likely": "🎯 Your target looks reachable soon.",
                "possible": "🎯 Your target is possible, but not guaranteed.",
                "unlikely": "🎯 Your target looks unlikely right now."}[code]
    return {"code": code, "headline": headline, "details": details}


def rupee_delta(diff):
    return f"{'-' if diff < 0 else '+'}₹{abs(diff):,.0f}"


def quick_trust(reviews, model, vectorizer):
    """Same Trust Score formula as the Product Trust Score page."""
    fake_prob, is_fake = predict_many(reviews, model, vectorizer)
    return int(round((1 - float(fake_prob.mean())) * 100)), int(is_fake.sum()), len(reviews)


def draw_price_chart(res, target=None):
    d = res["d"]
    dates = list(d["date"].dt.to_pydatetime())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(dates, d["price"], color="#2E86C1", linewidth=2,
            marker="o" if len(d) <= 40 else None, label="Past price")
    ax.axhline(res["min"], color="#27AE60", linestyle="--", linewidth=1, label="Lowest")
    ax.axhline(res["avg"], color="#7F8C8D", linestyle=":", linewidth=1, label="Average")
    if target:
        ax.axhline(target, color="#C2410C", linestyle="-.", linewidth=1.2, label="Your target")
    if res["can_forecast"]:
        last = dates[-1]
        fx = [last] + [last + timedelta(days=h) for h in (7, 15, 30)]
        fy = [res["current"]] + [res["proj"][h][0] for h in (7, 15, 30)]
        lo = [res["current"]] + [res["proj"][h][1] for h in (7, 15, 30)]
        hi = [res["current"]] + [res["proj"][h][2] for h in (7, 15, 30)]
        ax.plot(fx, fy, color="#E67E22", linestyle="--", marker="o", label="Estimate")
        ax.fill_between(fx, lo, hi, color="#E67E22", alpha=0.18, label="Likely range")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    ax.set_ylabel("Price (Rs)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def render_price_result(res, product_name, trust, target=None):
    v = res["verdict"]
    title = html.escape(product_name.strip()) if product_name.strip() else "This product"
    st.markdown(
        f"""<div class="score-card" style="background:{v['bg']};">
        <div style="font-size:15px; opacity:0.9;">{title}</div>
        <div class="score-number">₹{res['current']:,.0f}</div>
        <div class="score-label">{v['icon']} {v['label']}</div>
        <div class="score-sub">{html.escape(v['reason'])}</div></div>""",
        unsafe_allow_html=True,
    )

    # ---- trust of the reviews (optional)
    trust_score = None
    if trust is not None:
        trust_score, fake_n, total = trust
        if trust_score < 50:
            st.markdown(
                f'<div class="flag-box">⚠️ <b>Review Trust Score: {trust_score}/100</b> — {fake_n} of {total} reviews look fake. '
                f'Be careful with this product even if the price looks good.</div>', unsafe_allow_html=True)
        elif trust_score < 75:
            st.markdown(
                f'<div class="flag-box">⚠️ <b>Review Trust Score: {trust_score}/100</b> — some reviews look fake '
                f'({fake_n} of {total}). Read the reviews carefully.</div>', unsafe_allow_html=True)
        else:
            st.success(f"✅ Review Trust Score: {trust_score}/100 — the reviews look mostly genuine ({fake_n} of {total} flagged).")
        if trust_score >= 75 and v["code"] in ("buy", "good"):
            st.success("👍 Price and reviews both look good.")

    # ---- deal score
    ds = deal_score(res, trust_score)
    st.markdown("#### 🏆 Deal Score")
    st.markdown(f'<div class="deal-badge" style="background:{ds["color"]};">{ds["score"]}/100 · {ds["label"]}</div>',
                unsafe_allow_html=True)
    for part_name, value, weight in ds["parts"]:
        if weight > 0:
            st.progress(min(max(value / 100.0, 0.0), 1.0))
            st.caption(f"{part_name}: {value:.0f}/100 (counts for {weight * 100:.0f}% of the score)")
    st.caption("A higher score means today looks like a better time to buy.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Today", f"₹{res['current']:,.0f}")
    c2.metric("Lowest", f"₹{res['min']:,.0f}", delta=f"{res['above_min']:.1f}% above" if res["above_min"] > 0.05 else "at lowest",
              delta_color="off")
    c3.metric("Highest", f"₹{res['max']:,.0f}")
    c4.metric("Average", f"₹{res['avg']:,.0f}")

    # ---- what it was
    st.markdown("#### 🕘 What the price was")
    d = res["d"]
    first = d.iloc[0]
    rows = [("First recorded", f"₹{first['price']:,.0f}", first["date"].strftime("%d %b %Y"))]
    if res["price_30_ago"]:
        p30, d30 = res["price_30_ago"]
        rows.append(("About 30 days ago", f"₹{p30:,.0f}", d30.strftime("%d %b %Y")))
    rows += [
        ("Lowest", f"₹{res['min']:,.0f}", res["min_date"].strftime("%d %b %Y")),
        ("Highest", f"₹{res['max']:,.0f}", res["max_date"].strftime("%d %b %Y")),
        ("Average", f"₹{res['avg']:,.0f}", f"{res['n']} prices"),
        ("Today", f"₹{res['current']:,.0f}", res["last_date"].strftime("%d %b %Y")),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["", "Price", "When"]))
    if res["n_drops"]:
        st.caption(f"📉 The price dropped by 8% or more {res['n_drops']} time(s) in this period "
                   f"(about {res['avg_drop']:.0f}% each time), which looks like sale offers.")
    else:
        st.caption("The price did not have any big drops (8% or more) in this period.")

    # ---- what it may be
    st.markdown("#### 🔮 What it may be")
    if res["can_forecast"]:
        table = []
        for h in (7, 15, 30):
            est, lo, hi = res["proj"][h]
            table.append({
                "In": f"{h} days",
                "Estimated price": f"₹{est:,.0f}",
                "Likely range": f"₹{lo:,.0f} – ₹{hi:,.0f}",
                "Change vs today": f"{(est - res['current']) / res['current'] * 100:+.1f}%",
            })
        st.dataframe(pd.DataFrame(table))
        conf = forecast_confidence(res)
        st.markdown(f"**Forecast confidence: {conf['level']}** — {conf['why']}")
        st.caption("The likely range is where the price should be about 8 times out of 10 if the recent pattern continues. "
                   "It gets wider further into the future.")

        # ---- buy today or wait
        est30, lo30, hi30 = res["proj"][30]
        diff = est30 - res["current"]
        st.markdown("#### 💸 Buy today or wait?")
        b1, b2, b3 = st.columns(3)
        b1.metric("Buy today", f"₹{res['current']:,.0f}")
        b2.metric("Wait 30 days (estimate)", f"₹{est30:,.0f}", delta=rupee_delta(diff), delta_color="inverse")
        b3.metric("Possible range", f"₹{lo30:,.0f} – ₹{hi30:,.0f}")
        if diff < -0.01 * res["current"]:
            st.write(f"If the recent trend continues, waiting 30 days could save you about **₹{-diff:,.0f}**. "
                     f"But the price can also end up as high as ₹{hi30:,.0f}, so decide how much risk you are okay with.")
        elif diff > 0.01 * res["current"]:
            st.write(f"The price looks like it is rising. Waiting could cost you about **₹{diff:,.0f}** more.")
        else:
            st.write("The price is expected to stay about the same, so waiting is unlikely to change much.")
    else:
        st.info("To see a forecast, add at least 5 prices that cover 7 days or more.")

    # ---- target price
    if target and target > 0:
        st.markdown("#### 🎯 Your target price")
        ta = target_analysis(res, target)
        box = {"reached": st.success, "likely": st.success, "possible": st.info, "unlikely": st.warning}[ta["code"]]
        box(ta["headline"])
        for line in ta["details"]:
            st.write("• " + line)

    st.pyplot(draw_price_chart(res, target if target and target > 0 else None))

    # ---- sale seasons
    active, upcoming = sale_season_info(today_midnight())
    st.markdown("#### 🎉 Sale seasons")
    if active:
        names = " and ".join(name for name, _ in active)
        st.write(f"**Right now:** {names}. Discounts are common in this period, so check that today's price "
                 f"is close to the lowest price seen (₹{res['min']:,.0f}).")
    if upcoming:
        start, name = upcoming[0]
        days = (start - today_midnight()).days
        st.write(f"**Next usual sale:** {name}, in about {days} day(s). "
                 + ("Waiting for it may get you a lower price." if res["verdict"]["code"] in ("wait", "fair") else ""))
    st.caption("These are usual sale periods in India, not guaranteed. Exact dates change every year.")

    st.caption("⚠️ This is an estimate from past prices only. Real prices also depend on sales, stock and the seller. "
               "It is a guide, not a promise.")


def run_price_analysis(df, product_name, current_price, reviews_text, target=0.0):
    df = df.dropna()
    if len(df) < 3:
        st.warning("⚠️ Please give at least 3 prices (with dates) so that something useful can be shown.")
        return
    today = today_midnight()
    if (df["date"] > today + timedelta(days=1)).any():
        st.warning("⚠️ Some dates are in the future. Please check the date format "
                   "(use YYYY-MM-DD, like 2026-09-15, or day-month-year like 15-09-2026).")

    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if current_price and current_price > 0:
        last = df["date"].iloc[-1]
        if last.date() == today.date():
            df.loc[df.index[-1], "price"] = float(current_price)
        elif last < today:
            df.loc[len(df)] = [today, float(current_price)]

    res = analyze_prices(df)

    trust = None
    reviews, _ = normalize_reviews(str(reviews_text).split("\n"))
    if reviews:
        model, vectorizer = load_model()
        trust = quick_trust(reviews, model, vectorizer)

    render_price_result(res, product_name, trust, target)


def price_prediction_page():
    st.markdown('<h1 class="hero-title">💰 Price Prediction</h1>', unsafe_allow_html=True)
    st.write("Check what a product's price **was**, what it **may be** in the next 7, 15 and 30 days, "
             "and whether it is a good time to buy.")

    with st.expander("ℹ️ How does this work? Where do I get the price history?"):
        st.write(
            "Shopping sites do not allow other apps to read their live prices, so you give the app the product's "
            "**price history**. Many free price-tracker websites (for example PriceHistory, PriceBefore, or Keepa for Amazon) "
            "show a product's past prices. Copy a few dates and prices from there.\n\n"
            "The app then finds the lowest, highest and average price, looks at the recent trend, and gives an "
            "**estimate** for the coming weeks with a likely range. It never promises a price."
        )
        st.code("2026-09-01, 2499\n2026-09-10, 2499\n2026-09-20, 2299\n2026-09-28, 1999\n"
                "2026-10-05, 1899\n2026-10-12, 1949\n2026-10-18, 1799\n2026-10-22, 1749", language="text")
        st.caption("One price per line: date, then price. Dates can be 2026-09-15 or 15-09-2026 (day first).")

    product_name = st.text_input("🏷️ Product name (optional)", placeholder="e.g. boAt Airdopes 141", key="pp_name")
    current_in = st.number_input(
        "💵 Today's price in ₹ (optional)", min_value=0.0, value=0.0, step=1.0, key="pp_cur",
        help="Leave 0 to use the last price in your list as today's price.")

    target_in = st.number_input(
        "🎯 My target price in ₹ (optional)", min_value=0.0, value=0.0, step=1.0, key="pp_target",
        help="The price at which you would like to buy. The app tells you how realistic it is.")

    with st.expander("➕ Also check the product's reviews (optional)"):
        reviews_text = st.text_area("Paste reviews of this product, one per line:", height=140, key="pp_reviews")
        st.caption("Adds a Review Trust Score to the advice, so you also know if the reviews can be trusted.")

    tab1, tab2 = st.tabs(["📋 Paste price history", "📁 Upload CSV"])

    with tab1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        price_only = st.checkbox("My list has only prices (one per day, oldest first, the last one is today)",
                                 key="pp_only")
        pasted = st.text_area(
            "Price history (one per line):", height=200, key="pp_paste",
            placeholder="2026-09-01, 2499\n2026-09-20, 2299\n2026-10-05, 1899\n...")
        if st.button("📈 Predict Price", type="primary", key="pp_btn_paste"):
            df, bad = parse_price_history(pasted, price_only)
            if bad:
                st.warning(f"⚠️ I could not read {len(bad)} line(s), for example: `{bad[0][:60]}`")
            if df.empty:
                if bad and not price_only:
                    st.error("❌ I could not find dates in your list. If it has only prices, tick the box above.")
                else:
                    st.warning("⚠️ Please paste your price history first.")
            else:
                run_price_analysis(df, product_name, current_in, reviews_text, target_in)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        up = st.file_uploader("Upload CSV file", type=["csv"], key="pp_csv")
        if up is not None:
            try:
                table = pd.read_csv(up, dtype=str, keep_default_na=False)
            except Exception:
                table = None
                st.error("❌ Could not read this file as a CSV.")
            if table is not None and len(table.columns):
                st.dataframe(table.head())
                none_label = "(none — one price per day, oldest first)"
                date_choice = st.selectbox("Column with the date:", [none_label] + list(table.columns), key="pp_datecol")
                price_col = st.selectbox("Column with the price:", list(table.columns), key="pp_pricecol")
                if st.button("📈 Predict Price", type="primary", key="pp_btn_csv"):
                    df, bad = history_from_table(table, None if date_choice == none_label else date_choice, price_col)
                    if bad:
                        st.warning(f"⚠️ {bad} row(s) could not be read and were skipped.")
                    if df.empty:
                        st.error("❌ No usable prices found. Check the date and price columns you selected.")
                    else:
                        run_price_analysis(df, product_name, current_in, reviews_text, target_in)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"; st.rerun()


# ============================================================
# BATCH CHECKER PAGE (with logging)
# ============================================================
def run_batch_predictions(review_list, model, vectorizer):
    results = []
    logs = []
    progress = st.progress(0)
    total = len(review_list)
    for i, text in enumerate(review_list):
        label, confidence, _ = predict_review(text, model, vectorizer)
        logs.append({"review": text, "prediction": label, "confidence": round(confidence, 1)})
        results.append({
            "review": text,
            "prediction": label,
            "confidence (%)": round(confidence, 1),
            "red flags": len(find_red_flags(text)),
        })
        progress.progress((i + 1) / total)
    log_many_to_sheet(logs)
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
            lines, skipped = normalize_reviews(pasted_text.split("\n"))
            if not lines:
                st.warning("⚠️ Please paste at least one review.")
            else:
                result_df = run_batch_predictions(lines, model, vectorizer)
                show_batch_results(result_df)
                if skipped:
                    st.caption(f"Empty lines skipped: {skipped}")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV or TXT file", type=["csv", "txt"])

        if uploaded_file is not None:
            df = read_review_file(uploaded_file, "batch")
            st.write("Preview of uploaded file:")
            st.dataframe(df.head())

            col_name = st.selectbox("Select the column that contains the review text:",
                                    df.columns, format_func=str)

            if st.button("🔍 Check All Reviews", type="primary"):
                reviews, skipped = normalize_reviews(df[col_name].tolist())
                if not reviews:
                    st.warning("⚠️ No review text found in that column.")
                else:
                    result_df = run_batch_predictions(reviews, model, vectorizer)
                    show_batch_results(result_df)
                    if skipped:
                        st.caption(f"Empty rows skipped: {skipped}")
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
def prepare_logs(df):
    """Adds proper date/time and numeric columns to the raw log table (times shown in IST)."""
    d = df.copy()
    if "user" not in d.columns:
        d["user"] = "unknown"
    d["user"] = d["user"].astype(str)
    d["ts"] = pd.to_datetime(d["timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    d["is_fake"] = d["prediction"].astype(str).str.contains("Fake")
    d["confidence_num"] = pd.to_numeric(d["confidence"], errors="coerce")
    return d


def dashboard_page():
    st.markdown('<h1 class="hero-title">📈 Admin Dashboard</h1>', unsafe_allow_html=True)

    # Role-based access: normal users are blocked even if they reach this page somehow.
    if not is_admin():
        st.error("🚫 Access denied. The Dashboard is available to administrators only.")
        if st.button("⬅️ Back to Home"):
            st.session_state.page = "Home"; st.rerun()
        return

    st.write("Statistics from every review checked on this system.")

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
                result = call_admin_script({"type": "clear_logs"})
                if result and result.get("status") == "ok":
                    st.cache_data.clear()
                    st.success("✅ All data cleared!")
                    st.rerun()
                elif result and result.get("status") in ("no_key", "unauthorized"):
                    st.error("🔑 ADMIN_KEY is missing or does not match. Check Streamlit secrets and Code.gs.")
                else:
                    st.error("❌ Could not clear data. Try again.")

    df = get_dashboard_data()

    if df is None:
        st.error("❌ Could not connect to the Google Sheet. Check the Apps Script URL and try again.")
    elif len(df) == 0:
        st.info("No reviews have been logged yet. Go check a few reviews first!")
    else:
        d_all = prepare_logs(df)

        period = st.radio("📅 Period", ["Last 7 days", "Last 30 days", "All time"],
                          index=1, horizontal=True)
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        d = d_all
        if period == "Last 7 days":
            d = d_all[d_all["ts"] >= now - pd.Timedelta(days=7)]
        elif period == "Last 30 days":
            d = d_all[d_all["ts"] >= now - pd.Timedelta(days=30)]

        if len(d) == 0:
            st.info(f"No reviews were checked in this period ({period.lower()}). Try 'All time'.")
        else:
            total = len(d)
            fake_count = int(d["is_fake"].sum())
            real_count = total - fake_count
            today_count = int((d["ts"].dt.date == now.date()).sum())

            # ---------- Key numbers ----------
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Reviews Checked", total)
            c2.metric("Fake Detected", fake_count)
            c3.metric("Real Detected", real_count)
            c4, c5, c6 = st.columns(3)
            c4.metric("Fake %", f"{fake_count / total * 100:.1f}%")
            c5.metric("Active Users", int(d["user"].nunique()))
            c6.metric("Checked Today", today_count)
            st.markdown('</div>', unsafe_allow_html=True)

            # ---------- Daily trend ----------
            dd = d.dropna(subset=["ts"]).copy()
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.subheader("📅 Daily Trend")
            if len(dd) == 0:
                st.info("Dates are not available for these rows, so the trend cannot be drawn.")
            else:
                dd["day"] = dd["ts"].dt.normalize()
                daily = dd.groupby(["day", "is_fake"]).size().unstack(fill_value=0)
                for col in (True, False):
                    if col not in daily.columns:
                        daily[col] = 0
                full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
                daily = daily.reindex(full_range, fill_value=0)
                labels = [x.strftime("%d %b") for x in daily.index]
                xs = list(range(len(labels)))
                step = max(1, len(labels) // 10)

                fig_a, ax_a = plt.subplots(figsize=(7, 3.2))
                ax_a.bar(xs, daily[False], color="#27AE60", label="Real")
                ax_a.bar(xs, daily[True], bottom=daily[False], color="#C0392B", label="Fake")
                ax_a.set_xticks(xs[::step])
                ax_a.set_xticklabels(labels[::step], rotation=45, ha="right")
                ax_a.set_ylabel("Reviews checked")
                ax_a.set_title("Reviews checked per day")
                ax_a.legend()
                fig_a.tight_layout()
                st.pyplot(fig_a)

                day_total = daily[True] + daily[False]
                fake_pct = (daily[True] / day_total.where(day_total > 0) * 100)
                fig_b, ax_b = plt.subplots(figsize=(7, 3.0))
                ax_b.plot(xs, fake_pct.values, marker="o", color="#C0392B")
                ax_b.set_ylim(0, 100)
                ax_b.set_xticks(xs[::step])
                ax_b.set_xticklabels(labels[::step], rotation=45, ha="right")
                ax_b.set_ylabel("Fake %")
                ax_b.set_title("Share of fake reviews per day")
                ax_b.grid(alpha=0.3)
                fig_b.tight_layout()
                st.pyplot(fig_b)
            st.markdown('</div>', unsafe_allow_html=True)

            # ---------- Top users ----------
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.subheader("🏆 Most Active Users")
            g = d.groupby("user").agg(checks=("is_fake", "size"), fake=("is_fake", "sum")).reset_index()
            g["fake %"] = (g["fake"] / g["checks"] * 100).round(1)
            top = g.sort_values("checks", ascending=False).head(5)
            fig_u, ax_u = plt.subplots(figsize=(7, 0.8 + 0.5 * len(top)))
            ax_u.barh(top["user"][::-1], top["checks"][::-1], color="#2E86C1")
            ax_u.set_xlabel("Reviews checked")
            fig_u.tight_layout()
            st.pyplot(fig_u)
            st.dataframe(top.rename(columns={"user": "user", "checks": "reviews checked", "fake": "fake found"}))
            st.markdown('</div>', unsafe_allow_html=True)

            # ---------- Fake vs Real pie ----------
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.subheader("Fake vs Real Distribution")
            fig, ax = plt.subplots(figsize=(4, 4))
            if fake_count == 0 or real_count == 0:
                ax.pie([1], labels=["Fake" if fake_count else "Real"],
                       colors=["#C0392B" if fake_count else "#27AE60"])
            else:
                ax.pie([fake_count, real_count], labels=["Fake", "Real"], autopct="%1.1f%%",
                       colors=["#C0392B", "#27AE60"], startangle=90)
            ax.axis("equal")
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

            # ---------- Recent activity ----------
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.subheader("Recent Activity")
            recent = d.sort_values("ts", ascending=False).head(15)[["ts", "review", "prediction", "confidence", "user"]].copy()
            recent["ts"] = recent["ts"].dt.strftime("%d %b %Y, %I:%M %p")
            st.dataframe(recent.rename(columns={"ts": "time (IST)"}))
            st.markdown('</div>', unsafe_allow_html=True)

    # ---------- User feedback (always shown, independent of the period filter) ----------
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.subheader("💬 User Feedback on Predictions")
    admin_key = get_admin_key()
    if not admin_key:
        st.info("Add `ADMIN_KEY` to Streamlit secrets to see user feedback here.")
    else:
        fb = get_feedback_data(admin_key)
        if fb is None:
            st.error("❌ Could not load feedback. Check that ADMIN_KEY matches in Code.gs and Streamlit secrets.")
        elif len(fb) == 0:
            st.info("No feedback yet. Users can rate predictions with 👍 / 👎 on the Review Checker page.")
        else:
            n = len(fb)
            correct_n = int((fb["verdict"].astype(str) == "correct").sum())
            wrong_n = n - correct_n
            f1, f2, f3 = st.columns(3)
            f1.metric("Feedback Received", n)
            f2.metric("Marked Correct 👍", correct_n)
            f3.metric("Marked Wrong 👎", wrong_n)
            st.progress(correct_n / n)
            st.caption(f"User-rated accuracy: **{correct_n / n * 100:.1f}%** "
                       "(based on users who chose to give feedback, so it is only an estimate).")

            wrong = fb[fb["verdict"].astype(str) == "wrong"]
            if len(wrong):
                st.markdown("**Predictions users said were wrong:**")
                st.dataframe(wrong[["review", "model_said", "actual_label", "user"]])
            csv_fb = fb.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download all feedback (CSV) — for retraining the model",
                               csv_fb, "user_feedback.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"; st.rerun()


# ============================================================
# MANAGE USERS PAGE (admin only)
# ============================================================
ADMIN_ERRORS = {
    "not_found": "❌ User not found.",
    "weak_password": "❌ Password is too weak (min 8 characters, with a letter and a number).",
    "unauthorized": "🔑 ADMIN_KEY does not match between Streamlit secrets and Code.gs.",
    "no_key": "🔑 ADMIN_KEY is missing in Streamlit secrets.",
    "last_admin": "❌ You cannot remove or demote the last admin.",
    "invalid_role": "❌ Invalid role.",
}


def load_users():
    st.session_state.users_data = call_admin_script({"type": "list_users"})


def run_admin_action(payload, success_msg):
    result = call_admin_script(payload)
    if result is None:
        st.error("❌ Could not reach the account database. Try again.")
    elif result.get("status") == "ok":
        st.session_state.um_flash = success_msg
        load_users()
        st.rerun()
    else:
        st.error(ADMIN_ERRORS.get(result.get("status"), "❌ Something went wrong."))


def manage_users_page():
    st.markdown('<h1 class="hero-title">👥 Manage Users</h1>', unsafe_allow_html=True)

    def back_button(key):
        if st.button("⬅️ Back to Home", key=key):
            st.session_state.page = "Home"; st.rerun()

    if not is_admin():
        st.error("🚫 Access denied. This page is available to administrators only.")
        back_button("um_back_denied")
        return

    if not get_apps_script_url():
        st.warning("⚠️ The account database is not connected yet.")
        back_button("um_back_nodb")
        return

    if not get_admin_key():
        st.warning("⚠️ Add `ADMIN_KEY` to your Streamlit secrets (same text as in Code.gs) to use this page.")
        back_button("um_back_nokey")
        return

    st.write("Passwords are stored as hashes, so **nobody can see them** — not even the admin. "
             "If a user forgets their password, set a new one for them below.")

    if st.session_state.get("um_flash"):
        st.success(st.session_state.pop("um_flash"))

    refresh = st.button("🔄 Refresh")
    if refresh or "users_data" not in st.session_state:
        load_users()

    res = st.session_state.get("users_data")
    if res is None:
        st.error("❌ Could not load users. Check the Apps Script URL and try again.")
        back_button("um_back_err")
        return
    if res.get("status") != "ok":
        st.error(ADMIN_ERRORS.get(res.get("status"), "❌ Could not load users."))
        back_button("um_back_err2")
        return

    users = res.get("users", [])
    if not users:
        st.info("No registered users yet.")
        back_button("um_back_empty")
        return

    df = pd.DataFrame(users)
    df["status"] = df["locked"].map(lambda x: "🔒 Locked" if x else "Active")

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Users", len(df))
    c2.metric("Admins", int((df["role"] == "admin").sum()))
    c3.metric("Locked Accounts", int(df["locked"].sum()))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.subheader("All Users")
    q = st.text_input("🔍 Search by username or email", key="um_search")
    view = df
    if q.strip():
        mask = (df["username"].astype(str).str.contains(q, case=False, regex=False)
                | df["email"].astype(str).str.contains(q, case=False, regex=False))
        view = df[mask]
    st.dataframe(view[["username", "email", "role", "status"]])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.subheader("Manage a user")
    selected = st.selectbox("Select user", df["username"].tolist(), key="um_selected")
    row = df[df["username"] == selected].iloc[0]
    st.caption(f"📧 {row['email']}  ·  Role: **{row['role']}**  ·  Status: **{row['status']}**")

    t1, t2, t3, t4 = st.tabs(["🔑 Reset Password", "🛡️ Change Role", "🔓 Unlock", "🗑️ Delete"])

    with t1:
        with st.form("um_reset_form", clear_on_submit=True):
            np1 = st.text_input("New password", type="password")
            np2 = st.text_input("Confirm new password", type="password")
            st.caption("Min 8 characters, with at least one letter and one number.")
            submitted = st.form_submit_button("Set New Password", type="primary")
        if submitted:
            err = password_problem(np1) if np1 else "Please enter a new password."
            if err:
                st.error(f"❌ {err}")
            elif np1 != np2:
                st.error("❌ Passwords do not match.")
            else:
                run_admin_action(
                    {"type": "admin_reset_password", "username": selected, "password": np1},
                    f"✅ Password for {selected} was reset (and the account unlocked).")

    with t2:
        new_role = st.selectbox("Role", ["user", "admin"],
                                index=0 if row["role"] != "admin" else 1, key="um_role")
        if st.button("Update Role", key="um_role_btn"):
            if new_role == row["role"]:
                st.info("ℹ️ That is already the user's role.")
            else:
                run_admin_action({"type": "set_role", "username": selected, "role": new_role},
                                 f"✅ {selected} is now '{new_role}'.")

    with t3:
        if row["locked"]:
            st.warning("This account is locked because of too many wrong password attempts.")
        else:
            st.write("This account is not locked.")
        if st.button("Unlock Account", key="um_unlock_btn", disabled=not bool(row["locked"])):
            run_admin_action({"type": "unlock_user", "username": selected},
                             f"✅ {selected} was unlocked.")

    with t4:
        st.warning("⚠️ This permanently deletes the account. Their past review history stays in the logs.")
        sure = st.checkbox(f"Yes, delete {selected}", key="um_delete_sure")
        if st.button("Delete User", key="um_delete_btn", type="primary", disabled=not sure):
            run_admin_action({"type": "delete_user", "username": selected},
                             f"✅ {selected} was deleted.")
    st.markdown('</div>', unsafe_allow_html=True)

    back_button("um_back_main")


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
    pages = ["Home", "Review Checker", "Product Trust Score", "Price Prediction", "Batch Checker", "Insights", "My History"]
    if is_admin():
        pages.append("Dashboard")
        pages.append("Manage Users")

    # If the stored page is not allowed for this role, send the user home.
    if st.session_state.page not in pages:
        st.session_state.page = "Home"

    nav_choice = st.sidebar.radio("Go to:", pages, index=pages.index(st.session_state.page))
    st.session_state.page = nav_choice

    with st.sidebar.expander("🎨 Theme"):
        st.radio("Choose a look", list(THEMES), key="theme", label_visibility="collapsed")

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
    elif st.session_state.page == "Price Prediction":
        price_prediction_page()
    elif st.session_state.page == "Batch Checker":
        batch_checker_page()
    elif st.session_state.page == "Insights":
        insights_page()
    elif st.session_state.page == "My History":
        my_history_page()
    elif st.session_state.page == "Dashboard":
        dashboard_page()
    elif st.session_state.page == "Manage Users":
        manage_users_page()
