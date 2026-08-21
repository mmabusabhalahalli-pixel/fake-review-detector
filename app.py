import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
</style>
""", unsafe_allow_html=True)

VALID_USERNAME = "admin"
VALID_PASSWORD = "admin123"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Home"

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
    """Return top words pushing the prediction toward Fake and toward Real."""
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
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔍 Review Checker", type="primary"):
            st.session_state.page = "Review Checker"; st.rerun()
    with c2:
        if st.button("📂 Batch Checker"):
            st.session_state.page = "Batch Checker"; st.rerun()
    with c3:
        if st.button("📊 Insights"):
            st.session_state.page = "Insights"; st.rerun()


# ============================================================
# REVIEW CHECKER PAGE (with explanation)
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
# BATCH CHECKER PAGE
# ============================================================
def batch_checker_page():
    st.markdown('<h1 class="hero-title">📂 Batch Review Checker</h1>', unsafe_allow_html=True)
    st.write("Upload a CSV file with a column of reviews to check many reviews at once.")

    model, vectorizer = load_model()

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded file:")
        st.dataframe(df.head())

        col_name = st.selectbox("Select the column that contains the review text:", df.columns)

        if st.button("🔍 Check All Reviews", type="primary"):
            results = []
            progress = st.progress(0)
            total = len(df)
            for i, text in enumerate(df[col_name].astype(str)):
                label, confidence, _ = predict_review(text, model, vectorizer)
                results.append({"review": text, "prediction": label, "confidence (%)": round(confidence, 1)})
                progress.progress((i + 1) / total)

            result_df = pd.DataFrame(results)
            st.success(f"✅ Checked {total} reviews!")
            st.dataframe(result_df)

            fake_count = (result_df["prediction"].str.contains("Fake")).sum()
            real_count = total - fake_count
            c1, c2 = st.columns(2)
            c1.metric("Real Reviews", real_count)
            c2.metric("Fake Reviews", fake_count)

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results as CSV", csv_out, "review_results.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"; st.rerun()


# ============================================================
# INSIGHTS PAGE (Word Clouds from model weights)
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
# MAIN APP FLOW
# ============================================================
if not st.session_state.logged_in:
    login_page()
else:
    st.sidebar.title("📂 Navigation")
    pages = ["Home", "Review Checker", "Batch Checker", "Insights"]
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
