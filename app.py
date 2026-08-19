import streamlit as st
import joblib
import re

# ---------- Page Setup ----------
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🕵️",
    layout="centered"
)

# ---------- Load Model & Vectorizer ----------
@st.cache_resource
def load_model():
    model = joblib.load("fake_review_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ---------- Text Cleaning (same as training) ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def predict_review(review_text):
    cleaned = clean_text(review_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    confidence = max(probability) * 100
    label = "Fake (Computer Generated)" if prediction == 1 else "Real (Genuine)"
    return label, confidence, prediction

# ---------- UI ----------
st.title("🕵️ Fake Review Detection System")
st.write("Enter a product review below and the model will predict whether it looks **Real** or **Fake**.")

st.markdown("---")

review_input = st.text_area(
    "✍️ Enter a product review:",
    height=150,
    placeholder="e.g. This product is amazing! Best purchase ever, highly recommend!"
)

col1, col2 = st.columns([1, 4])
with col1:
    check_btn = st.button("🔍 Check Review", type="primary")

if check_btn:
    if review_input.strip() == "":
        st.warning("⚠️ Please enter a review first.")
    else:
        label, confidence, prediction = predict_review(review_input)

        st.markdown("---")
        if prediction == 1:
            st.error(f"### ❌ {label}")
        else:
            st.success(f"### ✅ {label}")

        st.metric("Model Confidence", f"{confidence:.1f}%")
        st.progress(confidence / 100)

st.markdown("---")

with st.expander("ℹ️ About this project"):
    st.write("""
    This is a **Fake Review Detection System** built using Machine Learning.

    - **Model:** Logistic Regression
    - **Feature Extraction:** TF-IDF (Term Frequency–Inverse Document Frequency)
    - **Dataset:** 40,432 labeled product reviews (Kaggle)
    - **Accuracy:** ~89.9% on unseen test data

    Built as part of a BCA project — *Shri Gavisiddeswar Arts, Commerce & Science College, Koppal*.
    """)

# Sample reviews to try
with st.expander("💡 Try these sample reviews"):
    st.write("**Sample 1 (often Real):**")
    st.code("Bought this for my kitchen last month. Works well but the handle feels a bit loose after a few uses.")
    st.write("**Sample 2 (often Fake):**")
    st.code("Amazing amazing amazing product!!! Best thing I have ever bought in my life!!! Everyone must buy this now!!!")
