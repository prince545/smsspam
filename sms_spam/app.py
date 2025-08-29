import streamlit as st
import pickle
import os
from datetime import datetime

# --- Load Pipeline ---
@st.cache_resource
def load_pipeline():
    pipeline_path = os.path.join(os.path.dirname(__file__), 'spam_detection_pipeline.pkl')
    try:
        with open(pipeline_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        st.stop()

spam_pipeline = load_pipeline()

# --- Streamlit Page Config ---
st.set_page_config(page_title="SMS Spam Detector", page_icon="📧", layout="wide")

# --- Custom CSS ---
def set_background(spam_detected=False):
    color = "#ff4c4c" if spam_detected else "#ffffff"  # Red if spam
    st.markdown(f"""
        <style>
        body {{
            background-color: {color};
            transition: background-color 0.5s ease;
        }}
        .stTextArea textarea {{
            font-size: 1.2rem;
        }}
        .stButton>button {{
            font-size: 1.2rem;
            padding: 10px 20px;
        }}
        </style>
    """, unsafe_allow_html=True)

# --- Page Header ---
st.markdown("<h1 style='text-align:center;color:#1f77b4;'>📧 SMS Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter a message below to check if it is Spam or Not Spam.</p>", unsafe_allow_html=True)

# --- User Input ---
user_input = st.text_area("Message:", height=120, placeholder="Type your message here...")

# --- Predict Button ---
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        try:
            prediction = spam_pipeline.predict([user_input])[0]
            prediction_prob = spam_pipeline.predict_proba([user_input])[0]

            spam_detected = prediction == 1
            set_background(spam_detected)

            # Display result
            if spam_detected:
                st.markdown("## 🚨 SPAM DETECTED!")
            else:
                st.markdown("## ✅ Not Spam")

            # Show confidence
            st.write(f"**Confidence Scores:** Not Spam: {prediction_prob[0]:.2%}, Spam: {prediction_prob[1]:.2%}")

            # Timestamp
            st.caption(f"Analyzed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
