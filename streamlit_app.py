import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.main {
    background-color: rgba(0,0,0,0.6);
    padding: 2rem;
    border-radius: 15px;
}
h1, h2, h3 {
    color: #E50914;
}
.stButton>button {
    background-color: #E50914;
    color: white;
    font-size: 1.2rem;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
}
.stButton>button:hover {
    background-color: #ff1f1f;
}
.footer {
    margin-top: 3rem;
    padding: 2rem;
    background: rgba(0,0,0,0.7);
    border-radius: 15px;
}
.metric-card {
    background: rgba(255,255,255,0.1);
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
MODEL_PATH = "artifacts/models/kmeans_model.pkl"
SCALER_PATH = "artifacts/models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ------------------ HEADER ------------------
st.title("📊 Customer Segmentation Dashboard")
st.markdown("### 🚀 AI-powered customer clustering (MLOps Pipeline)")

st.divider()

# ------------------ INPUT SECTION ------------------
st.subheader("🧾 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    recency = st.slider("📅 Recency (days)", 0, 365, 10)
    frequency = st.slider("🔁 Frequency", 1, 50, 5)

with col2:
    monetary = st.number_input("💰 Monetary Value", min_value=0.0, value=2000.0)
    quantity = st.slider("📦 Quantity", 1, 500, 50)

with col3:
    discount = st.slider("🏷️ Discount", 0.0, 1.0, 0.1)
    delivery_days = st.slider("🚚 Delivery Days", 0, 30, 3)
    customer_rating = st.slider("⭐ Rating", 1.0, 5.0, 4.5)

st.divider()

# ------------------ PREDICTION ------------------
if st.button("🚀 Predict Customer Segment"):

    input_df = pd.DataFrame([{
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "quantity": quantity,
        "discount": discount,
        "delivery_days": delivery_days,
        "customer_rating": customer_rating
    }])

    try:
        input_df = np.log1p(input_df)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.success(f"🎯 Predicted Cluster: {prediction}")

        # Cluster meaning
        cluster_map = {
                    0: "🔥 Active High-Value Customers",
                    1: "💡 Dormant / Low-Engagement Customers"
                }

        if prediction in cluster_map:
            st.info(cluster_map[prediction])

        # Metrics display
        st.subheader("📊 Input Summary")
        m1, m2, m3 = st.columns(3)

        m1.metric("Recency", recency)
        m2.metric("Frequency", frequency)
        m3.metric("Monetary", monetary)

    except Exception as e:
        st.error(f"Error: {e}")

st.divider()

# ------------------ FOOTER ------------------
footer_col1, footer_col2, footer_col3 = st.columns([1,2,1])

with footer_col2:
    st.markdown(
        """
        <div class='footer'>
            <div style='text-align: center;'>
                <h4 style='color: #E50914; margin-bottom: 1rem; font-size: 1.5rem;'>👨‍💻 Developed By</h4>
                <p style='font-size: 1.4rem; font-weight: bold; color: #FFFFFF; margin-bottom: 0.5rem;'>
                    Harshith Narasimhamurthy
                </p>
                <p style='margin-bottom: 0.5rem; font-size: 1.1rem; color: #E6E6E6;'>
                    📧 harshithnchandan@gmail.com | 📱 +919663918804
                </p>
                <p style='margin-bottom: 1rem; font-size: 1.1rem;'>
                    🔗 <a href='https://www.linkedin.com/in/harshithnarasimhamurthy69/' target='_blank' 
                       style='color: #E50914; text-decoration: none; font-weight: bold;'>
                       Connect with me on LinkedIn
                    </a>
                </p>
                <p style='font-size: 1rem; color: rgba(255,255,255,0.8);'>
                    Powered by Scikit-learn & Streamlit | Customer Segmentation MLOps
                </p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )