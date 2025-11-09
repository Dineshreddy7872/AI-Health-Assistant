import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import os

# Page Congiguration
st.set_page_config(page_title="AI Health Assistant 🧠", page_icon="💉", layout="wide")

# Custom Styling
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #141E30 0%, #243B55 100%);
    color: white;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #243B55 0%, #141E30 100%);
}
h1, h2, h3 { color: #00FFFF; }
.stButton>button {
    background-color: #00B4D8;
    color: white;
    border-radius: 10px;
    font-size: 18px;
}
.stButton>button:hover { background-color: #0077B6; }
</style>
""", unsafe_allow_html=True)

# Sidebar Menu
with st.sidebar:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT6mpoQuMtyruBh6lGBHHFW9MzUItgB5io-3X2tqJVyzg&s", width=100)
    selected = option_menu(
        "Menu",
        ["🏠 Home", "🩺 Predict Health", "📊 Model Info", "👨‍💻 About"],
        icons=["house", "activity", "bar-chart", "person"],
        default_index=1
    )

# Load Model and Artifacts
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("models/health_model.h5")
    scaler = joblib.load("models/scaler.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return model, scaler, le

model, scaler, le = load_artifacts()

# Home Page
if selected == "🏠 Home":
    st.title("🧠 Welcome to AI Health Assistant")
    st.markdown("""
    This deep learning-powered app predicts your **health condition**
    based on your vital parameters like blood pressure, cholesterol, sugar, and more.
    
    ✅ Built with TensorFlow  
    ✅ Achieved ~95% validation accuracy  
    ✅ Designed for educational and demo purposes
    """)

# Prediction Page
elif selected == "🩺 Predict Health":
    st.title("🔍 Predict Your Health Condition")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("👶 Age", 1, 100, 30)
        bp = st.number_input("❤️ Blood Pressure (mmHg)", 60, 220, 120)
    with col2:
        cholesterol = st.number_input("🩸 Cholesterol (mg/dL)", 100, 400, 180)
        sugar = st.number_input("🍬 Sugar Level (mg/dL)", 50, 400, 100)
    with col3:
        hr = st.number_input("💓 Heart Rate (bpm)", 40, 200, 75)

    if st.button("🚀 Predict Now"):
        input_data = np.array([[age, bp, cholesterol, sugar, hr]])
        scaled_input = scaler.transform(input_data)
        preds = model.predict(scaled_input)
        pred_class = np.argmax(preds)
        label = le.inverse_transform([pred_class])[0]
        confidence = np.max(preds) * 100

        classes = ["Healthy", "At Risk", "Disease"]
        colors = ['#00FF85', '#FFD93D', '#FF4B4B']

        st.success(f"### 🩺 Prediction: **{classes[label]}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        # Confidence chart
        fig, ax = plt.subplots(figsize=(5,3))
        bars = ax.bar(classes, preds[0], color=colors)
        for bar in bars:
            y = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, y+0.02, f"{y*100:.1f}%", ha='center', color='white')
        ax.set_ylim([0,1])
        st.pyplot(fig)

# model Info Page
elif selected == "📊 Model Info":
    st.title("📈 Model Information")
    st.markdown("""
    **Architecture:**
    - Dense(128, ReLU) + BatchNorm + Dropout(0.35)  
    - Dense(64, ReLU) + BatchNorm + Dropout(0.25)  
    - Dense(32, ReLU)  
    - Output: Dense(3, Softmax)
    """)
    st.markdown("**Framework:** TensorFlow / Keras  \n**Accuracy:** ~95%")

    if os.path.exists("models/confusion_matrix.png"):
        st.image("models/confusion_matrix.png", caption="Confusion Matrix", use_column_width=True)
    else:
        st.warning("Confusion matrix not found. Train the model first.")

# About Page
elif selected == "👨‍💻 About":
    st.title("👨‍💻 About Developer")
    st.markdown("""
    **Name:** Dinesh Reddy  
    **Role:** AI & ML Enthusiast  
    **Project:** Deep Learning Health Prediction  
    **Stack:** Python | TensorFlow | Streamlit  
    """)
    st.image("https://www.concretecms.com/application/files/1716/8329/9745/Medical_Website_Design.jpg", width=400)
