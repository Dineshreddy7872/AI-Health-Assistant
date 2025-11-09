# AI-Health-Assistant

Deep Learning-powered Health Condition Prediction System



1) 📘 Overview


The AI Health Assistant is an interactive web application built using TensorFlow and Streamlit that predicts a user's health condition based on vital parameters such as blood pressure, cholesterol, sugar, and heart rate.

It uses a deep learning classifier trained on synthetic health data, achieving around 95% accuracy.

🚀 Features

✅ Predicts health condition as Healthy, At Risk, or Disease

✅ Built with TensorFlow (Keras) and Streamlit

✅ Real-time, interactive UI with custom themes

✅ Displays prediction confidence and visual confidence chart

✅ Includes model training script and saved artifacts

✅ Clean modular code for educational and deployment use


🧩 Tech Stack

Category	Tools & Libraries

Frontend	Streamlit, Streamlit Option Menu

Backend / ML	TensorFlow, Keras, Scikit-learn

Data Handling	Pandas, NumPy

Visualization	Matplotlib, Seaborn

Utilities	Joblib


🧠 Model Architecture

Model Summary:

Dense(128, ReLU) + BatchNorm + Dropout(0.35)

Dense(64, ReLU) + BatchNorm + Dropout(0.25)

Dense(32, ReLU)

Output: Dense(3, Softmax)

Training Setup:

Optimizer: Adam (LR = 1e-3)

Loss: Sparse Categorical Crossentropy

Epochs: 80

EarlyStopping + ReduceLROnPlateau

Accuracy: ~95%


📊 Example Output


Prediction:


🩺 "At Risk"

Confidence: 87.6%



Visualization:


Displays a real-time bar chart showing prediction confidence across all classes.


👨‍💻 About Developer


Name: Dinesh Reddy

Role: AI & ML Enthusiast

Focus: Data Science | Deep Learning | Generative AI

Portfolio: [Coming Soon 🔥]



