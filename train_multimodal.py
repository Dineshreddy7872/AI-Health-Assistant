# Import necessary libraries
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# setup directories and seeds

os.makedirs("models", exist_ok=True)
np.random.seed(42)
tf.random.set_seed(42)


# Synthetic Data Generation

def generate_data(n=3000):
    rows = []
    for i in range(n):
        label = np.random.choice([0, 1, 2], p=[0.45, 0.35, 0.20])
        if label == 0:  # Healthy
            age = np.random.normal(30, 8)
            bp = np.random.normal(115, 8)
            chol = np.random.normal(170, 20)
            sugar = np.random.normal(90, 10)
            hr = np.random.normal(72, 6)
        elif label == 1:  # At Risk
            age = np.random.normal(45, 10)
            bp = np.random.normal(135, 12)
            chol = np.random.normal(210, 25)
            sugar = np.random.normal(110, 15)
            hr = np.random.normal(80, 8)
        else:  # Disease
            age = np.random.normal(60, 10)
            bp = np.random.normal(155, 15)
            chol = np.random.normal(260, 30)
            sugar = np.random.normal(150, 25)
            hr = np.random.normal(90, 12)
        rows.append([age, bp, chol, sugar, hr, label])
    df = pd.DataFrame(rows, columns=["age", "bp", "cholesterol", "sugar", "hr", "label"])
    return df

df = generate_data()
X = df[["age", "bp", "cholesterol", "sugar", "hr"]].values
y = df["label"].values


# data preprocessing

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, "models/label_encoder.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
)

class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights = {i: w for i, w in enumerate(class_weights)}
print("Class weights:", class_weights)


# model building

model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.35),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    layers.Dense(32, activation="relu"),
    layers.Dense(3, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()


# training the model

es = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)
ckpt = callbacks.ModelCheckpoint("models/health_model.h5", monitor="val_loss", save_best_only=True, verbose=1)
rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=80,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[es, ckpt, rlr],
    verbose=2
)


# evaluate and save confusion matrix

model = tf.keras.models.load_model("models/health_model.h5")
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:")
target_names = [str(c) for c in le.classes_]
print(classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("models/confusion_matrix.png", bbox_inches="tight")
plt.close()

print("✅ Model, scaler, label encoder, and confusion matrix saved successfully!")
