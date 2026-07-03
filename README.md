# 🫀 Heart Disease Risk Assessment

A clinical-style, step-by-step web app that predicts the likelihood of heart disease using a K-Nearest Neighbors (KNN) machine learning model — built with **Streamlit** and trained on real clinical features.

### 🔗 [**Try the Live App →**](https://heartdiseaseprediction-gv8a3zqkx9bqzqrbayqoeh.streamlit.app/)

---

## ✨ Features

- **🧭 Step-by-step wizard flow** — Patient Info → Clinical Vitals → Symptoms → Result, with a progress indicator
- **🎨 Clean clinical UI** — calm blue theme designed to feel like a real medical screening tool
- **📊 Risk gauge** — visual probability meter (green → yellow → red) instead of a flat yes/no
- **📝 Personalized results** — predictions are addressed directly to the patient by name
- **📈 Prediction history tracker** — every assessment is logged and visualized as a risk trend chart over time
- **⚡ Instant predictions** — powered by a trained KNN classifier with a standardized feature pipeline

---

## 🧠 How It Works

1. User enters demographic details (name, age, sex)
2. User enters clinical vitals (resting BP, cholesterol, fasting blood sugar, resting ECG, max heart rate)
3. User enters symptom-related inputs (chest pain type, exercise-induced angina, oldpeak, ST slope)
4. Inputs are one-hot encoded, aligned to the model's expected feature columns, and scaled using a pre-fitted `StandardScaler`
5. The trained **KNN model** predicts the likelihood of heart disease, visualized with a risk gauge and a clear result message

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Frontend / App | [Streamlit](https://streamlit.io) |
| ML Model | Scikit-learn (K-Nearest Neighbors) |
| Data Handling | Pandas, NumPy |
| Visualization | Plotly |
| Model Persistence | Joblib |

---

## 📂 Project Structure

```
Heart_Disease_prediction/
├── app.py                     # Main Streamlit application
├── HeartdiseaseFinal.ipynb    # Model training & experimentation notebook
├── KNN_heart.pkl              # Trained KNN model
├── scaler.pkl                 # Fitted StandardScaler
├── columns.pkl                # Expected feature columns (for encoding alignment)
├── requirements.txt           # Python dependencies
└── .streamlit/
    └── config.toml            # Custom theme configuration
```

---

## 🚀 Run It Locally

```bash
# Clone the repo
git clone https://github.com/ayanpaul14/Heart_Disease_prediction.git
cd Heart_Disease_prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
python -m streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## ⚠️ Disclaimer

This tool is built for **educational and demonstration purposes only**. It is **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult a qualified healthcare provider for any health concerns.

---

## 👤 Author

**Ayan Paul**
GitHub: [@ayanpaul14](https://github.com/ayanpaul14)

---

⭐ If you found this project interesting, consider giving it a star!
