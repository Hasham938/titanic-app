# app.py
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic — Survival Prediction")
st.write("Enter passenger info and click Predict. Make sure `titanic_model.pkl` and `model_columns.pkl` are in the same folder as this app.")

# -------------------------
# 1) Load model & columns
# -------------------------
MODEL_PATH = "titanic_model.pkl"
COLS_PATH = "model_columns.pkl"
model = None
model_cols = None

# Load trained model
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
else:
    st.error(f"Model file not found: {MODEL_PATH}")

# Load column order
if os.path.exists(COLS_PATH):
    try:
        model_cols = joblib.load(COLS_PATH)
        # Ensure 'Survived' is removed if present
        model_cols = [c for c in model_cols if c != 'Survived']
    except Exception as e:
        st.warning(f"Could not read {COLS_PATH}: {e}")
        model_cols = None
else:
    st.warning(f"Columns file not found: {COLS_PATH}")
    
# Default columns (if model_columns.pkl is missing)
DEFAULT_FEATURES = ['Pclass','Age','SibSp','Parch','Fare','Sex_male','Embarked_Q','Embarked_S']
if model_cols is None:
    model_cols = DEFAULT_FEATURES

# -------------------------
# 2) Sidebar: user inputs
# -------------------------
st.sidebar.header("Passenger info")

pclass = st.sidebar.selectbox("Pclass", [1,2,3], index=2)
sex = st.sidebar.selectbox("Sex", ["male","female"], index=0)
age = st.sidebar.number_input("Age (years)", min_value=0.0, max_value=120.0, value=30.0, step=0.5)
sibsp = st.sidebar.number_input("SibSp (siblings/spouses aboard)", min_value=0, max_value=10, value=0, step=1)
parch = st.sidebar.number_input("Parch (parents/children aboard)", min_value=0, max_value=10, value=0, step=1)
fare = st.sidebar.number_input("Fare", min_value=0.0, max_value=10000.0, value=7.25, step=0.1)
embarked = st.sidebar.selectbox("Embarked", ["S","C","Q","Unknown"], index=0)

st.markdown("---")
st.write("**Notes:** The app uses one-hot fields `Sex_male`, `Embarked_Q`, `Embarked_S`. Missing columns will be filled with 0.")

# -------------------------
# 3) Build input row
# -------------------------
def build_input_row(pclass, age, sibsp, parch, fare, sex, embarked):
    row = {
        'Pclass': int(pclass),
        'Age': float(age),
        'SibSp': int(sibsp),
        'Parch': int(parch),
        'Fare': float(fare),
        'Sex_male': 1 if sex=='male' else 0,
        'Embarked_Q': 1 if embarked=='Q' else 0,
        'Embarked_S': 1 if embarked=='S' else 0
    }
    return row

input_row = build_input_row(pclass, age, sibsp, parch, fare, sex, embarked)
input_df = pd.DataFrame([input_row])

# Align columns with model_cols
aligned_df = input_df.reindex(columns=model_cols, fill_value=0).astype(float)

st.write("### Input features (aligned to model):")
st.dataframe(aligned_df.T.rename(columns={0:"value"}))

# -------------------------
# 4) Predict
# -------------------------
if st.button("Predict survival"):
    if model is None:
        st.error("No model loaded. Place `titanic_model.pkl` in this folder.")
    else:
        try:
            # Predict probability
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(aligned_df)[:,1][0]
                pred_class = 1 if prob >= 0.5 else 0
            else:
                pred_class = int(model.predict(aligned_df)[0])
                prob = float(pred_class)
            if pred_class == 1:
                st.success(f"🎉 Predicted: SURVIVED")
            else:
                st.error(f"💀 Predicted: NOT SURVIVED")
            st.info(f"Survival probability: **{prob*100:.1f}%**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------
# 5) Troubleshooting
# -------------------------
st.markdown("---")
st.write("### Troubleshooting / reminders")
st.write("""
- Ensure `titanic_model.pkl` and `model_columns.pkl` exist in the same folder as `app.py`.
- Filenames are case-sensitive.
- Model must be trained without the Cabin column to match this app.
- If you trained your model with different features, re-save `model_columns.pkl`.
""")
