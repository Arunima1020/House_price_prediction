import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

@st.cache_data
def load_data():
    df = pd.read_excel(r"house_prices.xlsx")  

    drop_cols = ['date','floors','waterfront','view','condition',
                 'yr_built','yr_renovated','street','city','statezip','country']
    df = df.drop(columns=drop_cols, errors='ignore')

    # Drop missing values if any
    df = df.dropna()

    return df

data = load_data()


X = data.drop(columns=["price"])   
y = data["price"]                  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.title("🏠 House Price Prediction App")
st.write("Use the sliders below to set house features and click **Predict Price**.")

input_data = {}
for col in X.columns:
    min_val = float(data[col].min())
    max_val = float(data[col].max())
    mean_val = float(data[col].mean())
    input_data[col] = st.slider(f"{col}", min_val, max_val, mean_val)


if st.button("🔮 Predict Price"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted House Price: ${prediction:,.2f}")


st.subheader("📊 Model Performance")
st.write(f"**R² Score:** {score:.2f}")
st.write(f"**RMSE:** {rmse:,.2f}")



