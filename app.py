import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.title("Car Price Prediction :car:")

st.image("car.jpeg", width=500, use_container_width=False)


# File path is already given
file_path = "car_price_dataset.csv"
df = pd.read_csv(file_path)

# Selecting features and target
features = ['Year', 'Engine_Size', 'Mileage', 'Fuel_Type', 'Transmission']
target = 'Price'

if all(col in df.columns for col in features + [target]):
    X = df[features]
    y = df[target]

    # One-Hot Encoding for categorical columns
    categorical_cols = ['Year', 'Fuel_Type', 'Transmission']
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(X[categorical_cols])
    new_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    X = pd.concat([X.drop(columns=categorical_cols), new_df], axis=1)

    # Standard Scaling for numerical columns
    numerical_cols = ['Engine_Size', 'Mileage']
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # User input for prediction
    st.write("### Enter Car Details for Prediction")
    year = st.number_input("Year", min_value=2005, max_value=int(df["Year"].max()),
                           value=int(df["Year"].mean()))
    engine_size = st.slider("Engine Size", min_value=float(df["Engine_Size"].min()),
                            max_value=float(df["Engine_Size"].max()),value=float(df["Engine_Size"].mean()))
    mileage = st.slider("Mileage", min_value=int(df["Mileage"].min()), max_value=int(df["Mileage"].max()),
                              value=int(df["Mileage"].mean()))
    fuel_type = st.radio("Fuel Type", df["Fuel_Type"].unique(), horizontal=True)
    transmission = st.radio("Transmission", df["Transmission"].unique(), horizontal=True)

    # Encoding user input
    user_input = pd.DataFrame([[year, engine_size, mileage, fuel_type, transmission]],
                              columns=['Year', 'Engine_Size', 'Mileage', 'Fuel_Type', 'Transmission'])
    encoded_input = encoder.transform(user_input[categorical_cols])
    new_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(categorical_cols))
    user_input = pd.concat([user_input.drop(columns=categorical_cols), new_input_df], axis=1)
    user_input[numerical_cols] = scaler.transform(user_input[numerical_cols])

    st.divider()
    # Predict button
    if st.button("Predict"):
        prediction = model.predict(user_input)
        st.write(f"### Predicted Car Price: ${prediction[0]:,.2f}")
else:
    st.error("Dataset must contain the required columns: Year, Engine_Size, Mileage, Fuel_Type, Transmission, and Price.")
