import streamlit as st
import joblib
import numpy as np

model = joblib.load("D:\BINUS\MD\Session7_Assignment\Iris_RF_Model.pkl")

def main():
    st.title("Machine Leaning Iris Prediction Model Deployment")
    
    sepal_length = st.slider('sepal_length', min_value=0.0, max_value=10.0, value=0.1)
    sepal_width = st.slider('sepal_width', min_value=0.0, max_value=10.0, value=0.1)
    petal_length = st.slider('petal_length', min_value=0.0, max_value=10.0, value=0.1)
    petal_width = st.slider('petal_width', min_value=0.0, max_value=10.0, value=0.1)
    
    if st.button("Predict"):
        features = [sepal_length, sepal_width, petal_length, petal_width]
        result = make_predictions(features)
        if result is 1:
            result = "Iris-setosa"
        elif result is 2:
            result = "iris-versicolor"
        else:
            result = "Iris-virginica"
        st.success(f"The prediction flower is: {result}")
        
def make_predictions(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]
    
if __name__ == "__main__":
    main()