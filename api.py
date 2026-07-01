from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load('sonar_model.pkl')

@app.post("/predict")
async def predict(data: list):
    input_data = np.array(data).reshape(1, -1)
    prediction = model.predict(input_data)  
    return {"prediction": int(prediction[0])}