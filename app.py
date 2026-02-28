import os
import sys
from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            data = CustomData(
                Date=request.form.get('Date'),
                Location=request.form.get('Location'),
                MinTemp=float(request.form.get('MinTemp')),
                MaxTemp=float(request.form.get('MaxTemp')),
                Rainfall=float(request.form.get('Rainfall')),
                Evaporation=float(request.form.get('Evaporation')),
                Sunshine=float(request.form.get('Sunshine')),
                WindGustDir=request.form.get('WindGustDir'),
                WindGustSpeed=float(request.form.get('WindGustSpeed')),
                WindDir9am=request.form.get('WindDir9am'),
                WindDir3pm=request.form.get('WindDir3pm'),
                WindSpeed9am=float(request.form.get('WindSpeed9am')),
                WindSpeed3pm=float(request.form.get('WindSpeed3pm')),
                Humidity9am=float(request.form.get('Humidity9am')),
                Humidity3pm=float(request.form.get('Humidity3pm')),
                Pressure9am=float(request.form.get('Pressure9am')),
                Pressure3pm=float(request.form.get('Pressure3pm')),
                Cloud9am=float(request.form.get('Cloud9am')),
                Cloud3pm=float(request.form.get('Cloud3pm')),
                Temp9am=float(request.form.get('Temp9am')),
                Temp3pm=float(request.form.get('Temp3pm')),
                RainToday=request.form.get('RainToday')
            )

            pred_df = data.get_data_as_data_frame()

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            if results[0] == 1 or results[0] == 'Yes':
                prediction_text = "🌧️ ALERT: Rain Expected Tomorrow! (Yes)"
            else:
                prediction_text = "☀️ GREAT: No Rain Expected Tomorrow. (No)"

            return render_template('index.html', results=prediction_text)

        except Exception as e:
            return render_template('index.html', results=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)