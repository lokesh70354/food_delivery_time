from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained pipeline
with open("elasticnet_delivery_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect form data
    data = {
        "Distance_km": float(request.form["distance"]),
        "Weather": request.form["weather"],
        "Traffic_Level": request.form["traffic"],
        "Time_of_Day": request.form["time"],
        "Vehicle_Type": request.form["vehicle"],
        "Preparation_Time_min": float(request.form["prep_time"]),
        "Courier_Experience_yrs": float(request.form["experience"])
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # Predict
    prediction = model.predict(input_df)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Delivery Time: {round(prediction, 2)} minutes"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
