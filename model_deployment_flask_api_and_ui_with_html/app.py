from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# Load your model
# model = pickle.load(open("models/credit_random_model.pkl", "rb"))
model = pickle.load(open("models/best_model_catboost.pkl", "rb"))
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json

        # Prepare features for prediction
        features = [
            data['person_age'],
            data['person_income'],
            data['person_emp_length'],
            data['loan_amnt'],
            data['loan_percent_income'],
            data['cb_person_default_on_file'],
            data['person_home_ownership_OTHER'],
            data['person_home_ownership_OWN'],
            data['person_home_ownership_RENT'],
            data['loan_intent_DEBTCONSOLIDATION'],
            data['loan_intent_EDUCATION'],
            data['loan_intent_HOMEIMPROVEMENT'],
            data['loan_intent_MEDICAL'],
            data['loan_intent_PERSONAL'],
            data['loan_intent_VENTURE'],
            data['loan_grade_A'],
            data['loan_grade_B'],
            data['loan_grade_C'],
            data['loan_grade_D'],
            data['loan_grade_E'],
            data['loan_grade_F'],
            data['loan_grade_G']
        ]

        # Make prediction
        prediction = model.predict([features])

        # Return prediction as JSON
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)