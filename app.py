from flask import Flask, render_landing, request, jsonify
import numpy as np # type: ignore
import pickle

app = Flask(__name__)

# Load the trained model (you'll need to train and save it as a .pkl file)
model = pickle.load(open('house_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_landing('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        data['latitude'],
        data['longitude'],
        data['housing_median_age'],
        data['total_rooms'],
        data['total_bedrooms'],
        data['population'],
        data['households'],
        data['median_income']
    ]
    
    prediction = model.predict([features])
    return jsonify({'predicted_price': prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
