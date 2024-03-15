import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    bedrooms = request.form.get("bedrooms", type=int)
    area = request.form.get("area", type=int)
    bathrooms = request.form.get("bathrooms", type=int)
    stories = request.form.get("stories", type=int)
    parking = request.form.get("parking", type=int)
    mainroad = request.form.get("mainroad", type=int)
    guestroom = request.form.get("guestroom", type=int)
    basement = request.form.get("basement", type=int)
    hotwaterheating = request.form.get("hotwaterheating", type=int)
    airconditioning = request.form.get("airconditioning" , type=int)
    prefarea = request.form.get("prefarea", type=int)
    furnishingstatus = request.form.get("furnishingstatus", type=int)

    features = np.array([[area, bedrooms, bathrooms, stories, parking, mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus]])
    
    prediction = model.predict(features) 
    
    result = prediction[0]

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)