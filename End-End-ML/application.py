import pickle
from flask import Flask,request, jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
application = Flask(__name__)
app = application

#import ridge regressor and scaler
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract features from form
            Temperature = float(request.form['Temperature'])
            RH = float(request.form['RH'])
            Ws = float(request.form['Ws'])
            Rain = float(request.form['Rain'])      
            FFMC = float(request.form['FFMC'])      
            DMC = float(request.form['DMC'])        
            ISI = float(request.form['ISI']) 
            Classes = float(request.form['Classes'])    
            Region = float(request.form['Region'])      


            new_data = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result = ridge_model.predict(new_data)
            #
            return render_template('home.html', result = result[0])
        except Exception as e:
            return f" enter float values only. Error: {e}"
       
    else:
        return render_template('home.html')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)