from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
import sklearn

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index4.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        CRIM = float(request.form['CRM'])
        ZN=float(request.form['ZN'])
        INDUS=float(request.form['IND'])
        CHAS=float(request.form['CHA'])
        NOX=float(request.form['NOX'])
        RM=float(request.form['RMS'])
        AGE=float(request.form['AGE'])
        DIS=float(request.form['DIS'])
        RAD=float(request.form['RAD'])
        TAX=float(request.form['TAX'])
        PTRATIO=float(request.form['PTR'])
        B=float(request.form['BLK'])
        LSTAT=float(request.form['LST'])
        prediction=model.predict([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]])
        output=prediction[0]
        if output<0:
            return render_template('index4.html', Predicted_price="Sorry you cannot sell this House")
        else:
            return render_template('index4.html', Predicted_price="You Can Sell The House at {}".format(output))
    else:
        return render_template('index1.html')

if __name__=="__main__":
    app.run(debug=True)
