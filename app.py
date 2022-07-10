from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle as pickle
model = pickle.load(open('deploy_model1.pkl', 'rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    GRE = int(request.form.get('G'))
    TOEFL = int(request.form.get('T'))
    Uni = int(request.form.get('U'))
    lr = pickle.load(open('deploy_model1.pkl', 'rb'))
    result = lr.predict(np.array([GRE,TOEFL,Uni]).reshape(1, 3))
    print(result)
    result =np.array2string(result)
    result=  result.replace("[",'')
    return result.replace("]", '')

if __name__ != "__name__":
    app.run(host='0.0.0.0', port=8080)
s