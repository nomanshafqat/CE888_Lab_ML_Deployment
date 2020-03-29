from flask import Flask, request, render_template
import os
import pickle

print(os.getcwd())
path = os.getcwd()



with open('Models/mlp.pkl', 'rb') as f:
    mlp = pickle.load(f)


def get_predictions(feature_map_array, req_model):

    vals = [feature_map_array]
    if req_model == 'MLP':
        #print(req_model)
        return mlp.predict(vals)[0]

    else:
        return "Cannot Predict"


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('home.html')

@app.route('/', methods=['POST', 'GET'])
def my_form_post():



    age = request.form['age']
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = request.form['trestbps']

    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exang = request.form['exang']

    oldpeak = request.form['oldpeak']
    slopethe = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']
    req_model = request.form['req_model']

    mylist = [age,sex,cp,trestbps,chol, fbs, restecg, thalach,exang,oldpeak,slopethe,ca,thal]
    mylist = [float(i) for i in mylist]

    target = get_predictions(mylist, req_model)

    if target==1:
        sale_making = 'Patient is likely to have heart disease'
    else:
        sale_making = 'Patient is unlikely to have heart disease'

    return render_template('home.html', target = target, sale_making = sale_making)


if __name__ == "__main__":
    app.run(debug=True)
