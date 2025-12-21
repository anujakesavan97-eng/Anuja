from flask import Flask, request
import pickle

app = Flask(__name__)

@app.route('/Predictions', methods = ['POST'])
def preds():
    
    sl = float(request.form['sepal_length'])
    sw = float(request.form['sepal_width'])
    pl = float(request.form['petal_length'])
    pw = float(request.form['petal_width'])

    data = [[sl,sw, pl,pw]]

    with open('model_iris.pkl', 'rb') as pred_model:
        main_model = pickle.load(pred_model)

    final_predictions = main_model.predict(data)

    return str(final_predictions[0])

app.run()     
