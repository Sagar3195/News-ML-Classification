from flask import *
import numpy as np
import joblib

clf = joblib.load('news_model.pkl')
vectorizer = joblib.load('transform.pkl')
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        data = [text]
        vect = vectorizer.transform(data).toarray()
        prediction = clf.predict(vect)

    return render_template('result.html', prediction_text = prediction)
if __name__ == '__main__':
    app.run(debug= True)

