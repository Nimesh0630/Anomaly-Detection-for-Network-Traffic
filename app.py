import joblib
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

classifier = joblib.load('naive_bayes_model.joblib')
cv = joblib.load('count_vectorizer.joblib')

from flask import Flask, request,flash, render_template, url_for, redirect

app = Flask(__name__)
app.secret_key = 'supersecretkey'

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        traffic_info = request.form.get('traffic-info')
        messege = get_predictions(traffic_info)
        if messege :
            flash(messege)
        return redirect(url_for('home'))
    return render_template('index.html')


if __name__ == "__main__":
    app.run()


def preprocess_text(text):
    log = re.sub('[^a-zA-Z0-9]', ' ', text)
    log = log.lower()
    log = log.split()
    ps = PorterStemmer()
    log = [ps.stem(word) for word in log if not word in set(stopwords.words('english'))]
    log = ' '.join(log)
    return log

def get_predictions(text):
    if text:
        preprocessed_text = preprocess_text(text)
        new_text_bow = cv.transform([preprocessed_text]).toarray()
        predict = classifier.predict(new_text_bow)
        if predict == 1:
            return "Anomaly detected in the traffic!"
        else:
            return "No anomaly detected in the traffic."
    return ""
