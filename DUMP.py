import joblib
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tkinter import messagebox

# Load the dumped models
classifier = joblib.load('naive_bayes_model.joblib')
cv = joblib.load('count_vectorizer.joblib')

# Function to Clean the texts
def preprocess_text(text):
    log = re.sub('[^a-zA-Z0-9]', ' ', text)
    log = log.lower()
    log = log.split()
    ps = PorterStemmer()
    log = [ps.stem(word) for word in log if not word in set(stopwords.words('english'))]
    log = ' '.join(log)
    return log

# New log insert
new_text = input("Enter the traffic info: ")
preprocessed_text = preprocess_text(new_text)

# Use the loaded CountVectorizer object to transform the preprocessed text into a bag of words representation
new_text_bow = cv.transform([preprocessed_text]).toarray()

# Use the loaded GaussianNB object to make predictions on the new log data
predicted_class = classifier.predict(new_text_bow)

# Display a pop-up message based on the predicted class
if predicted_class:
    messagebox.showinfo("Anomaly Detected", "Anomaly detected in the traffic!")
else:
    messagebox.showinfo("No Anomaly", "No anomaly detected in the traffic.")
    
# Non Malicious Packet Info
# Adjacency Message (Syn)
# 19717  >  6068 [ACK] Seq=1285 Ack=785 Win=3344 Len=0

# Malicious Packet Info
# 80  >  51451 [ACK] Seq=214260 Ack=712 Win=16128 Len=1460 [TCP segment of a reassembled PDU]
# 80  >  49234 [PSH, ACK] Seq=500760 Ack=325 Win=64240 Len=1448 [TCP segment of a reassembled PDU]
