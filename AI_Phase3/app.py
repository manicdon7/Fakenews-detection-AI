from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk

app = Flask(__name__)

# Load the 'True.csv' and 'False.csv' datasets
true_data = pd.read_csv('True.csv')
false_data = pd.read_csv('Fake.csv')

# Add a 'label' column to indicate real news (0) and fake news (1)
true_data['label'] = 0
false_data['label'] = 1

# Concatenate the two datasets
df = pd.concat([true_data, false_data])

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# Data cleaning and preprocessing (e.g., removing stopwords, punctuation, and lowercasing)
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    # Add your text preprocessing steps here
    # For example, lowercasing and removing stopwords
    words = text.split()
    words = [word.lower() for word in words if word.lower() not in stop_words]
    return ' '.join(words)

df['text'] = df['text'].apply(preprocess_text)

# Load the trained TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df['text'])
y = df['label']

# Load the trained classifier (e.g., Naive Bayes)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_tfidf, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['text']
        news_text = preprocess_text(news_text)
        news_tfidf = tfidf_vectorizer.transform([news_text])
        prediction = clf.predict(news_tfidf)[0]

        if prediction == 0:
            result = "Real News"
        else:
            result = "Fake News"

        return render_template('index.html', prediction=result, news_text=news_text)

if __name__ == '__main__':
    app.run(debug=True)
