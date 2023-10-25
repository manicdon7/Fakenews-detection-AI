import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Load the datasets
true_data = pd.read_csv('True.csv')
false_data = pd.read_csv('Fake.csv')

# Add a 'label' column to indicate real news (0) and fake news (1)
true_data['label'] = 0
false_data['label'] = 1

df = pd.concat([true_data, false_data])

df = df.sample(frac=1).reset_index(drop=True)

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    # Add your text preprocessing steps here (e.g., lowercasing, removing stopwords)
    words = text.split()
    words = [word.lower() for word in words if word.lower() not in stop_words]
    return ' '.join(words)

df['text'] = df['text'].apply(preprocess_text)

# Sentiment analysis using VADER SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']  # Compound score combines positive and negative sentiment

df['sentiment'] = df['text'].apply(analyze_sentiment)

df['text_length'] = df['text'].apply(lambda x: len(x))

X = df[['text', 'sentiment', 'text_length']] 
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['text'])
X_test_tfidf = tfidf_vectorizer.transform(X_test['text'])

X_train_features = pd.concat([pd.DataFrame(X_train_tfidf.toarray()), X_train[['sentiment', 'text_length']].reset_index(drop=True)], axis=1)
X_test_features = pd.concat([pd.DataFrame(X_test_tfidf.toarray()), X_test[['sentiment', 'text_length']].reset_index(drop=True)])

clf = RandomForestClassifier()
clf.fit(X_train_features, y_train)

y_pred = clf.predict(X_test_features)
print(classification_report(y_test, y_pred))
