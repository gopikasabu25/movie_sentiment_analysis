import pickle
import gradio as gr
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Load your pretrained files (same as your notebook)
model = pickle.load(open('sentiment_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove tags
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def predict_sentiment(text):
    cleaned_text = clean_text(text)
    vector = tfidf.transform([cleaned_text]).toarray()
    prediction = model.predict(vector)[0]
    return "Positive" if prediction == 1 else "Negative"  # Convert to label

# Gradio Interface
gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Movie Review"),
    outputs="label",
    title="🎬 Movie Review Sentiment Analysis",
    examples=["This film was fantastic!", "Terrible plot."]
).launch(server_name="0.0.0.0", server_port=7860)
