# 🎬 Movie Review Sentiment Analysis

This project builds a **Sentiment Analysis** system for movie reviews using a **Logistic Regression** model and **TF-IDF** vectorization. It classifies reviews as either **positive** or **negative**, providing a simple yet effective natural language processing (NLP) pipeline.

## 📁 Files Included

| File Name                          | Description                                     |
|-----------------------------------|-------------------------------------------------|
| `Movie_Review_Sentimental_Analysis.ipynb` | Main notebook containing data loading, preprocessing, training, and prediction steps. |
| `sentiment_model.pkl`             | Pre-trained logistic regression model.         |
| `tfidf_vectorizer.pkl`            | Pre-trained TF-IDF vectorizer used to convert text into numerical features. |


## For Live Demo

Try the sentiment analyzer here: 
     https://movie-sentiment-analyzer-wixf.onrender.com/

## Sample Output :

Here is a screenshot showing the model predicting the sentiment of a movie review:

     ![Screenshot](https://github.com/gopikasabu25/movie_sentiment_analysis/blob/main/sample%20review.png)

      ![Screenshot](https://github.com/gopikasabu25/movie_sentiment_analysis/blob/main/positive%20review%20sample.png)

      ![Screenshot](https://github.com/gopikasabu25/movie_sentiment_analysis/blob/main/neagtive%20review%20sample.png)
## 📊 Dataset

This project uses the open-source dataset from **Kaggle**:

**📂 [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)**
- Creator: [Rounak Banik](https://www.kaggle.com/rounakbanik)
- License: MIT (https://opensource.org/licenses/MIT) *(For the dataset only)*  
- Description: A large dataset of metadata for over 45,000 movies from TMDB, including user ratings and reviews.
- ⚠️ Note: The dataset itself is **not included** in this repository due to licensing — please download it manually from the link above.

## 🔧 Model Info

- **Vectorization**: `TfidfVectorizer(max_features=5000)`
- **Classifier**: `LogisticRegression(max_iter=1000)`
- **Frameworks Used**: Scikit-learn, pandas, numpy

## 📄 License

The dataset used in this project is under the MIT License by Kaggle user [rounakbanik](https://www.kaggle.com/rounakbanik).


