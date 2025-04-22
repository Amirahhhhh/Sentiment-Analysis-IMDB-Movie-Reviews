# 🎬 Sentiment-Analysis-IMDB-Movie-Reviews
Sentiment analysis on IMDB movie reviews using classical machine learning and deep learning models, with evaluation, tuning, and visualisation techniques.

## 1.	📌 PROJECT OVERVIEW
This project applies machine learning and deep learning techniques to perform sentiment classification on the IMDB movie review dataset, which contains 50,000 labelled movie reviews. The reviews are categorized as either positive or negative, enabling the development of a binary classification model. This task falls under Natural Language Processing (NLP) and text analytics, making it an ideal case study for comparing traditional ML and DL approaches.
The dataset is split into 25,000 reviews for training and 25,000 for testing, offering a robust benchmark for experimentation with various models and techniques.

## 2.	📊 ABOUT THE DATASET
The dataset consist of:

|     Feature names                  |     Description                                        |
|------------------------------------|--------------------------------------------------------|
|     Reviews                        |     Text of movie review                               |
|     Sentiment                      |     Lable: positive or negative                        |

Dataset can be access through the link : https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## 3.	🎯 OBJECTIVE
•  To predict the sentiment (positive or negative) of a given movie review uwing machine learning and deep learning  
•  To compare traditional ML algorithms and advanced DL models like LSTM and CNN. 
•  To understand the impact of hyperparameter tuning and text representation techniques (TF-IDF vs. embeddings).

## 4.	🔎 METHODOLOGY
a)	Data Preprocessing
•  Cleaned text: removed HTML tags, punctuation, numbers, and stop words
•  Converted to lowercase and performed lemmatization
•  Balanced classes across training and testing sets

b)	Exploratory Data Analysis (EDA)
•  Word cloud visualization of frequent words in positive vs. negative reviews
•  Class distribution, word count histograms
•  Sentiment polarity trend across the dataset

c)	Data Transformation
•  Tokenization and padding for deep learning models
•  TF-IDF vectorization for traditional ML models
•  Label encoding for binary sentiment output

d)	Model Building & Evaluation
i)	Machine Learning Models
•	Naïve Bayes
•	Stochastic Gradient Descent (SGD)
•	XGBoost

ii)	Deep Learning Models
•	LSTM with GloVe embeddings
•	CNN with 1D convolutional layers

iii)	Evaluation Metrics
•	Accuracy, Precision, Recall, F1-Score
•	ROC Curve and AUC Score
•	Plotted Before vs After Hyperparameter Tuning results

e)	Hyperparameter Tuning 
•  Used GridSearchCV for traditional models (SGD, XGBoost)
•  Applied custom Keras Tuner for LSTM & CNN
•  Visualized performance differences pre- and post-tuning in subplots

## 5.	⚙️ RESULT 

|        Model          |     Accuracy (Before tuning)     |     Accuracy (After tuning)     |
|-----------------------|----------------------------------|---------------------------------|
|     Naive Bayes       |             0.8634               |             0.8634              |
|     SGD               |             0.8474               |             0.8973              |
|     XGBoost           |             0.8564               |             0.8278              |
|     LSTM              |             0.6920               |             0.8668              |
|     CNN               |             0.8427               |             0.8537              |

✅ SGD and LSTM showed the highest improvement after tuning
✅ Best performing model overall: SGD (Traditional) and LSTM (Deep Learning)

## 6.	📝 CONCLUSION
•  The project demonstrated traditional ML models (SGD) can be highly competitive for text classification. 
•  Deep learning models like LSTM improved significantly after tuning, highlighting the importance of architecture and optimization. 
•  TF-IDF was effective for ML models , while GloVe embeddings enhanced DL models. 





