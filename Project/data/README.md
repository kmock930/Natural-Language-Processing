# Datasets
This project intends to collect necessary data (for modeling) from a combination of **web scraping** which adheres to ethical considerations as well as **publicly-available datasets**.

## [**Reddit SuicideWatch Posts**](./Reddit_SuicideWatch/)
**Web Scraping**: By launching a web request to Reddit, we load a list of 100 records of posts from `curl https://reddit.com/r/SuicideWatch/new.json?limit=100` which is in the subreddit (i.e., category) named "SuicideWatch". Note that this request could only fetch 100 records at a time. 
Raw data are loaded into `reddit_suicidewatch.json`.

## [**Social Media Sentiments Analysis Dataset**](./Social_Media_Sentiments_Analysis_Dataset/)
Dataset is downloaded into a .csv format from <url>https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset?resource=download</url>.

## [**Twitter Suicidal Data**](./Twitter_Suicidal_Data/)
Dataset is downloaded into a .csv format from <url>https://www.kaggle.com/datasets/hosammhmdali/twitter-suicidal-data</url>.

## [**Depression Tweets**](./Depression_Tweets/)
Dataset is downloaded into a .json format from <url>https://www.kaggle.com/datasets/senapatirajesh/depression-tweets</url>.

# [**Data Processing**](data_processing.py)
Original text is normalized before classification: 
* Removing emojis
* Removing symbols - such as hashtag # sign, the @ symbol, and URLs.
* Removing punctuations
* Converting the entire text to lowercase
* Lemmatization - to replace abstract words with its base form
* Word Tokenization
* Removing Stopwords

# Data Splitting
The training set uses data from the following datasets:
* Reddit
* Social Media Sentiment Analysis

The validation set uses data from the following dataset:
* Twitter

The test set uses data from the following dataset:
* Depression Tweet

# Labeling:
We consider a binary classification problem with the following labels and interpretation:
* 0: non-suicidal
* 1: suicidal