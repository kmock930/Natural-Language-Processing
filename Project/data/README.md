# Datasets
This project intends to collect necessary data (for modeling) from a combination of **web scraping** which adheres to ethical considerations as well as **publicly-available datasets**.

## [Reddit SuicideWatch Posts](./Reddit_SuicideWatch/)
By launching a web request to Reddit, we load a list of 1000000000000 records of posts from `curl https://reddit.com/r/SuicideWatch/new.json?limit=1000000000000` which is in the subreddit (i.e., category) named "SuicideWatch". 
Raw data are loaded into `reddit_suicidewatch.json`.