# Natalie's Little Helper

This project is focused on creating an automated comment analysis and response tool for professional social media accounts, with the case study being on Twitter interactions between airlines and their customers. The tool aims to help businesses respond to the massive influx of comments and reviews on social media, especially during peak travel times and severe travel disruptions. By analyzing the sentiment and intent expressed in tweets, the tool hopes to provide relevant and appropriate responses to customer complaints, requests, and dissatisfaction.

## Method

The method for this project involved the use of the Twitter Airline data set from Kaggle as the corpus for experimentation. The features and labels of interest in the data set included tweets from users directed at an airline, the sentiment of the tweet, the negative reason for a negative sentiment tweet, and the confidence scores for sentiment and negative reason labels. Data cleaning was performed by resolving contractions, normalizing whitespace, lowercasing, and removing URLs, HTML tags, and non-alphabetical characters. Pre-processing was conducted using the NLTK library for stop-word removal and lemmatization. Feature extraction was assisted by TF-IDF vectorization, average word embedding, and attention mask embedding.

The sentiment classification was performed using the DistilBERT model, which was fine-tuned on optimal hyperparameters. The model was used to handle positive/neutral and negative sentiments separately. For negative intent classification, an AWD-LSTM classifier was trained using the ULMFiT approach. A Fast.AI Language Model (LM) was fine-tuned using the Twitter corpora and a negative sentiment classifier was built for negative sentiment tweets with annotated intent. The ULMFiT LM and AWD-LSTM classifier were trained cyclically with an annealing learning rate.

Supervised K-means clustering was performed on negative sentiment tweets with labeled intent and unsupervised clustering was performed on positive and neutral sentiment samples. The elbow method was applied to find the optimal K-values in the unsupervised task. The k-means algorithm was applied to various feature embedding representations, including TF-IDF, TF-IDF with SVD, Word2Vec, and Sense2Vec.

For response generation, a rule-based generator was used for negative sentiment tweets and a language model was used to analyze sentiment and intent and create appropriate and unique responses. The Grounded Open Dialogue Language (GODEL) model was used for response generation for negative tweets. The intent and cleaned tweet text were used as input to provide appropriate response generation with sensitivity context.

## Data

Here's the link to the Twitter US Airline Sentiment Dataset on kaggle:
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

## Results

The results of the project showed that the best-performing simple model for sentiment analysis used a Linear SVC. For complex models, the DistilBERT model had the highest performance with an improvement of 5% for binary classification compared to the baseline models. The ULMFiT model showed the best performance for intent classification with a 5% improvement over the baseline models. All classifiers showed a 10% improvement when grouping the intents into the four broader classes. However, oversampling and balancing categories decreased performance by 1%.

For the clustering part of the project, the best results were achieved with TF-IDF with strict pre-processing and SVD-reduction to 30 components. The clusters were uneven in size and many tweets had positive or neutral sentiment annotations. The clustering method was seen as having a purpose in the overall pipeline but intent was not fully captured by the clusters.

The GODEL model produced somewhat relevant sentences, but it could not clearly capture the perspective of a brand representative. The Text-Davinci model produced natural, relevant responses for positive sentiment tweets but for neutral tweets, which were primarily queries, the response was not grounded in truth. This aspect of the project is left open for future work to incorporate a knowledge base to reflect company policy.

## Environment

### Python Virtual Environment

Using python 3.10.2

May need to install pytorch separately based on cuda version your system requires

To create a new virtual environment use these commands:
    
    $ python3 -m venv <name-of-env>
    $ ./<name-of-env>/Scripts/activate
    $ pip install -r requirements.txt

To deactivate use this command:
    
    $ deactivate

Note for package ray[tune] needed to pip install using this command if using python 3.9 or newer:

    $ pip install -U "ray[default] @ LINK_TO_WHEEL.whl"

where LINK_TO_WHEEL.whl is dependent on your python version and Operating System and can be found on <a href="https://docs.ray.io/en/latest/ray-overview/installation.html#daily-releases-nightlies">this</a> site

For example for python3.10 on Windows you would install it like this:

    $ pip install -U "ray[default] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-win_amd64.whl"

Generated a requirements.txt file using this command:
    
    $ pip freeze > requirements.txt

(Ended up having a bunch of issues with conda because of huggingface transformers package)

### For using conda environments 

To create a conda environment from a file:
    
    $ conda env create -f environment.yml

To generate the environment.yml file:
    
    $ conda env export > environment.yml

To activate this environment, use
    
    $ conda activate nlh

To deactivate an active environment, use
    
    $ conda deactivate
