# natalies-little-helper

#### Data

Here's the link to the Twitter US Airline Sentiment Dataset on kaggle:
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

I put it in a directory "data/" that is in the gitignore, but it should have this structure

natalies-little-helper/
    data/
        Tweets.csv
    notebooks/
        preprocessing.ipynb
        sentiment.ipynb



#### Environment

Using conda environments 

To create a conda environment from a file:

    $ conda env export > environment.yml

To generate the environment.yml file:

    $ conda env create -f environment.yml

To activate this environment, use

    $ conda activate test-nlh

To deactivate an active environment, use

    $ conda deactivate
