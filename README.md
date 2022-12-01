# natalies-little-helper

11-2-22 Used PyTorch for FNN for sentiment saved one of the models I got with ~78% accuracy, and Hugging Face Transformer models DistilBERT performed the best saved the model in ./models/tuned_distilbert_sentiment had to exclude it before pushing becuase file too large, tried some stuff for the negative intent classification but not much luck the simple models have performed the best so far

## Data

Here's the link to the Twitter US Airline Sentiment Dataset on kaggle:
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment



I put it in a directory "data/" that is in the gitignore, but it should have this structure

natalies-little-helper/
    data/
        Tweets.csv
    notebooks/
        preprocessing.ipynb
        sentiment.ipynb
    util/
        helpers.py


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
