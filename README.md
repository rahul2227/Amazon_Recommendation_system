# Amazon_Recommendation_System

# Project Aimed:
A recommendation system is an extensive class of web applications that involves predicting the user responses to the options.
Often termed as Recommender Systems, they are simple algorithms which aim to provide the most relevant and accurate items to the user by filtering useful stuff from of a huge pool of information base. Recommendation engines discovers data patterns in the data set by learning consumers choices and produces the outcomes that co-relates to their needs and interests.
The objective of this task is to recommend the products based on the similar reviews.The model recommends two more products based on the similar reviews i.e, If a person has purchased coffee of A type, two more types of coffees are recommended to him/her.


# Pre-Requisites:
Python3 
Some of the Python Libraries
1.  numpy           "https://numpy.org/"
2.  pandas          "https://pypi.org/project/pandas/"
3.  seaborn         "https://pypi.org/project/seaborn/"
4.  sklearn         "https://pypi.org/project/sklearn/"
5.  matplotlib      "https://pypi.org/project/matplotlib/"
6.  nltk            "https://pypi.org/project/nltk/"
7.  re              "https://pypi.org/project/regex/"

# Installation:
There are some steps for installation
1.  Install Python3.                        "https://www.python.org/downloads/"
2.  Download and Install Anaconda Toolkit   "https://www.anaconda.com/products/individual"  
3.  Install Spyder.                         
4.  Install all the libraries above mentioned by using "pip install library_name".
5.  Download the project. Run it in your system.

# Dataset:
This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.
The dataset initially had 8 columns i.e, id,username,productname,profilename,rating,summary,text,time when reviewed. Data is preprocssed first and all the irrelevant details such as username,time,profilename are removed, only the productname and summary(review) is kept in dataset.

Note: The dataset is downloaded from Kaggle: "https://www.kaggle.com/snap/amazon-fine-food-reviews".

I used the one dataset Reviews for training and testing my model.

I am getting 80% accuracy in this project.

