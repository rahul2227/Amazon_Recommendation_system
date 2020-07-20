#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame 
import nltk

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from scipy.spatial.distance import cosine
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.neighbors import KNeighborsClassifier as knn

import re
import string
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import mean_squared_error


#Reading dataset
df = pd.read_csv("Reviews.csv")

#Analyzing
print(df.columns)
print(df.shape)

count = df.groupby("ProductId", as_index=False).count()
mean = df.groupby("ProductId", as_index=False).mean()


dfMerged = pd.merge(df, count, how='right', on=['ProductId'])
dfMerged


dfMerged["totalReviewers"] = dfMerged["UserId_y"]
dfMerged["overallScore"] = dfMerged["Score_x"]
dfMerged["summaryReview"] = dfMerged["Summary_x"]

dfNew = dfMerged[['ProductId','summaryReview','overallScore',"totalReviewers"]]

dfMerged = dfMerged.sort_values(by='totalReviewers', ascending=False)
dfCount = dfMerged[dfMerged.totalReviewers >= 100]
dfCount

dfProductReview = df.groupby("ProductId", as_index=False).mean()
ProductReviewSummary = dfCount.groupby("ProductId")["summaryReview"].apply(list)
ProductReviewSummary = pd.DataFrame(ProductReviewSummary)
ProductReviewSummary.to_csv("ProductReview.csv")

df3 = pd.read_csv("ProductSummary.csv")
df3 = pd.merge(df3, dfProductReview, on="ProductId", how='inner')

df3 = df3[['ProductId','summaryReview','Score']]


regEx = re.compile('[^a-z]+')
def cleanReviews(Text):
    Text = Text.lower()
    Text = regEx.sub(' ', Text).strip()
    return Text

df3["summaryClean"] = df3["summaryReview"].apply(cleanReviews)
df3 = df3.drop_duplicates(['Score'], keep='last')
df3 = df3.reset_index()

reviews = df3["summaryClean"] 
countVector = CountVectorizer(max_features = 300, stop_words='english') 
transformedReviews = countVector.fit_transform(reviews)

dfReviews = DataFrame(transformedReviews.A, columns=countVector.get_feature_names())
dfReviews = dfReviews.astype(int)

dfReviews.to_csv("dfReviews.csv")

X = np.array(dfReviews)
# create train and test
tpercent = 0.9
tsize = int(np.floor(tpercent * len(dfReviews)))
dfReviews_train = X[:tsize]
dfReviews_test = X[tsize:]
#len of train and test
lentrain = len(dfReviews_train)
lentest = len(dfReviews_test)

print(lentrain)
print(lentest)
neighbor = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(dfReviews_train)
distances, indices = neighbor.kneighbors(dfReviews_train)

for i in range(lentest):
    a = neighbor.kneighbors([dfReviews_test[i]])
    related_product_list = a[1]

    first_related_product = [item[0] for item in related_product_list]
    first_related_product = str(first_related_product).strip('[]')
    first_related_product = int(first_related_product)
    second_related_product = [item[1] for item in related_product_list]
    second_related_product = str(second_related_product).strip('[]')
    second_related_product = int(second_related_product)
    
    print ("Based on product reviews, for ", df3["ProductId"][lentrain + i] ," average rating is ",df3["Score"][lentrain + i])
    print ("The first similar product is ", df3["ProductId"][first_related_product] ," average rating is ",df3["Score"][first_related_product])
    print ("The second similar product is ", df3["ProductId"][second_related_product] ," average rating is ",df3["Score"][second_related_product])
    print ("-----------------------------------------------------------")
df5_train_target = df3["overallScore"][:lentrain]
df5_test_target = df3["overallScore"][lentrain:lentrain+lentest]
df5_train_target = df5_train_target.astype(int)
df5_test_target = df5_test_target.astype(int)


n_neighbors = 3
knnclf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
knnclf.fit(dfReviews_train, df5_train_target)
knnpreds_test = knnclf.predict(dfReviews_test)

print(classification_report(df5_test_target, knnpreds_test)) 
print (accuracy_score(df5_test_target, knnpreds_test))

print(mean_squared_error(df5_test_target, knnpreds_test))

knn.score(df5_test_target, knnpreds_test)
acc = accuracy_score(df5_test_target, knnpreds_test)

