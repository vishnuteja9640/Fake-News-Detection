#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Fake News Prediction Using Machine Learning With Python
#Here we are dealing with textual data instead of Numerical data

''' First we collected the data from Kaggle which is the labelled data, and now we will preprocess this data and then split
 this dataset into train and test. Now after developing a model with the train data we will test our model accuracy'''


# In[3]:


# We use Logistic Regression model here because this is a binary classification project which classifies the data 
#into two (1-Fake News) and (0-Real News)


# In[4]:


#Now importing the required libraries
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords     # Stopwords are the words which adds no useful to the paragraph
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[5]:


import nltk                     # Downloading the stopwords
nltk.download('stopwords')


# In[6]:


print(stopwords.words('English'))   # These are the stopwords


# In[7]:


# Now importing the dataset
dataframe = pd.read_csv('train.csv')
dataframe.shape


# In[8]:


dataframe.head()


# In[9]:


# Checking whether they are any missing values present inbetween the data, checking for every column
dataframe.isnull().sum() 


# In[10]:


# Replacing the missing values with null strings
dataframe = dataframe.fillna('')


# In[11]:


# Now we have to choose any two columns to perform datapreprocessing. We selected title and author name for now and creating a new column
dataframe['New_column'] = dataframe['author'] + ' ' + dataframe['title']
dataframe.head()
# NOw we use this data present in the New_column and label for prediction


# In[12]:


X = dataframe.drop(columns ='label',axis=1) # THis will remove the label column and store the remaining data into X
Y = dataframe['label'] # This will store the label column in Y variable


# In[13]:


# Now will perform Stemming which is shorting the words to it's root 
# After performing stemming, the next step is Vectorizing where we convert the root words to Feature Vectors which are numerical data
port_stem = PorterStemmer()
def FuncStemming(input_data):
    stemmed_data = re.sub('[^a-zA-Z]',' ',input_data)
    stemmed_data = stemmed_data.lower()
    stemmed_data = stemmed_data.split()
    stemmed_data = [port_stem.stem(word) for word in stemmed_data if not word in stopwords.words('english')]
    stemmed_data = ' '.join(stemmed_data)
    return stemmed_data



# In[14]:


# The input to the above described function is the dataframe['New_column'] which we have previously created
dataframe['New_column'] = dataframe['New_column'].apply(FuncStemming)


# In[15]:


X = dataframe['New_column'].values   # Only the last column which is the output of the function
Y = dataframe['label'].values        # Here the label column is stored here 1: Fake and 0: Real


# In[16]:


print(X)
print(Y)


# In[17]:


# Now we have to convert this root words in the X(Textual data) into corresponding feature vectors which are numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)


# In[18]:


# If we fed a textual data to our machine learning model it will have problem understanding the textual data. NOw since we 
# converted into numerical data the model now can easily understand 
print(X)


# In[19]:


# Now we have to split the data into training and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
# The entire dataset is split into 80% training and 20% for testing. The label for X_train is stored in Y_train.
# And the label for X_test is stored in Y_test.


# In[20]:


# Now training the model
model = LogisticRegression()
model.fit(X_train,Y_train)


# In[22]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print(training_data_accuracy)
# The accuracy score of the training data


# In[25]:


X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print(testing_data_accuracy)
# The accuracy score of the testing data


# In[27]:


# Checking for a different input
X_new = X_test[47]
prediction = model.predict(X_new)
print(prediction)

if prediction == 1:
    print("The news is fake")
else:
    print("The news is real")


# In[28]:


# Now checking with the actual Label
print(Y_test[47])


# In[29]:


X_new = X_test[1047]
prediction = model.predict(X_new)
print(prediction)

if prediction == 1:
    print("The news is fake")
else:
    print("The news is real")


# In[30]:


print(Y_test[1047])


# In[31]:


X_new = X_test[997]
prediction = model.predict(X_new)
print(prediction)

if prediction == 1:
    print("The news is fake")
else:
    print("The news is real")


# In[32]:


print(Y_test[997])


# In[36]:


X_new = X_test[2220]
prediction = model.predict(X_new)
print(prediction)

if prediction == 1:
    print("The news is fake")
else:
    print("The news is real")


# In[37]:


print(Y_test[2220])


# In[38]:


X_new = X_test[1500]
prediction = model.predict(X_new)
print(prediction)

if prediction == 1:
    print("The news is fake")
else:
    print("The news is real")


# In[39]:


print(Y_test[1500])

