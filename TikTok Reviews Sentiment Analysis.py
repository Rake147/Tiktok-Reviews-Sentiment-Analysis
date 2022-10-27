#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string
import re
nltk.download('stopwords')
stemmer=nltk.SnowballStemmer("english")


# In[2]:


data=pd.read_csv('C:/Users/Rakesh/Datasets/tiktok_google_play_reviews.csv')


# In[3]:


data.head()


# In[4]:


data=data[['content','score']]


# In[5]:


data.head()


# In[6]:


data.isnull().sum()


# In[7]:


data=data.dropna()


# In[8]:


stopword=set(stopwords.words('english'))


# In[9]:


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text


# In[10]:


data['content']=data['content'].apply(clean)


# In[11]:


ratings=data['score'].value_counts()
numbers=ratings.index
quantity=ratings.values
import plotly.express as px
figure=px.pie(data,values=quantity,names=numbers,hole=0.5)
figure.show()


# In[12]:


text=''.join(i for i in data.content)
stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords,background_color='white').generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[13]:


data.head()


# In[15]:


nltk.download('vader_lexicon')
sentiments=SentimentIntensityAnalyzer()
data['Positive']=[sentiments.polarity_scores(i)['pos'] for i in data['content']]
data['Negative']=[sentiments.polarity_scores(i)['neg'] for i in data['content']]
data['Neutral']=[sentiments.polarity_scores(i)['neu'] for i in data['content']]
data=data[['content','Positive','Negative','Neutral']]
data.head()


# In[16]:


positive=''.join([i for i in data['content'][data['Positive']>data['Negative']]])
stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords,background_color='white').generate(positive)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')


# In[17]:


negative =' '.join([i for i in data['content'][data['Negative'] > data["Positive"]]])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(positive)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# In[ ]:




