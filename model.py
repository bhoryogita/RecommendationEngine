import re # for regular expressions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle

nltk.download('punkt')
nltk.download('stopwords')

words_removed= ['collect', 'part', 'promote', 'review', 'really', 'product']

stemmer = PorterStemmer()

spell = SpellChecker()

with open('logisticReg_final_model.pkl', 'rb') as file:
    model = pickle.load(file)
    
with open('tf_idf_vectorizer_new.pkl', 'rb') as file:
    tf_idf = pickle.load(file)
    

class get_sentiment():
    
    def __init__(self):
        
        return None
    
    def clean_review(self, review_text):
        review_text = re.sub(r"http\S+|www.\S+", "", review_text)                        
        review_text = re.sub("[^a-zA-Z]", " ", review_text)                       
        review_text = str(review_text).lower()                                    
        review_text = word_tokenize(review_text)                                  
        review_text = [i for i in review_text if len(i)>2]                        
        review_text = [i for i in review_text if i not in stopwords.words('english')]  
        review_text = [i for i in review_text if i not in words_removed]
        review_text = [stemmer.stem(i) for i in review_text]                           
        review_text = [spell.correction(i) for i in review_text]                       
        review_text = ' '.join(review_text)
        
        return(review_text)
    
    def sentiment (self, review_text):
        
        review_text = self.clean_review(review_text)
        
        X = tf_idf.transform([review_text])
        
        y= model.predict(X)
        
        return(int(y))
        

        
import pandas as pd
import numpy as np
import random

item_based_reco = pd.read_csv('item_based_reco.csv')
item_based_reco.set_index('reviews_username', inplace=True)

dataset = pd.read_csv('sample30.csv')

id_name= pd.read_csv('id_name.csv')

class pdt_recommendation(get_sentiment):
    
    def __init__(self):
        
        #self.username= username
        return None
    
    
    def recom_using_item_based(self, username): ##takes username as input and returns 20 reco pdts
        
        d = item_based_reco.loc[username].sort_values(ascending=False)[0:20]
        top_20 = list(d.index)
        
        return top_20
    
    def pdt_overall_sentiment(self, pdt_id): ##takes a pdt id as input serach the reviews and give the sentiment_score
        
        filt_df= dataset.loc[dataset['id']== pdt_id, ['reviews_title', 'reviews_text']]
        filt_df['merge']= filt_df['reviews_title']+ str(' ')+ filt_df['reviews_text']
        all_reviews = list(filt_df['merge'])
        
        if len(all_reviews)>300:
            all_reviews= random.sample(all_reviews, 300)
        
        #print(len(all_reviews))
        
        sentiment_score= sum([self.sentiment(str(review)) for review in all_reviews])
        
        return (round(sentiment_score/len(all_reviews),2))
    
    def predict(self, username):
        
        top_20_pdt_id= self.recom_using_item_based(username)
        top_20_pdt_sent={}
        for id in top_20_pdt_id:
            top_20_pdt_sent[id] = self.pdt_overall_sentiment(id)
            
        top_20_pdt_sent= dict(sorted(top_20_pdt_sent.items(), key= (lambda x: -x[1])))

        top_5_pdt_id= list(top_20_pdt_sent.keys())[0:5]
        
        
        top_5_pdt_name= [id_name.loc[id_name['id']==x , 'name_by_brand'].values[0] for x in top_5_pdt_id]
        
        
        return(top_5_pdt_name)
            

        