#  Recommend top 5 products based on sentiment analysis of products

- In this project we will create models to predict sentiment of reviews

- Approach taken to build sentiment model :
     - use only necessary columns for creating models
     - Text Preprocessing on review text like : remove urls, make reviews in the same case, remove words which have maximum frquency in both sentiments, Tokenize words,Correct the spelling of words
     - Split data into train-test
     - Create word embedding using : "TF-IDF" and " Word2Vec"
     - Handle class imbalance using SMOTE
     - Then on this dataset  create Logistic, XGBoost, Naive bayes model
     - Compare which model and which embedding is good and use that model for final reviews
     - Do the hyperparameter tuning on Final selected model to impeove accuracy
     - Use this model for sentiment prediction
	 
- Approach taken to build recommendation Enging :
	 - Use colaborative User user method and make the prediction
	 - Use item-item method and make the predictions
	 - Decide best method two use by comparing above two methods
	 - and recommend top 20 products based on user name using above model
	 
- Final top 5 pediction approach :
     - take top 20 products recommended using recomendation model
	 - for those top 20 products predict the sentiment score using sentiment model build earlier
	 - sort the values and take top 5  values having positive score.
	 - Recommend those top 5 products to user