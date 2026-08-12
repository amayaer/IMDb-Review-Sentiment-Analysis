# %%
import pandas as pd 
import matplotlib.pyplot as plt 
import nltk 
from nltk.corpus import stopwords

# %%
#Load dataset 

data = pd.read_csv('/Users/amayasmacbook/Documents/imdbdataset.csv')

# %%
#Pre processing
def clean_data(review):
    cleaned_review = []
    
    for word in review.split(): 
        if word.lower() not in stopwords.words('english'):
            cleaned_review.append(word) 
        else: 
            continue 
        
    return ' '.join(cleaned_review) 

# %%
#Data cleaning 
data['review'] = data['review'].apply(clean_data)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=2500)

reviews = cv.fit_transform(data['review']).toarray()

# %%
data['sentiment'] = data['sentiment'].replace(['positive','negative'],[1,0])                                           

# %%
#Model Training 
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# %%
from sklearn.model_selection import train_test_split

reviews_train, reviews_test, sentiment_train, sentiment_test = train_test_split(reviews, data['sentiment'],test_size=0.2) 

# %%
model.fit(reviews_train,sentiment_train)

# %%
predict = model.predict(reviews_test)

# %%

from sklearn.feature_extraction.text import TfidfVectorizer


user_review = input("Enter movie review:") 
clean_review = clean_data(user_review) 
review_vectorized = cv.transform([clean_review]).toarray()

prediction = model.predict(review_vectorized) 

if prediction == 1: 
    print("Prediction: This review is positive.")

else: 
    print("Prediction: This review is negative.")





