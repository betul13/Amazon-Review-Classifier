##############################################
#Introduction to text mining and natural language processing
##############################################

#sentiment analysis and sentiment modeling for amazon reviews

#1 Text Preprocessing
#2 Text Visulation
#3 Sentiment Analysis
#4 Feature Engineering
#5 sentiment modeling


from warnings import filterwarnings
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud
import re
filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x :'%.2f' % x)

###################
#Text Preprocessing
###################

df = pd.read_csv(r"amazon_reviews.csv",sep = ',')

df.head()

############################################
#Normalizing Case Folding
###########################################

# Şimdi kodunuzu çalıştırarak ve sonucu kontrol ederek işlemi gerçekleştirin
df["reviewText"] = df["reviewText"].str.lower()

# Noktalama işaretlerini boşluk karakterleriyle değiştirelim
df["reviewText"] = df["reviewText"].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
df["reviewText"]

#################################
#numbers
################################

df["reviewText"] = df["reviewText"].apply(lambda x: re.sub(r'\d', '', str(x)))
df["reviewText"]

#################################
#stopwords
################################

sw = stopwords.words('english')

df["reviewText"] = df["reviewText"].apply(lambda x : " ".join(x for x in str(x).split() if x not in sw))

####################################
#rarewords
####################################

#temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()
temp_df = df["reviewText"].str.split().explode().value_counts()

drops = temp_df[temp_df <= 1]

df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

#####################################
#tokenization
#####################################

df["reviewText"].apply(lambda x : TextBlob(x).words).head()

###################################
#lemmatization kök indirgemesi
####################################

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


###############################################################
#text visualization
###############################################################

tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
"""Bu kod parçası, her bir metni kelimelere ayırarak (split(" ")) ve ardından her kelimenin sayısını hesaplayarak (pd.value_counts()) doğru bir şekilde çalışır. 
Bu sayede, her kelimenin toplam sayısını içeren bir Seri elde edilir. Daha sonra, sum(axis=0) kullanılarak bu Serilerin toplamı alınır ve sonuç olarak bir Seri elde edilir. ,
En son olarak, reset_index() kullanılarak indeks sıfırlanır ve sonuç DataFrame'e dönüştürülür."""

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending = False)

#####################
#barplot
#####################

tf[tf["tf"]>500].plot.bar(x = "words", y = "tf")
plt.show()

####################
#wordcloud
####################

text = " ".join(i for i in df.reviewText)
wordcloud = WordCloud().generate(text) #text dosyasından wordcloud oluşturur.
plt.imshow(wordcloud,interpolation = "bilinear")
plt.axis("off")
plt.show()

################################
#Sentiment Modellemesi
###############################

#nltk.download("vader_lexicon")
sia =SentimentIntensityAnalyzer()

df["polarity_score"] = df["reviewText"].apply(lambda x : sia.polarity_scores(x)["compound"])

df[(df["polarity_score"] > 0) & (df['overall'] < 3.0)]

###############################
#sentiment modeling
###############################

##Feature Engineering

df["sentiment_label"] = df["reviewText"].apply(lambda x : "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg" )
df.groupby("sentiment_label")["overall"].mean()

df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

y = df["sentiment_label"]
X = df["reviewText"]

###################################
#Count Vectors
###################################

#from sklearn.feature_extraction.text import CountVectorizer

##################################
#TF-IDF
#################################

from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X)

######################################
#logistic regression
####################################

log_model = LogisticRegression().fit(X_tf_idf_word, y)

cross_val_score(log_model,
                X_tf_idf_word,
                y, scoring = "accuracy",
                cv = 5).mean()

new_review = pd.Series("this is product is great")

new_review = TfidfVectorizer().fit(X).transform(new_review)

log_model.predict(new_review)

#kendi setimizden örnek

random_review = pd.Series(df["reviewText"].sample(1).values)

new_review = TfidfVectorizer().fit(X).transform(random_review)

log_model.predict(new_review)

########################################
#Random Forests
########################################

rf_model = RandomForestClassifier().fit(X_tf_idf_word, y)

cross_val_score(rf_model, X_tf_idf_word, y, cv = 5, n_jobs=-1).mean()

###########################################
#Hiperparametre Optimizasyonu
###########################################

rf_model = RandomForestClassifier(random_state=17)

rf_params = { "max_depth" : [8, None],
             "max_features" : [7 ,"auto"],
             "min_samples_split" : [2, 5, 8],
             "n_estimators" : [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv = 5,
                            n_jobs = -1,
                            verbose = True).fit(X_tf_idf_word,y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state = 17).fit(X_tf_idf_word, y)

cross_val_score(rf_final, X_tf_idf_word, y, cv = 5, n_jobs = -1).mean()