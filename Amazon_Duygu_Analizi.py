
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
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate,train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import TfidfVectorizer
filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x :'%.2f' % x)

df = pd.read_excel("amazon.xlsx")

df.head()
df.info()

df["Review"] = df["Review"].str.lower() #tüm harfleri küçük harfe çevirdik.

df["Review"] = df["Review"].apply(lambda x : re.sub(r'[^\w\s]', "", str(x))) #noktalama işaretlerini kaldırdık.

df["Review"] = df["Review"].str.replace("\d", '') #sayıları kaldıralım

#stopwords önemsiz kelimeleri çıkarmak için:
#nltk.download("stopwords")

sw = stopwords.words("english")

df["Review"] = df["Review"].apply(lambda x : " ".join(x for x in str(x).split() if x not in sw))#burada split metoduyla kelimeleri ayırıp liste haline getiriyoruz daha sonra bu listede geziyoruz ve stopwords olmayan kelimeleri alıyoruz. ve join ile birleştiriyoruz.

#nadir kullanılan kelimeleri çıkaralım.

drop = pd.Series(" ".join(df["Review"]).split()).value_counts() #bütün yorumları birleştirip tek metin haline getirdik. daha sonra kelimeleri ayırdık ve kaçar tane olduğunu series olarak ttutuk.
drop = drop.reset_index()
drop = drop[drop["count"] < 1000]
df["Review"] = df["Review"].apply(lambda x : " ".join(x for x in x.split() if x not in drop ))

################################################
#Lemmatization
################################################
#nltk.download("wordnet")

df["Review"] = df["Review"].apply(lambda x : " ".join([Word(word).lemmatize() for word in x.split()]))

#Metin Görselleştirme
tf = df["Review"].apply(lambda x : pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf.columns = ["words","tf"]
tf[tf["tf"] > 500].plot.bar(x = 'words',y = "tf")
plt.show()

#wordcloud

text = " ".join(x for x in df.Review)

wordcloud = WordCloud().generate(text)

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()

#Duygu Analizi
#nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

df["polarity_score"] = df["Review"].apply(lambda x : "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
#sentiment analyzer ile yorumları etiketleyerek yorum sınıflandırma makine öğrenmesi model,için bağımlı değişken oluşturulmuş olur.

#model

y = df["polarity_score"]
X = df["Review"]
# Veriyi train ve test setlerine bölmek
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tf_idf_word = TfidfVectorizer()
X_tf_idf_word_train = tf_idf_word.fit_transform(X_train)
X_tf_idf_word_test = tf_idf_word.transform(X_test)


log_model = LogisticRegression().fit(X_tf_idf_word_train,y_train)
y_pred = log_model.predict(X_tf_idf_word_test)

print(classification_report(y_pred,y_test))
cross_val_score(log_model,X_tf_idf_word_test,y_test,cv=5).mean()

rf_model = RandomForestClassifier().fit(X_tf_idf_word_train,y_train)
cross_val_score(rf_model, X_tf_idf_word_test, y_test, cv=5, n_jobs=-1).mean()