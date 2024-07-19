import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')

df = pd.read_csv('dataset/spotify.csv')
df = df.iloc[:25000,:]
df = df.drop('link',axis=1).reset_index(drop=True)
df =df.sample(25000)

# Text Preprocessing

df['text'].str.lower().replace(r'^\W\s' , '  ').replace(r'\n',' ',regex=True)

stemmer = PorterStemmer()

def token(txt):
    token = nltk.word_tokenize(txt)
    a = [stemmer.stem(w) for w in token]
    return "".join(a)

df['text'].apply(lambda x:token(x))
tfid = text.TfidfVectorizer(analyzer='word',stop_words='english')
matrix=tfid.fit_transform(df['text'])
similar=cosine_similarity(matrix)

def recommender(song_name):
    idx = df[df['song']==song_name].index[0]
    distance = sorted(list(enumerate(similar[idx])),reverse = True , key = lambda x:x[1])
    song = []
    for s_id in distance[1:5]:
        song.append(df.iloc[s_id[0]].song)
    return song
pickle.dump(similar,open('similarity.pkl','wb'))
pickle.dump(df,open('df.pkl','wb'))