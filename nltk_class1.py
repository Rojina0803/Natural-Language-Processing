#! expression -- tokenization --  stemmization -- lemmatization -- remove stop words( is, am, are, the) -- TF(Term Frequency)-- IDF

#!tokenization -- taking a single word as a token 
#!stemmization -- here having is written as have, eating as eat
#!lemmatization -- writing good as best
#!Term Frequency -- counting the occurences of a token by a total token
#!Inverse Document Frequency -- total documents by number of occurrences

import nltk
print(nltk.__version__)


from nltk.corpus import stopwords 
# print(stopwords.words('english')[0:10])
messages = [line.rstrip() for line in open('SMSSpamCollection')]
# print(messages)
import pandas as pd 
messages = pd.read_csv('SMSSpamCollection', sep= '\t', names=['label', 'message'])

# print(messages.groupby('label').describe())
messages['length'] = messages['message'].apply(len)
# print(messages.head())
#!TEXT PRE PROCESSING 

import string 

mess = "Sample Message! Notice: it has punctuation."

nopunc = [char for char in mess if char not in string.punctuation]  #! to remove the punctuation

nopunc = ''.join(nopunc)
# print(nopunc)
# print(nopunc.split())
from nltk.corpus import stopwords

clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
# print(clean_mess)
def text_process(mess):
    #remove punctuation 
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

messages['message'].apply(text_process)
# print(messages.head())
from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer= text_process).fit(messages['message'])
# print(len(bow_transformer.vocabulary
# ))

messages_bow = bow_transformer.transform(messages['message'])
# print(messages_bow)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer().fit(messages_bow)
message_tfidf=TfidfTransformer.transform
tf_idf3=tfidf_transformer.transform(bow_transformer)
# print(tf_idf3)

# print(tfidf_transformer.idf_[bow_transformer.vocabulary_['early']])

from sklearn.naive_bayes import MultinomialNB
spam_detect_model= MultinomialNB().fit(message_tfidf,messages['label'])

# print(messages['message'][3])
# print(messages['label'][3])
# print(spam_detect_model.predict(tf_idf3)[0])

random_text='FREE FREEE!! FEEE CARS AVAILABLE!!!'
val=text_process(random_text)
print(val)
bow_val= bow_transformer.transform(val)
tfidf_val=tfidf_transformer.transform(bow_val)
print(spam_detect_model.predict(tfidf_val)[0])


# how to implement it to wwebsite
import pickle
with open('spam_detector.pkl','wb') as file:
    pickle.dump(spam_detect_model)

with open('spam detector.pkl','rb') as file:
    loaded_model=pickle.load(file)
print(loaded_model.predict(tfidf_val)[0])