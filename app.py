######################
# Import libraries
######################

import pandas as pd
import numpy as np
import streamlit as st

from  keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import os


######################
# Page Title
######################

st.write("""
# Sentiment Classifier
""")

######################
# Main Content
######################
st.write("## Please Enter the sentance you'd like to test")

def model_load():
    model_path = os.path.join('model','my_model.h5')
    model = tf.keras.models.load_model(model_path)
    return model

model = model_load()

def get_key(value):
    dictionary={'joy':0,'anger':1,'love':2,'sadness':3,'fear':4,'surprise':5}
    for key,val in dictionary.items():
          if (val==value):
            return key

def read_data():
    train_file = os.path.join('Data','train.txt')
    df_train = pd.read_csv(train_file, header =None, sep =';', names = ['Input','Sentiment'], encoding='utf-8')
    return(df_train)


def predict(sentence):
    df = read_data()
    X=df['Input']
    sentence_lst=[sentence]
    tokenizer = Tokenizer(15212,lower=True,oov_token='UNK')
    tokenizer.fit_on_texts(X)
    sentence_seq=tokenizer.texts_to_sequences(sentence_lst)
    sentence_padded=pad_sequences(sentence_seq,maxlen=80,padding='post')
    
    predict_x = model.predict(sentence_padded)
    classes_x=np.argmax(predict_x,axis=1)
    ans=get_key(classes_x)
    answer = "## The emotion predicted is {}.".format(ans)
    return(answer)

def main():
    text_a = st.text_input('Input your sentance')
    if text_a != '':
        text = predict(text_a)
        st.write(text)
    else:
        st.write("### You didn't type anything")

if __name__ == '__main__':
    main()