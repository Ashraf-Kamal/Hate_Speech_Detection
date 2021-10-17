import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Input, Embedding, Bidirectional, TimeDistributed, merge, concatenate, GRU
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import keras_metrics
import keras.backend as K
from keras.models import Sequential
import tweepy
import os
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras_self_attention import SeqSelfAttention
import Capsule_Class as Caps

import random as rn
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
rn.seed(1254)
tf.set_random_seed(89)
from keras import backend as K
session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#df = pd.read_csv(r'Final_Datasets\Founta_18\Founta_Final.csv',delimiter=',',encoding='latin-1')
df = pd.read_csv(r'Final_Datasets\Roy_20\Roy_Final.csv',delimiter=',',encoding='latin-1')

df=df.dropna()

Routings = 3
Num_capsule = 5
Dim_capsule = 4

X = df.Text
Y = df.Label
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)

train_msg, test_msg, train_labels, test_labels = train_test_split(X, Y,test_size=0.2, random_state = 42)

max_words =5000
max_len = 25

tokenizer= Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_msg)

training_sequences = tokenizer.texts_to_sequences(train_msg)
training_padded = sequence.pad_sequences(training_sequences,maxlen=max_len)

test_sequences = tokenizer.texts_to_sequences(test_msg)
testing_padded = sequence.pad_sequences(test_sequences,maxlen=max_len)

embeddings_index = dict()
f = open('F:\Glove\glove.twitter.27B.50d.txt',encoding="utf8")
for line in f:    
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
print (len(embeddings_index))
f.close()
embedding_matrix = np.zeros((max_words, 50))
for word, index in tokenizer.word_index.items():
    if index > max_words - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
print (embedding_matrix)


main_input = Input(shape=(25,), name='main_input')

embedding=Embedding(max_words, 50, input_length=max_len, weights=[embedding_matrix], trainable=False)(main_input)
cnn=Conv1D(128, 3, activation='relu')(embedding)
max=MaxPooling1D(pool_size=2)(cnn)
BiGRU=Bidirectional(GRU(128, return_sequences=True))(max)
BiGRU_drop = Dropout(0.4)(BiGRU)
capsule = Caps.Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(BiGRU)
Flat_capsule_output = Flatten()(capsule)
Final_output = Dense(1, activation='sigmoid', trainable=True)(Flat_capsule_output)
model = Model(inputs=[main_input], outputs=Final_output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', precision_m, recall_m, f1_m])
model.summary()

hist =model.fit(training_padded,train_labels, validation_data=(testing_padded, test_labels), batch_size=256, epochs=50, verbose=2)
accr = model.evaluate(testing_padded, test_labels, verbose=0)
print('Testing set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f} \n  Precision: {:0.3f} \n  Recall: {:0.3f} \n  F-score: {:0.3f}'.format(accr[0],accr[1],accr[2],accr[3],accr[4]))

print ('Training loss:', np.mean(hist.history['loss']))
print ('Training accuracy:', np.mean(hist.history['acc']))
print ('Training precision:', np.mean(hist.history['precision_m']))
print ('Training recall:', np.mean(hist.history['recall_m']))
print ('Training f-score:', np.mean(hist.history['f1_m']))

print ('**************')

print ('Testing loss:', np.mean(hist.history['val_loss']))
print ('Testing accuracy:', np.mean(hist.history['val_acc']))
print ('Testing precision:', np.mean(hist.history['val_precision_m']))
print ('Testing recall:', np.mean(hist.history['val_recall_m']))
print ('Testing f-score:', np.mean(hist.history['val_f1_m']))

#model.save('Model_Result\model.Proposed_Model_Founta')
model.save('Model_Result\model.Proposed_Model_Roy')



