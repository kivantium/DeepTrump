#!/usr/bin/python
#-*- coding: utf-8 -*-

from __future__ import print_function
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
from requests_oauthlib import OAuth1Session
import json
import sys
import codecs
import string

# specify encode (may not necessary)
reload(sys)
sys.setdefaultencoding("utf-8")
sys.stdout = codecs.getwriter('utf_8')(sys.stdout) 

CK = "XgzvXbBDU9gi8CbeWBDlkoZIn"
CS = "qsgeJg7BcVN0PSQ3yqgyTdn0v6qJnhJskIygxa776sDYGOJjgG"
AT = "818765750844334080-XawykMR4fAbWq05H6s6WEXWtEjaJHau" 
AS = "jtLQ8RJMFjUpedqjgr8IUG7Pd59MrMa0N3H29roE4nKlI"

# open file
text = codecs.open('data.txt', 'r', 'utf-8').read()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 10
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
url = "https://api.twitter.com/1.1/statuses/update.json"

twitter = OAuth1Session(CK, CS, AT, AS)

# train the model, output generated text after each iteration
for iteration in range(1, 50):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    # learning
    model.fit(X, y, batch_size=128, nb_epoch=1)
    print('batch finished')
    model.save('kana-model{}.h5'.format(iteration))
    # generate tweet
    generated = ''
    start_index = random.randint(0, len(text) - maxlen - 1)
    sentence = text[start_index: start_index + maxlen]
    start = sentence
    generated = ''
    for i in range(140):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, 0.4)
        next_char = indices_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
    # encoding in Python is very difficult)
    tweet = '[{}]{}'.format(start, generated)
    tweet = tweet.decode('utf-8')
    # replace @ to (at)
    tweet = tweet.replace('@', '(at)')[0:139]
    print(tweet)
    # post
    params = {"status": tweet}
    req = twitter.post(url, params = params)
    if req.status_code == 200:
        print ("OK")
    else:
        print ("Error: {} {} {}".format(req.status_code, req.reason, req.text))
