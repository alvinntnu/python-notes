# Seqeunce Model with Attention (Thushan)

- Bahdanau Attention Layber developed in [https://github.com/thushv89/attention_keras]
- Thushan Ganegedara's
[Attention in Deep Networks with Keras](https://towardsdatascience.com/light-on-math-ml-attention-with-keras-dc8dbc1fad39)
- Still not working yet

import tensorflow as tf
import keras as keras
print(tf.__version__)
print(keras.__version__)

from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras import Model
from keras.models import Sequential
from keras.layers import LSTM, GRU, Concatenate
from keras.layers import Attention
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras import Input
from attention import AttentionLayer

# generate a sequence of random integers
def generate_sequence(length, n_unique):
    return [randint(0, n_unique - 1) for _ in range(length)]


# one hot encode sequence
def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)


# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


# prepare data for the LSTM
def get_pair(n_in, n_out, cardinality):
    # generate random sequence
    sequence_in = generate_sequence(n_in, cardinality)
    sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in - n_out)]
    # one hot encode
    X = one_hot_encode(sequence_in, cardinality)
    y = one_hot_encode(sequence_out, cardinality)
    # reshape as 3D
    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], y.shape[1]))
    return X, y


# # define the encoder-decoder model
# def baseline_model(n_timesteps_in, n_features):
#     model = Sequential()
#     model.add(LSTM(150, input_shape=(n_timesteps_in, n_features)))
#     model.add(RepeatVector(n_timesteps_in))
#     model.add(LSTM(150, return_sequences=True))
#     model.add(TimeDistributed(Dense(n_features, activation='softmax')))
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     return model


# # define the encoder-decoder with attention model
# def attention_model(n_timesteps_in, n_features):
#     model = Sequential()
#     model.add(
#         LSTM(150,
#              input_shape=(n_timesteps_in, n_features),
#              return_sequences=True))
#     model.add(AttentionDecoder(150, n_features))
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     return model


# # train and evaluate a model, return accuracy
# def train_evaluate_model(model, n_timesteps_in, n_timesteps_out, n_features):
#     # train LSTM
#     for epoch in range(5000):
#         # generate new random sequence
#         X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
#         # fit model for one epoch on this sequence
#         model.fit(X, y, epochs=1, verbose=0)
#     # evaluate LSTM
#     total, correct = 100, 0
#     for _ in range(total):
#         X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
#         yhat = model.predict(X, verbose=0)
#         if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
#             correct += 1
#     return float(correct) / float(total) * 100.0

n_features = 50
n_timesteps_in = 5
n_timesteps_out = 2
X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)

print(one_hot_decode(X[0]))
print(one_hot_decode(y[0]))
print(X.shape)
print(y.shape)

get_pair(n_timesteps_in, n_timesteps_out, n_features)

batch_size=1
en_timesteps=5
fr_timesteps=2
en_vsize=50
fr_vsize=50
hidden_size=150


encoder_inputs = Input(batch_shape=(batch_size,en_timesteps, en_vsize), name='encoder_inputs') 
decoder_inputs = Input(batch_shape=(batch_size, fr_timesteps, fr_vsize),name='decoder_inputs')

#encoder_inputs = X
#decoder_inputs = y

encoder_gru =GRU(hidden_size, return_sequences=True, return_state=True, name='encoder_gru') 
encoder_out, encoder_state = encoder_gru(encoder_inputs)

decoder_gru =GRU(hidden_size, return_sequences=True, return_state=True, name='decoder_gru') 
decoder_out, decoder_state = decoder_gru(decoder_inputs,initial_state=encoder_state)


attn_layer = AttentionLayer(name='attention_layer') 
attn_out, attn_states = attn_layer([encoder_out, decoder_out])


decoder_concat_input =Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

dense =Dense(fr_vsize, activation='softmax', name='softmax_layer') 
dense_time = TimeDistributed(dense, name='time_distributed_layer') 
decoder_pred = dense_time(decoder_concat_input)
full_model =Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred) 
full_model.compile(optimizer='adam', loss='categorical_crossentropy')

full_model.summary()

losses = []
for epoch in range(100):  
    X, y = get_pair(en_timesteps, fr_timesteps,en_vsize)
    #     X = array(one_hot_decode(X[0])).reshape(1, X[0].shape[0])
    #     y = array(one_hot_decode(y[0])).reshape(1,y[0].shape[0])
    #     full_model.fit(X,y, epochs=1, verbose=1)

    #     en_onehot_seq = to_categorical(
    #         en_seq[bi:bi + batch_size, :], num_classes=en_vsize)
    #     fr_onehot_seq = to_categorical(
    #         fr_seq[bi:bi + batch_size, :], num_classes=fr_vsize)

#     full_model.fit([X, y[:, :-1, :]], y[:, 1:, :])
    full_model.fit([X,y],y)

#     l = full_model.evaluate([X, y[:, :-1, :]], y[:, 1:, :],
#                             batch_size=batch_size, verbose=2)
    l = full_model.evaluate([X,y],y, batch_size=batch_size, verbose=2)

    losses.append(l)

print(np.mean(losses))

total, correct= 100,0
for _ in range(10):
    X,y = get_pair(en_timesteps, fr_timesteps,en_vsize)
    yhat = full_model.predict([X,y], verbose=0)
    print('Expected', one_hot_decode(y[0]), 
          'Predicted', one_hot_decode(yhat[0]))