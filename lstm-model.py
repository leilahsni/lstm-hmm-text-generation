from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint # will help us save model at best training epoch
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

import tensorflow
import keras
import pandas as pd
import numpy as np

tokenizer = Tokenizer()

def text_to_sequence_ngram(corpus):
    ''' funciton to get numerical value for each word of corpus and put them in list '''

    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    sequences = []

    for headline in corpus:
        tokens = tokenizer.texts_to_sequences([headline])[0]
        for i in range(1, len(tokens)):
            n_gram = tokens[:i+1]
            sequences.append(n_gram)

    return sequences, total_words

def generate_padded_sequences(sequences):
    ''' function to pad sequences '''

    max_sequence_len = max([len(x) for x in sequences])
    sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))
    
    predictors, label = sequences[:,:-1],sequences[:,-1]
    label = to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len

def create_model(max_sequence_len, total_words):
    ''' create our model for text generation '''

    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=(max_sequence_len - 1)))
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

def generate_text(predictors, next_words, model, max_sequence_len):

    for __ in range(next_words):

        tokens = tokenizer.texts_to_sequences([predictors])[0]
        tokens = pad_sequences([tokens], maxlen=(max_sequence_len-1), padding='pre')
        predicted = model.predict(tokens, verbose=0)
        predicted = np.argmax(predicted, axis=1)
        
        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        predictors += " " + output_word

    return predictors.title()

if __name__ == '__main__':

    dataset = pd.read_csv('dataset/news-headlines.csv')
    corpus = dataset['headline_text'][:100000]

    sequences, total_words = text_to_sequence_ngram(corpus)

    predictors, label, max_sequence_len = generate_padded_sequences(sequences)

    # uncomment lines 91 to 113 to re-train model

    # model = create_model(max_sequence_len, total_words) 

    filepath = 'model/lstm-text-generation-model.hdf5'

    '''determining optimal number of epochs'''
    #es = EarlyStopping(monitor = 'val_loss', patience=5, mode = 'min', verbose = 1)

    '''checkpoint = ModelCheckpoint(filepath=filepath, 
                                monitor='val_loss',
                                verbose=1, 
                                save_best_only=True,
                                mode='min')

    history = model.fit(predictors, label, batch_size=10, epochs=100, callbacks=[es,checkpoint], validation_split=0.2)''' # epoch = number of time the model goes through training corpus

    # plot the training history
    '''plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.savefig('graphs/model_training_history') # save graph
    plt.show()'''

    model = keras.models.load_model(filepath)

    print(generate_text("Benedict Cumberbatch", 5, model, max_sequence_len))
    print(generate_text("United States", 5, model, max_sequence_len))
    print(generate_text("Donald Trump", 7, model, max_sequence_len))