import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import random

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


'''
Here we iterate through the patterns and tokenize the sentence using nltk.word_tokenize() function and append 
each word in the words list. We also create a list of classes for our tags.
'''
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


'''
Now we will lemmatize each word and remove duplicate words from the list. 
Lemmatizing is the process of converting a word into its lemma form and then creating a pickle file to store the Python 
objects which we will use while predicting.
'''


# lemmatize each word and remove duplicate words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# sort classes
classes = sorted(list(set(classes)))

# documents = combinations between patterns and intents
print(len(documents), 'documents')

# classes = intents
print(len(classes), 'classes', classes)

# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('word.pkl', 'wb'))
pickle.dump(classes, open('class.pkl', 'wb'))


'''
Now, we will create the training data in which we will provide the input and the output. Our input will be the pattern 
and output will be the class our input pattern belongs to. But the computer does not understand text so we will convert 
text into numbers.
'''

# create your training data
training = []

# create an empty array for the output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words#
    bag = []

    # list of tokenized words for the pattern
    pattern_words = doc[0]

    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # Create our bag of word arrays with 1 if word match found in current pattern.
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")


'''
We have our training data ready, now we will build a deep neural network that has 3 layers. 
We use the Keras sequential API for this. After training the model for 200 epochs, 
we achieved 100% accuracy on our model. Let us save the model as ‘chatbot_model.h5’.
'''
model = Sequential()
model.add(Dense(128, input_shape= (len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True, name='SGD')
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# Fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5',hist)

print('model created')
