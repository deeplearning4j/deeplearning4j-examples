import tensorflow as tf
from tensorflow import keras

import numpy as np

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# save indices for later use from java
with open('word_index.txt', 'w') as f:
    for k, v in word_index.items():
        f.write(str(k.encode('utf-8'))[2:-1] + ',' + str(v) + '\n')

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# save some text for testing from java
save_text = ''
for i in range(10):
    save_text += decode_review(test_data[i]) + '\n'

save_text = save_text[:-1]

with open('reviews.txt', 'w') as f:
    f.write(save_text)


# prepare the data
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

vocab_size = 10000

with tf.Session() as sess:
    keras.backend.set_session(sess)
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.summary()


    model.compile(optimizer=tf.train.AdamOptimizer(),
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    history = model.fit(train_data,
                    train_labels,
                    epochs=40,
                    batch_size=512,
                    validation_data=(test_data, test_labels),
                    verbose=1)
    
    sess.run(tf.global_variables_initializer())
    output_node_name = model.output.name.split(':')[0]
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_node_name])
    with tf.gfile.GFile("imdb.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())





