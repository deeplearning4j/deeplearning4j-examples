import tensorflow as tf
from tensorflow import keras

import numpy as np

boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()


order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std


# save mean and std for use from java
with open('stats.txt') as f:
    f.write('{},{}'.format(mean, std))

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, 
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model


with tf.Session() as sess:
    keras.backend.set_session(sess)
    model = build_model()
    model.fit(train_data, train_labels, epochs=500,
                    validation_split=0.2)
    sess.run(tf.global_variables_initializer())
    output_node_name = model.output.name.split(':')[0]
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_node_name])
    with tf.gfile.GFile("boston.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())
