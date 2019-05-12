################################################################################
# Copyright (c) 2015-2019 Skymind, Inc.
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################

from pydatavec.utils import download_file
from pydatavec import Schema, TransformProcess, SparkExecutor
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import os
import pyspark
import json



# An extention of the iris example where we train a keras model with the output


# Download dataset (if not already downloaded)
filename = "iris.data"
temp_filename = filename + '_temp'
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

if not os.path.isfile(filename):
    if os.path.isfile(temp_filename):
        os.remove(temp_filename)
    download_file(url, temp_filename)
    os.rename(temp_filename, filename)

# We use pyspark to filter empty lines
sc = pyspark.SparkContext(master='local[*]', appName='iris')
data = sc.textFile('iris.data')
filtered_data = data.filter(lambda x: len(x) > 0)

# Define Input Schema
input_schema = Schema()
input_schema.add_double_column('Sepal length')
input_schema.add_double_column('Sepal width')
input_schema.add_double_column('Petal length')
input_schema.add_double_column('Petal width')
input_schema.add_categorical_column("Species", ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

# Define Transform Process
tp = TransformProcess(input_schema)
#tp.categorical_to_integer("Species")
tp.one_hot("Species")

# Do the transformation on spark
output = tp(filtered_data)


# Convert to numpy array
np_array = np.array([[float(i) for i in x.split(',')] for x in output])

x = np_array[:, :-3]
y = np_array[:, -3:]

 
# Build the Keras model

model = Sequential()

model.add(Dense(10, input_shape=(4,), activation='relu', name='fc1'))
model.add(Dense(10, activation='relu', name='fc2'))
model.add(Dense(3, activation='softmax', name='output'))

optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Model Summary: ')
print(model.summary())


# Train

model.fit(x, y, batch_size=5, epochs=200)


# Save Transform process
with open('iris_tp.json', 'w') as f:
    f.write(tp.to_java().toJson())


# Save keras model
model.save('iris_model.h5')
