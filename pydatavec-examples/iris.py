from pydatavec.utils import download_file
from pydatavec import Schema, TransformProcess, SparkExecutor
import os
import pyspark

# Another basic preprocessing and filtering example. We download the raw iris dataset from the internet.
# The file we downloaded has 2 problems:
# 1. an empty line at the bottom.
# 2. the labels are in the form of strings. We want an integer.


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
tp.categorical_to_integer("Species")

# Do the transformation on spark
output = tp(filtered_data)

print(list(output))
