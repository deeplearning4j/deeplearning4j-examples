# Similar to basic.py, but uses local executor.

from pydatavec import Schema, TransformProcess
from pydatavec import NotInSet, LessThan


# Let's define the schema of the data that we want to import
# The order in which columns are defined here should match the order in which they appear in the input data


input_schema = Schema()

input_schema.add_string_column("DateTimeString")
input_schema.add_string_column("CustomerID")
input_schema.add_string_column("MerchantID")

input_schema.add_integer_column("NumItemsInTransaction")

input_schema.add_categorical_column("MerchantCountryCode", ["USA", "CAN", "FR", "MX"])

# Some columns have restrictions on the allowable values, that we consider valid:

input_schema.add_double_column("TransactionAmountUSD", 0.0, None, False, False)  # $0.0 or more, no maximum limit, no NaN and no Infinite values

input_schema.add_categorical_column("FraudLabel", ["Fraud", "Legit"])


# Lets define some operations to execute on the data...
# We do this by defining a TransformProcess
# At each step, we identify column by the name we gave them in the input data schema, above

tp = TransformProcess(input_schema)

# Let's remove some column we don't need

tp.remove_column("CustomerID")
tp.remove_column("MerchantID")

# Now, suppose we only want to analyze transactions involving merchants in USA or Canada. Let's filter out
# everything except for those countries.
# Here, we are applying a conditional filter. We remove all of the examples that match the condition
# The condition is "MerchantCountryCode" isn't one of {"USA", "CAN"}

tp.filter(NotInSet("MerchantCountryCode", ["USA", "CAN"]))

# Let's suppose our data source isn't perfect, and we have some invalid data: negative dollar amounts that we want to replace with 0.0
# For positive dollar amounts, we don't want to modify those values

# First we build the condition object:

condition = LessThan("TransactionAmountUSD", 0.0)

# Next, we call .replace():

tp.replace("TransactionAmountUSD", 0.0, condition)  # Here 0.0 is the value we use to replace negative amounts

# Finally, let's suppose we want to parse our date/time column in a format like "2016/01/01 17:50.000"
# We use JodaTime internally, so formats can be specified as follows: http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html

tp.string_to_time("DateTimeString", "YYYY-MM-DD HH:mm:ss.SSS", time_zone="UTC")

# However, our time column ("DateTimeString") isn't a String anymore. So let's rename it to something better:
tp.rename_column("DateTimeString", "DateTime")

# At this point, we have our date/time format stored internally as a long value (Unix/Epoch format): milliseconds since 00:00.000 01/01/1970
# Suppose we only care about the hour of the day. Let's derive a new column for that, from the DateTime column

tp.derive_column_from_time(source_column="DateTime", new_column="HourOfDay", field='hour_of_day')

# We no longer need our "DateTime" column, as we've extracted what we need from it. So let's remove it
tp.remove_column("DateTime")

# Run the transform process over file exampledata.csv using spark:

result = tp('basic_example.csv', executor='local')
result.save('temp')


# `result` is a StringRDD object. You can save it to a csv file using:

result.save_to_csv('basic_example_output.csv')

# Or you can get a python list of strings:

print(list(result))
