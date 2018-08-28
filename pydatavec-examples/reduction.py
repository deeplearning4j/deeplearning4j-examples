################################################################################
# Copyright (c) 2015-2018 Skymind, Inc.
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


'''
In this simple example: We'll show how to combine multiple independent records by key.
Specifically, assume we have data like "person,country_visited,entry_time" and we want to know how many times
each person has entered each country.
'''



from pydatavec import Schema, TransformProcess

# Define the input schema

schema = Schema()
schema.add_string_column('person')
schema.add_categorical_column('country_visited', ['USA', 'Japan', 'China', 'India'])
schema.add_string_column('entry_time')


# Define the operations we want to do

tp = TransformProcess(schema)

# Parse date-time
# Format for parsing times is as per http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html

tp.string_to_time('entry_time', 'YYYY/MM/dd')


# Take the "country_visited" column and expand it to a one-hot representation
# So, "USA" becomes [1,0,0,0], "Japan" becomes [0,1,0,0], "China" becomes [0,0,1,0] etc

tp.one_hot('country_visited')


 # For each person, reduce all columns using `sum` op, except for entry_time; reduce it using `max` op :
tp.reduce('person', 'sum', {'entry_time' : 'max'}) 

# Rename column
tp.rename_column('max(entry_time)', 'most_recent_entry')

# Execute save output to csv file
output = tp("reduction_example.csv")
output.save_to_csv("reduction_example_output.csv")
