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

from jumpy import op, array
import jumpy as jp

# Use @op decorator to add a new jumpy op. 
# The function will receive an INDArray object, regardless of what it's called with : INDArray, Jumpy array or numpy array.
# The value returned by the function will normalized to Jumpy array.



# Op to element-wise compute x + y - z

@op
def my_op(x, y, z):
    # x, y, z are INDArrays
    return x.add(y).sub(z)  # this will be automatically converted to jumpy array


# Nd4j does not have broadcasting yet. If you need broadcasting in your op,
# convert to jumpy array and use jumpy's broadcasting

# op to element-wise compute 3x + 4y

@op
def my_broadcast_op(x, y):
    # x and y are INDArrays
    x = array(x)  # x is now a jumpy array
    y = array(y)  # y is now a jumpy array
    return 3 * x + 4 * y
