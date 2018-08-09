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

# You can use jumpy to run tensorflow models using nd4j from python

import jumpy as jp
import numpy as np


path_to_tf_model = 'Path to your TF model protobuff (binary or text)'

tf_model = jp.TFModel(path_to_tf_model)

# Now you can pass jumpy or numpy arrays to tf_model:

jp_array = jp.zeros((32, 28, 28))
output_array = tf_model(jp_array)

np_array = np.zeros((32, 28, 28))
output_array = tf_model(np_array)
