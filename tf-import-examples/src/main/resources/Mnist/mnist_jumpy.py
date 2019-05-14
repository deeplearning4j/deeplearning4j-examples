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

from scipy import ndimage
import jumpy as jp
import numpy as np
import os


path = os.path.dirname(os.path.abspath(__file__))

# load tensorflow model
tf_model = jp.TFModel(path + '/mnist.pb')

# load jpg to numpy array
image = ndimage.imread(path + '/img_1.jpg').reshape((1, 28, 28))

image = np.cast[float](image)

image /= 255.

# inference - uses nd4j
prediction = tf_model(image)  # prediction is a jumpy array

# get label from predction using argmax
label = jp.argmax(prediction.reshape((10,)))

print(label)
