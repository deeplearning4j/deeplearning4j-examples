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

import jumpy as jp
import numpy as np


# You can convert numpy arrays to jumpy and vice versa

# Convert numpy to jumpy:


np_array = np.zeros((32, 100))
jp_array = jp.array(np_array)

# Convert jumpy to numpy:

jp_array = jp.zeros((32, 100))
np_array = jp_array.numpy()
