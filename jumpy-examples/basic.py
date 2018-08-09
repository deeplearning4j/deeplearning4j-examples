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

import jumpy as jp

# Basic example


# Create an array:

x = jp.zeros((32, 10, 12))

# Reshape:

x = jp.reshape(x, (32, 120))

# Reduction ops:

sum_ = x.sum(axis=1)
mean = x.mean(axis=1)
std = x.std(axis=1)
max_ = x.max(axis=0)
min_ = x.min(axis=0)

# Inplace ops:

y = jp.ones((32, 120))
x += y

# Broadcasting:

x = y * 2 + 3
