#!/bin/sh

#
#
# This program and the accompanying materials are made available under the
#  terms of the Apache License, Version 2.0 which is available at
#  https://www.apache.org/licenses/LICENSE-2.0.
See the NOTICE file distributed with this work for additional
information regarding copyright ownership.
#
#   See the NOTICE file distributed with this work for additional
#   information regarding copyright ownership.
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  License for the specific language governing permissions and limitations
#  under the License.
#
#  SPDX-License-Identifier: Apache-2.0
#
#
#

cd /tmp

wget http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip
unzip cornell_movie_dialogs_corpus.zip "cornell movie-dialogs corpus/movie_lines.txt"
mv "cornell movie-dialogs corpus/movie_lines.txt" movie_lines_unsorted.txt
sed -i -e 's#^L##' -e 's#--#…#g' -e 's#\.\.\.#…#g' movie_lines_unsorted.txt
sort -n movie_lines_unsorted.txt > movie_lines.txt
rm movie_lines_unsorted.txt
