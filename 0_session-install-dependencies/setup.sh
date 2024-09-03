#!/bin/bash
# Copyright (c) 2024 Cloudera, Inc.

# This file is part of Chat with your doc AMP.

# Chat with your doc AMP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.

# Chat with your doc AMP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with Chat with your doc AMP. If not, see <https://www.gnu.org/licenses/>.
pip install -r 0_session-install-dependencies/requirements.txt
python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('punkt_tab')"