# Copyright (c) 2024 Cloudera, Inc.

# This file is part of Chat with your doc AMP.

# Chat with your doc AMP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.

# Chat with your doc AMP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with Chat with your doc AMP. If not, see <https://www.gnu.org/licenses/>.

import sys
from utils.check_dependency import check_unauthenticated_access_to_app_enabled

if check_unauthenticated_access_to_app_enabled() == False:
    sys.exit(
        "Please enable 'Allow applications to be configured with unauthenticated access' from security tab "
    )
