# Copyright (c) 2024 Cloudera, Inc.

# This file is part of Chat with your doc AMP.

# Chat with your doc AMP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.

# Chat with your doc AMP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with Chat with your doc AMP. If not, see <https://www.gnu.org/licenses/>.
import requests
from bs4 import BeautifulSoup
import os
import urllib

# The URL to scrape
url = "https://docs.llamaindex.ai/en/stable/"

# The directory to store files in
output_dir = "./llamindex-docs/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Fetch the page
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Find all links to .html files
links = soup.find_all("a", href=True)

print(f"got links length = {len(links)}")
count = 1
for link in links:
    href = link["href"]

    # If it's a .html file

    # Make a full URL if necessary
    if not href.startswith("http"):
        href = urllib.parse.urljoin(url, href)

    # Fetch the .html file
    print(f"downloading {href}")
    try:
        file_response = requests.get(href)

        if (
            file_response.status_code == 200
            and "text/html" in file_response.headers["Content-Type"]
        ):
            # Write it to a file
            print(f"writing the data {href} to file with name {str(count) + '.html'}")
            file_name = os.path.join(output_dir, str(count) + ".html")
            with open(file_name, "w", encoding="utf-8") as file:
                file.write(file_response.text)
            count += 1
    except Exception as e:
        print("An unexpected error occurred:", e)
