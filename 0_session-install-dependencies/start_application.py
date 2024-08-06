# Copyright (c) 2024 Cloudera, Inc.

# This file is part of Chat with your doc AMP.

# Chat with your doc AMP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.

# Chat with your doc AMP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with Chat with your doc AMP. If not, see <https://www.gnu.org/licenses/>.
import os
import cmlapi
import json
import random
import string
from utils.check_dependency import check_gpu_enabled


def generate_random_string(length=5):
    return "".join(random.choices(string.ascii_lowercase, k=length))


client = cmlapi.default_client(
    url=os.getenv("CDSW_API_URL").replace("/api/v1", ""),
    cml_api_key=os.getenv("CDSW_APIV2_KEY"),
)
available_runtimes = client.list_runtimes(
    search_filter=json.dumps(
        {"kernel": "Python 3.11", "edition": "Nvidia GPU", "editor": "PBJ Workbench"}
    )
)
print(available_runtimes)

## Set available runtimes to the latest runtime in the environment (iterator is the number that begins with 0 and advances sequentially)
## The JOB_IMAGE_ML_RUNTIME variable stores the ML Runtime which will be used to launch the job
print(available_runtimes.runtimes[0])
print(available_runtimes.runtimes[0].image_identifier)
APP_IMAGE_ML_RUNTIME = available_runtimes.runtimes[0].image_identifier

## Store the ML Runtime for any future jobs in an environment variable so we don't have to do this step again
os.environ["APP_IMAGE_ML_RUNTIME"] = APP_IMAGE_ML_RUNTIME
project = client.get_project(project_id=os.getenv("CDSW_PROJECT_ID"))


if check_gpu_enabled() == False:
    print("Start AI Chat with your documents without GPU")
    application_request = cmlapi.CreateApplicationRequest(
        name="AI Chat with your documents",
        description="AI Chat with your documents",
        project_id=project.id,
        subdomain="ai-chat-with-doc" + generate_random_string(),
        script="3_app-run-python-script/app.py",
        cpu=6,
        memory=24,
        runtime_identifier=os.getenv("APP_IMAGE_ML_RUNTIME"),
        bypass_authentication=True,
        environment={"CML": "yes", "TOKENIZERS_PARALLELISM": "false"},
    )

else:
    print("Start AI Chat with your documents with GPU")
    application_request = cmlapi.CreateApplicationRequest(
        name="AI Chat with your documents",
        description="AI Chat with your documents",
        project_id=project.id,
        subdomain="ai-chat-with-doc" + generate_random_string(),
        script="3_app-run-python-script/app.py",
        cpu=2,
        memory=16,
        nvidia_gpu=1,
        runtime_identifier=os.getenv("APP_IMAGE_ML_RUNTIME"),
        bypass_authentication=True,
        environment={"CML": "yes", "TOKENIZERS_PARALLELISM": "false"},
    )

app = client.create_application(project_id=project.id, body=application_request)
