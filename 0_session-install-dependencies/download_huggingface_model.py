# Copyright (c) 2024 Cloudera, Inc.

# This file is part of Chat with your doc AMP.

# Chat with your doc AMP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.

# Chat with your doc AMP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with Chat with your doc AMP. If not, see <https://www.gnu.org/licenses/>.

from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
from utils.common import supported_llm_models, supported_embed_models

MODELS_PATH = "./models"
EMBEDS_PATH = "./embed_models"

for supported_llm_model in supported_llm_models:
    print(
        f"download model {supported_llm_model} file {supported_llm_models[supported_llm_model]}"
    )
    hf_hub_download(
        repo_id=supported_llm_model,
        filename=supported_llm_models[supported_llm_model],
        resume_download=True,
        cache_dir=MODELS_PATH,
        local_files_only=False,
    )


for embed_model in supported_embed_models:
    print(f"download embed model {embed_model}")
    snapshot_download(
        repo_id=embed_model,
        resume_download=True,
        cache_dir=EMBEDS_PATH,
        local_files_only=False,
    )
