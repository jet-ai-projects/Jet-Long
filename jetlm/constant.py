# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# Repo-wide constants shared across the eval pipelines.

# Version string baked into the per-eval `.complete` marker. Bump it to force a
# variant to re-run past an existing marker without manually deleting files.
DEFAULT_VERSION_STR = "26_04_02_1238"
DEFAULT_EVAL_VERSION_STR = DEFAULT_VERSION_STR

# Suffix appended to W&B run hash when constructing the final-results run id.
DEFAULT_WANDB_RESULT_LOG_SUFFIX = "_results"
