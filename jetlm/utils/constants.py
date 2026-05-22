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

import numpy as np

DO_SYNC = False
ZERO_EPS = 1e-5
TRAINER_LOG_INTERVAL = 10

DTYPES = {
    "uint16": np.uint16,
    "uint32": np.uint32,
}

DATA_CACHE_DIR = "processed_data/data_cache"
MAX_DATA_STREAMING_REPEATS = 30
MAX_DATA_STREAMING_TOKENS = int(5 * 1024**4)  # 5T

ORANGE = "\033[38;5;208m" 
BLUE = "\033[38;5;27m"
RED = "\033[38;5;196m"
CYAN = "\033[38;5;51m"
DEBUG_RED = "\033[38;5;88m"
RESET = "\033[0m"