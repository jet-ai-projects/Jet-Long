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

"""
Compat patches applied before running HELMET's eval.py.

Why this exists: HELMET's `data.py` relies on `load_dataset(..., trust_remote_code=True)`
for a handful of datasets whose HF repos still ship loading scripts. This was removed
in `datasets>=4.0`. Our `jtl` conda env has datasets 4.6.1, so without intervention those
loads raise `RuntimeError("Dataset scripts are no longer supported")`.

Each patch module declares:
  - `SUPPORTED`: a boolean / condition indicating if the patch is relevant
  - `apply()`:  monkey-patches HELMET's `data` module after it's importable

Patches are injected by running HELMET via our launcher's `patched_eval.py`
wrapper — see `helmet_shard.py`.
"""
from . import multi_lexsum_patch


def apply_all() -> list[str]:
    applied = []
    for mod in (multi_lexsum_patch,):
        if mod.apply():
            applied.append(mod.__name__.rsplit(".", 1)[-1])
    return applied
