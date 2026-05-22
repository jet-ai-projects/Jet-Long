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
Thin wrapper that applies our compat patches to HELMET's `data.py`, then runs
HELMET's `eval.py` with the remaining CLI args as if invoked directly.

Invoke from within `helmet_dir` so all relative paths resolve the same way
HELMET's upstream scripts expect. `helmet_shard.py` does exactly that.
"""
import os
import runpy
import sys


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    # Expose HELMET's own modules: we must be cwd=helmet_dir so `data.py` is importable.
    sys.path.insert(0, os.getcwd())

    import patches  # noqa: E402
    applied = patches.apply_all()
    if applied:
        print(f"[patched_eval] applied: {applied}", file=sys.stderr, flush=True)

    # Forward argv to eval.py: our own script name is already stripped by the caller
    # if invoked as `python patched_eval.py <eval.py args...>`, so argv is ready.
    eval_py = os.path.join(os.getcwd(), "eval.py")
    if not os.path.isfile(eval_py):
        print(f"[patched_eval] cannot find {eval_py} (cwd must be the HELMET repo)",
              file=sys.stderr)
        sys.exit(2)

    # runpy preserves sys.argv; set argv[0] to eval.py's path so logging/paths match.
    sys.argv = [eval_py] + sys.argv[1:]
    runpy.run_path(eval_py, run_name="__main__")


if __name__ == "__main__":
    main()
