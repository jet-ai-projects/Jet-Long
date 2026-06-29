# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import io, json, os, glob, wandb, tempfile, portalocker, torch, shutil
from pathlib import Path

from typing import Any, Mapping, Optional
from .dist import is_master, dist_barrier


def _fsync_dir(dirpath: str) -> None:
    """Durability: ensure the directory entry for the replaced file is flushed."""
    try:
        fd = os.open(dirpath, os.O_DIRECTORY)
    except (AttributeError, FileNotFoundError, NotADirectoryError):
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: str, obj: Any, *, indent: Optional[int] = None) -> None:
    """
    Write JSON atomically (temp file -> fsync -> replace) so readers never see partial content.
    """
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=d, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", closefd=True) as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())  # ensure file bytes hit disk
        os.replace(tmp, path)  # atomic on POSIX & Windows
        _fsync_dir(d)  # ensure directory entry is durable (POSIX)
    finally:
        # If replace succeeded, tmp is gone; if not, try to clean it up.
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


def locked_atomic_write_json(
    path: str, obj: Any, *, timeout: float = 10.0, indent: Optional[int] = None
) -> None:
    """
    Take an exclusive lock on a dedicated lock file, then perform an atomic write.
    """
    lock_path = path + ".lock"
    # 'a+' ensures the lock file exists on first use; EX gives mutual exclusion.
    with portalocker.Lock(
        lock_path, mode="a+", flags=portalocker.LOCK_EX, timeout=timeout
    ):
        atomic_write_json(path, obj, indent=indent)


def locked_read_json(
    path: str, default: Optional[Mapping[str, Any]] = None, *, timeout: float = 20.0
) -> Mapping[str, Any]:
    """
    Take a shared lock on the lock file and read JSON safely.
    Returns `default` (or {}) if file is missing/invalid.
    """
    lock_path = path + ".lock"
    try:
        with portalocker.Lock(
            lock_path, mode="a+", flags=portalocker.LOCK_SH, timeout=timeout
        ):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
    except FileNotFoundError:
        return {} if default is None else default
    except json.JSONDecodeError:
        # If some external writer isn’t using our atomic+lock discipline, fall back gracefully.
        return {} if default is None else default

    # Optional: enforce dict output
    if isinstance(data, dict):
        return data
    raise ValueError("json dict failure load, a sync bug happened")


import time, json, hashlib, wandb


def log_and_flush(wandb_run, length, log_dict):
    # 1) Log metrics (async)
    wandb_run.log(log_dict, step=length, commit=True)

    # 2) Barrier marker: one artifact per run, metadata accumulates
    marker_meta = {
        "run_id": wandb_run.id,
        "step": length,
        "payload_hash": hashlib.sha1(
            json.dumps(log_dict, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "ts": time.time(),
    }

    art = wandb.Artifact(
        name=f"marker-run-{wandb_run.id}",
        type="marker",
        metadata={str(length): marker_meta},
    )

    # reuse the same name each step, so you just get new versions
    wandb_run.log_artifact(art, aliases=["latest", f"upto-{length}"]).wait()


REQUIRED_FILES = [
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state_0.pth",
]  # plus model file
MARKER_NAME = ".complete"  # written last
CKPT_NAME_FORMAT = "checkpoint-{step}"
CKPT_NAME_FORMAT_REGEX = CKPT_NAME_FORMAT.format(step="*")
DATALOADER_STATE_FNAME = "streaming_offsets.pt"


def is_valid_checkpoint(ckpt_dir: str) -> bool:
    """Check presence of required files and that trainer_state.json is readable."""
    d = Path(ckpt_dir)
    if not d.is_dir():
        return False
    # model can be .bin or .safetensors (or sharded)
    has_model = any(
        (d / f).exists() for f in ["pytorch_model.bin", "model.safetensors"]
    )
    if (
        not has_model
        and not glob.glob(str(d / "*model*.bin"))
        and not glob.glob(str(d / "*model*.safetensors"))
    ):
        return False
    for f in REQUIRED_FILES:
        if not (d / f).exists():
            return False
    # sanity-check JSON isn’t truncated
    try:
        with open(d / "trainer_state.json", "r", encoding="utf-8") as fh:
            json.load(fh)
    except Exception:
        return False
    # final marker written last
    if not (d / MARKER_NAME).exists():
        return False
    return True


def find_last_complete_checkpoint(output_dir: str) -> str | None:
    """Return newest checkpoint with a completion marker; ignore partial ones."""
    if not os.path.isdir(output_dir):
        return None
    # checkpoints are usually named checkpoint-<global_step>
    cks = sorted(
        [p for p in Path(output_dir).glob(CKPT_NAME_FORMAT_REGEX) if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
        reverse=True,
    )
    for p in cks:
        if is_valid_checkpoint(str(p)):
            return str(p)
    return None  # none valid


def find_last_complete_checkpoint_fsdp2(
    output_dir: str, version_str: str
) -> str | None:
    """Return newest checkpoint with a completion marker; ignore partial ones."""
    if not os.path.isdir(output_dir):
        return None
    # checkpoints are usually named checkpoint-<global_step>
    cks = sorted(
        [p for p in Path(output_dir).glob(CKPT_NAME_FORMAT_REGEX) if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
        reverse=True,
    )
    for p in cks:
        if check_complete_marker(str(p), version_str=version_str):
            return str(p)
    return None  # none valid


def get_complete_marker_path(path: str) -> str:
    """
    Returns the path to the completion marker.
    If path is a directory, returns path/MARKER_NAME.
    If path is a file, returns path with extension replaced by MARKER_NAME.
    """
    if os.path.isdir(path):
        return os.path.join(path, MARKER_NAME)
    else:
        # It is a file: replace extension
        base, _ = os.path.splitext(path)
        return base + MARKER_NAME


def create_complete_marker(path: str, version_str: int) -> str:
    complete_path = get_complete_marker_path(path=path)

    # Ensure directory exists
    os.makedirs(os.path.dirname(complete_path), exist_ok=True)

    # Create/overwrite the marker file with version id
    with open(complete_path, "w") as f:
        f.write(str(version_str))

    return complete_path


def create_complete_marker_atomic_swap(path: str, version_str: int) -> str:
    complete_path = get_complete_marker_path(path=path)
    target_dir = os.path.dirname(complete_path)
    # Ensure directory exists
    os.makedirs(target_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", dir=target_dir, delete=False) as tf:
        try:
            tf.write(str(version_str))
            tf.flush()
            os.fsync(tf.fileno())
            temp_name = tf.name
        except Exception:
            tf.close()
            if os.path.exists(tf.name):
                os.remove(tf.name)
            raise
    # Perform the atomic swap
    try:
        os.replace(temp_name, complete_path)
    except OSError:
        if os.path.exists(temp_name):
            os.remove(temp_name)
        raise

    return complete_path


def all_check_complete_marker(complete_path, version_str) -> bool:
    if os.path.exists(complete_path):
        with open(complete_path, "r") as f:
            existing = f.read().strip()
        if existing == str(version_str):
            return True  # matches, nothing to do
        else:
            # Version mismatch detected
            return False
    else:
        return False


def check_complete_marker(path: str, version_str: int, clean_dir: bool = True) -> bool:
    """
    Returns True if the marker exists and matches the version_str.

    If it exists but mismatches:
      - If clean_dir is True: Removes the entire directory at 'path' and recreates it.
      - If clean_dir is False: Removes only the marker file.
      - Returns False.

    If it doesn't exist, return False.
    """
    complete_path = get_complete_marker_path(path=path)
    dir_to_clean = os.path.dirname(complete_path)

    check_passed = all_check_complete_marker(
        complete_path=complete_path, version_str=version_str
    )

    dist_barrier()
    if not check_passed and clean_dir and is_master():
        if os.path.isdir(dir_to_clean):
            # Remove the entire directory tree and recreate the empty folder
            shutil.rmtree(dir_to_clean)
            os.mkdir(dir_to_clean)
    dist_barrier()
    return check_passed


REQUIRED_FSDP_FILES = (
    "pytorch_model.bin",  # FULL state dict saved on rank0
    "optimizer.pt",  # consolidated optimizer state (FSDP.optim_state_dict)
    "trainer_state.pt",  # torch-saved dict: scheduler, scaler, step, RNG states
)


def is_valid_fsdp_checkpoint(ckpt_dir: str) -> bool:
    """
    Validate an FSDP checkpoint produced by our trainer:
      - directory exists
      - rank-0 artifacts present: pytorch_model.bin, optimizer.pt, trainer_state.pt
      - trainer_state.pt is loadable and contains expected keys
      - atomic completion marker (MARKER_NAME) exists
    """
    d = Path(ckpt_dir)
    if not d.is_dir():
        return False

    # Model file: we save FULL state on rank0 as pytorch_model.bin (allow .safetensors optionally)
    has_model = (d / "pytorch_model.bin").exists() or (d / "model.safetensors").exists()
    if not has_model:
        # also accept custom names like *model*.bin / *model*.safetensors just in case
        if not glob.glob(str(d / "*model*.bin")) and not glob.glob(
            str(d / "*model*.safetensors")
        ):
            return False

    # Required rank-0 artifacts
    for f in REQUIRED_FSDP_FILES:
        if not (d / f).exists():
            return False

    # trainer_state.pt must be readable with torch.load and contain some expected keys
    try:
        tstate = torch.load(d / "trainer_state.pt", map_location="cpu")
        # sanity checks (these keys are created by our trainer)
        if not isinstance(tstate, dict):
            return False
        if "step" not in tstate:
            return False
        if "lr_scheduler" not in tstate:
            return False
        if "scaler" not in tstate:
            return False
        # optional: check RNG container
        if "random_states" in tstate and not isinstance(tstate["random_states"], dict):
            return False
    except Exception:
        return False

    # Completion marker written last
    if not (d / MARKER_NAME).exists():
        return False

    return True


def find_last_complete_fsdp_checkpoint(output_dir: str) -> str | None:
    """
    Return newest checkpoint directory that passes is_valid_checkpoint.
    We assume checkpoints are named 'checkpoint-<global_step>'.
    """
    if not os.path.isdir(output_dir):
        return None

    cks = sorted(
        [p for p in Path(output_dir).glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
        reverse=True,
    )
    for p in cks:
        print(p, is_valid_checkpoint(str(p)))
        if is_valid_fsdp_checkpoint(str(p)):
            return str(p)
    return None
