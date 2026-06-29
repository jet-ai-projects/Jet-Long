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

import logging, os
from colorama import init, Fore, Style
from .dist import get_dist_rank, is_master

init(autoreset=True)

class ColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.CRITICAL: Fore.LIGHTYELLOW_EX + Style.BRIGHT,
        logging.ERROR: Fore.RED + Style.BRIGHT,
        logging.WARNING: Fore.LIGHTRED_EX,
        logging.INFO: Fore.WHITE,
        logging.DEBUG: Fore.CYAN,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, Fore.WHITE)

        # Save originals (important if multiple handlers format the same record)
        orig_levelname = record.levelname
        orig_msg = record.msg
        orig_args = record.args

        try:
            msg = record.getMessage()  # <- correct way; always exists

            record.levelname = f"{color}{orig_levelname}{Style.RESET_ALL}"
            record.msg = f"{color}{msg}{Style.RESET_ALL}"
            record.args = ()  # prevent logging from trying to %-format again

            return super().format(record)
        finally:
            # Restore so other handlers/formatters aren't affected
            record.levelname = orig_levelname
            record.msg = orig_msg
            record.args = orig_args

def rank_log_path(base_path: str) -> str:
    """
    base_path like: logs/train.log  -> logs/train.rank0003.log
    """
    r = get_dist_rank()
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".log"
    return f"{root}.rank{r:04d}{ext}"

def setup_logging(log_path, delete_existing=True, level=logging.DEBUG, log_to_file=False):
    """
    Set up a logger that logs to both console and a file. Assume the directory of log_path is already created.
    Critical -> Orange, Warning -> Light Red (console only)
    """

    logger = logging.getLogger("train.jetlm_le")
    logger.setLevel(level)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)
        try: h.close()
        except Exception: pass
    
    # Base format (no color codes here; apply them in ColorFormatter for console)
    base_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # File handler (no color)
    if log_to_file:
        assert log_path is not None, "log_to_file is set to True but log_path is None"
        log_path = rank_log_path(log_path)
        log_dir = os.path.dirname(log_path)
        if log_dir:  # only assert if a directory component exists
            assert os.path.isdir(log_dir), f"log_dir does not exist: {log_dir}, should be created when train_config initialized"
        if delete_existing and os.path.exists(log_path):
            os.remove(log_path)

        fh = logging.FileHandler(log_path, mode="a")  # writes to disk
        fh.setFormatter(logging.Formatter(base_format, datefmt=date_format))
        fh.setLevel(level)
        logger.addHandler(fh)


    # Console handler (with color)
    if is_master():
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColorFormatter(base_format, datefmt=date_format))
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger

def silence_logger(logger: logging.Logger):
    # remove any existing handlers (file/console) to avoid side effects
    logger.handlers.clear()
    # prevent bubbling up to root logger (which might have handlers)
    logger.propagate = False

    logger.setLevel(logging.CRITICAL + 1)