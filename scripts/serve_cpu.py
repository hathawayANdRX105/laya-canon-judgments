#!/usr/bin/env python3
"""Start laya-serve with CPU fallback when no accelerator is available.

The pinned transformers queries torch.accelerator at import time and
raises when the CUDA driver is wedged or the box has no GPU at all; the
stub lets the import proceed so laya's own CUDA-unavailable fallback to
CPU takes over (laya/agent.py device resolution).
"""

import torch

torch.accelerator.current_accelerator = lambda: None

from laya.serve import main  # noqa: E402

main()
