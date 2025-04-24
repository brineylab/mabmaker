# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import os
from contextlib import redirect_stderr, redirect_stdout

# suppress warnings, which are crazy for Chai-1
with open(os.devnull, "w") as devnull:
    with redirect_stdout(devnull), redirect_stderr(devnull):
        from . import tl, tools, utils
