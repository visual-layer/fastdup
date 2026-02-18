"""
Utilities module for fastdup - platform detection, version checking, and shared helpers.
"""
import os
import sys
import platform

from fastdup.sentry import fastdup_capture_exception
from fastdup.definitions import (
    DEFAULT_MODEL_FEATURE_WIDTH,
    HIGH_ACCURACY_MODEL_FEATURE_WIDTH,
)

# Re-export everything from utils so that `from fastdup.utilities import X`
# works for functions defined in utils.py.
from fastdup.utils import *  # noqa: F401,F403
from fastdup.utils import find_model_path as _find_model_path_base

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_URL = "https://github.com/visual-layer/fastdup/issues"

_GETTING_STARTED_LINK = "https://visual-layer.readme.io/docs/getting-started"

_DOC_MSG = (
    "fastdup is a tool for managing, cleaning and curating visual data - "
    "finding duplicates, near-duplicates, outliers, clusters, and more. "
    "See https://visual-layer.readme.io for full documentation."
)

_POST_RUN_MSG = (
    "========================================================\n"
    "FastDup run completed successfully.\n"
    "Next steps:\n"
    "  - Explore results with fastdup galleries\n"
    "  - Visit https://visual-layer.readme.io for documentation\n"
    "========================================================"
)

_create_template_msg = lambda work_dir, input_dir: (
    "========================================================\n"
    f"FastDup work_dir: {work_dir}\n"
    f"FastDup input_dir: {input_dir}\n"
    "========================================================"
)


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

def is_macos_intel():
    """Return True when running on macOS with an Intel (x86_64) CPU."""
    return platform.system() == "Darwin" and platform.machine() == "x86_64"


def is_macos_arm64():
    """Return True when running on macOS with Apple Silicon (arm64)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def is_python_3_8():
    """Return True when the interpreter is Python 3.8.x."""
    return sys.version_info[:2] == (3, 8)


# ---------------------------------------------------------------------------
# Version check
# ---------------------------------------------------------------------------

def dcheck_latest_version(curversion):
    """Check whether the installed version is outdated (delegates to
    ``check_latest_version`` from *utils*)."""
    try:
        from fastdup.utils import check_latest_version
        return check_latest_version(curversion)
    except Exception as e:
        fastdup_capture_exception("dcheck_latest_version", e, True)
        return False


# ---------------------------------------------------------------------------
# Model path resolution (extended to accept high_accuracy)
# ---------------------------------------------------------------------------

def find_model_path(model_path, d, high_accuracy=False):
    """Resolve a human-friendly model name to an actual path and dimension.

    This extends the base ``find_model_path`` from *utils.py* by accepting
    the *high_accuracy* flag.  When *high_accuracy* is ``True`` and the
    caller has not supplied a custom model, the higher-accuracy variant of
    the default fastdup model is selected.
    """
    # First resolve any named model shortcuts (dinov2s, clip, …)
    model_path, d = _find_model_path_base(model_path, d)

    if high_accuracy:
        # Only adjust d when the caller has not already set a custom value.
        if d == DEFAULT_MODEL_FEATURE_WIDTH:
            d = HIGH_ACCURACY_MODEL_FEATURE_WIDTH

    return model_path, d
