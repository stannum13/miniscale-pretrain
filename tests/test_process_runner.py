import subprocess
import sys

import pytest

from scripts.process_runner import run_process_group


def test_process_group_runner_times_out() -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        run_process_group(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            timeout_seconds=0.1,
        )
