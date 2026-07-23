#!/usr/bin/env python3
"""Save machine and software specifications for the experiment section."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = (
    REPO_ROOT / "results/research_tables/runtime_machine_specifications.json"
)
OUTPUT_TEXT = (
    REPO_ROOT / "results/research_tables/runtime_machine_specifications.txt"
)


def command(args: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(
            list(args),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def main() -> None:
    data = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "git_commit": command(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]
        ),
        "git_branch": command(
            ["git", "-C", str(REPO_ROOT), "branch", "--show-current"]
        ),
    }

    try:
        import numpy as np
        data["numpy"] = np.__version__
    except Exception as error:
        data["numpy_error"] = str(error)

    try:
        import gurobipy as gp
        data["gurobipy"] = getattr(gp, "__version__", None)
        data["gurobi"] = ".".join(
            str(value) for value in gp.gurobi.version()
        )
    except Exception as error:
        data["gurobi_error"] = str(error)

    if platform.system() == "Darwin":
        data["cpu_brand"] = command(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        )
        memory = command(["sysctl", "-n", "hw.memsize"])
        if memory:
            try:
                data["memory_bytes"] = int(memory)
                data["memory_gib"] = round(int(memory) / (1024 ** 3), 3)
            except ValueError:
                data["memory_bytes"] = memory

        data["hardware_overview"] = command(
            ["system_profiler", "SPHardwareDataType"]
        )
        data["software_overview"] = command(
            ["system_profiler", "SPSoftwareDataType"]
        )

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "Runtime machine specifications",
        "==============================",
    ]
    for key, value in data.items():
        lines.append(f"{key}: {value}")

    OUTPUT_TEXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("JSON:", OUTPUT_JSON)
    print("Text:", OUTPUT_TEXT)


if __name__ == "__main__":
    main()
