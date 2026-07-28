# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LLNL/LBANN.
#
# SPDX-License-Identifier: (Apache-2.0)
"""
Guards on what importing the launcher's torch support is allowed to do.

Which GPUs a process can use is fixed by the ``*_VISIBLE_DEVICES``
environment variables, but only at the moment the CUDA/HIP runtime
initializes. Any accelerator call made while ``hpc_launcher.torch`` is being
imported therefore freezes the device list at whatever happened to be visible
then, and every later change to those variables silently does nothing.

That matters because narrowing a worker to its own GPU is exactly the kind of
thing a caller wants to do *after* import, once the local rank is known. If
the runtime is already initialized by then, every worker on the node keeps the
full device list, picks the same first entry, and the whole node computes on
one physical GPU -- with collectives failing on duplicate devices and that GPU
taking N times the memory pressure.

The import path is accelerator-free today, so these tests assert that
structural property directly rather than trying to observe the (hardware- and
vendor-dependent) pinning it would cause. Re-introducing an import-time probe
such as ``torch.cuda.is_available()`` fails them immediately.
"""
import json
import os
import subprocess
import sys

import pytest

from conftest import require_torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Imports the requested module with a recording stand-in installed in place of
# torch, then reports every torch attribute the import touched. Run in a
# subprocess so the stub can never leak into the importing test session.
_PROBE = '''
import importlib
import json
import sys
import types


class _Recorder(types.ModuleType):
    """A stand-in module that records every attribute access made on it."""

    def __init__(self, name, log):
        super().__init__(name)
        self.__dict__["_log"] = log
        # Present as a package so `import torch.<sub>` resolves.
        self.__path__ = []

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        self._log.append("{}.{}".format(self.__name__, item))
        return _Recorder("{}.{}".format(self.__name__, item), self._log)

    def __call__(self, *args, **kwargs):
        return None


log = []
for name in ("torch", "torch.distributed"):
    sys.modules[name] = _Recorder(name, log)

importlib.import_module(sys.argv[1])

with open(sys.argv[2], "w") as fh:
    json.dump(log, fh)
'''


def _torch_attributes_touched_by_import(module, tmp_path):
    """
    Import ``module`` in a subprocess with torch replaced by a recorder, and
    return the list of torch attributes the import touched.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(_PROBE)
    log_file = tmp_path / "touched.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    # A clean visibility env keeps the ROCR->HIP swap from printing warnings
    # that have nothing to do with what is being measured here.
    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES"):
        env.pop(var, None)

    proc = subprocess.run(
        [sys.executable, str(probe), module, str(log_file)],
        env=env,
        capture_output=True,
        universal_newlines=True,
    )
    assert proc.returncode == 0, (
        f"probing the import of {module} failed:\n{proc.stdout}\n{proc.stderr}"
    )
    return json.loads(log_file.read_text())


@pytest.mark.parametrize(
    "module",
    [
        "hpc_launcher.torch",
        "hpc_launcher.torch.torchrun_hpc_trampoline",
    ],
)
def test_import_does_not_touch_cuda(module, tmp_path):
    """
    Importing the module must not reach into ``torch.cuda`` at all.

    Deliberately stricter than "must not initialize the runtime": a bare
    ``torch.cuda.is_available()`` is enough to initialize HIP on a ROCm build,
    and whether it does so is not something a portable test can observe. So
    the whole namespace is off limits on the import path, and any accelerator
    work belongs in a function the caller invokes once the device is known.

    Using a recording stand-in for torch (rather than the real one) means this
    holds everywhere, including hosts with no torch and no GPU.
    """
    touched = _torch_attributes_touched_by_import(module, tmp_path)

    cuda_touches = [attr for attr in touched if ".cuda" in attr]
    assert not cuda_touches, (
        f"importing {module} touched {cuda_touches}; accelerator calls on the "
        "import path initialize the CUDA/HIP runtime before per-worker GPU "
        "visibility can be narrowed, which pins every worker to one GPU"
    )


def test_import_leaves_cuda_runtime_uninitialized(tmp_path):
    """
    The same guarantee, measured against the real torch: after importing the
    package the CUDA/HIP runtime must still be uninitialized, so a caller can
    change ``*_VISIBLE_DEVICES`` afterwards and have it take effect.

    On a host with no accelerator this can only ever pass, which is why the
    recorder-based test above is the portable guard; this one is what actually
    bites on a GPU node.
    """
    require_torch()

    script = tmp_path / "check_init.py"
    script.write_text(
        "import json\n"
        "import sys\n"
        "import hpc_launcher.torch\n"
        "import torch\n"
        "state = {'initialized': torch.cuda.is_initialized(),\n"
        "         'available': torch.cuda.is_available()}\n"
        "with open(sys.argv[1], 'w') as fh:\n"
        "    json.dump(state, fh)\n"
    )
    result_file = tmp_path / "state.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, str(script), str(result_file)],
        env=env,
        capture_output=True,
        universal_newlines=True,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"

    state = json.loads(result_file.read_text())
    assert state["initialized"] is False, (
        "importing hpc_launcher.torch initialized the CUDA/HIP runtime; any "
        "later narrowing of *_VISIBLE_DEVICES is then ignored"
    )
