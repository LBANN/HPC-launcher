# HPC-launcher `launch` Command Documentation

## Overview

The `launch` command is a versatile tool for launching distributed jobs on HPC clusters or cloud environments. It provides a unified interface across different schedulers (SLURM, LSF, Flux) and supports both interactive and batch job submission.

## Synopsis

```bash
launch [options] command [args...]
```

## Command Structure

```bash
launch [-h] [--verbose] [-N NODES] [-n PROCS_PER_NODE] [--gpus-per-proc GPUS_PER_PROC]
       [-q QUEUE] [-t TIME_LIMIT] [-g GPUS_AT_LEAST] [--gpumem-at-least GPUMEM_AT_LEAST]
       [--exclusive] [--local] [--comm-backend JOB_COMM_PROTOCOL]
       [-x KEY=VALUE [KEY=VALUE ...]] [--bg] [--batch-script BATCH_SCRIPT]
       [--scheduler {None,local,LocalScheduler,flux,FluxScheduler,slurm,SlurmScheduler,lsf,LSFScheduler}]
       [-l [LAUNCH_DIR]] [-o OUTPUT_SCRIPT] [--setup-only] [--dry-run]
       [--account ACCOUNT] [--dependency DEPENDENCY] [-J JOB_NAME]
       [--reservation RESERVATION] [--save-hostlist]
       [-p KEY=VALUE [KEY=VALUE ...]] [--out OUT_LOG_FILE] [--err ERR_LOG_FILE]
       [--color-stderr] command [args...]
```

## Positional Arguments

| Argument | Description |
|----------|-------------|
| `command` | Command to be executed |
| `args` | Arguments to the command that should be executed |

## Optional Arguments

### General Options

| Option | Short Form | Description |
|--------|------------|-------------|
| `--help` | `-h` | Show help message and exit |
| `--verbose` | `-v` | Run in verbose mode. Also save the hostlist as if `--save-hostlist` is set |

## Job Size Options

These options determine the number of nodes, accelerators, and ranks for the job.

| Option | Short Form | Description | Notes |
|--------|------------|-------------|-------|
| `--nodes` | `-N` | Specifies the number of requested nodes | |
| `--procs-per-node` | `-n` | Specifies the number of requested processes per node | Mutually exclusive with `-g` |
| `--gpus-per-proc` | | Specifies the number of requested GPUs per process | Default: 1 |
| `--queue` | `-q` | Specifies the queue to use | |
| `--time-limit` | `-t` | Set a time limit for the job in minutes | |
| `--gpus-at-least` | `-g` | Specifies the total number of accelerators requested | Mutually exclusive with `-n` and `-N` |
| `--gpumem-at-least` | | Constraint for accelerator memory needed (in GB) | System must be registered with launcher |
| `--exclusive` | | Request exclusive access from the scheduler | |
| `--local` | | Run locally (one process without batch scheduler) | |
| `--comm-backend` | | Indicate primary communication protocol | Options: MPI, *CCL (NCCL, RCCL) |
| `--xargs` | `-x` | Specify scheduler and launch arguments | Format: `KEY=VALUE` |

### Notes on `--xargs`:
- Will override any known key
- Use format: `--xargs k1=v1 k2=v2` or `--xargs k1=v1 --xargs k2=v2`
- Double dash `--` needed if this is the last argument
- Arguments with leading tilde `~` will be removed if found

## Schedule Options

Arguments that determine when a job will run.

| Option | Description | Notes |
|--------|-------------|-------|
| `--bg` | Run job in background | Launcher won't wait for job start; uses timestamped directory by default |
| `--batch-script` | Launch a user-provided batch script | |
| `--scheduler` | Override default batch scheduler | Options: None, local, LocalScheduler, flux, FluxScheduler, slurm, SlurmScheduler, lsf, LSFScheduler |

## Script Options

Batch scheduler script parameters.

| Option | Short Form | Description | Notes |
|--------|------------|-------------|-------|
| `--launch-dir` | `-l` | Control launch directory creation | See detailed behavior below |
| `--output-script` | `-o` | Output job setup script file | Uses temporary file if not specified |
| `--setup-only` | | Only write job setup script without scheduling | |
| `--dry-run` | | Output results without side-effects | |
| `--account` | | Specify account/bank for the job | |
| `--dependency` | | Specify scheduler dependency | |
| `--job-name` | `-J` | Specify job name | |
| `--reservation` | | Add reservation argument | Typically for DAT runs |
| `--save-hostlist` | | Write hostlist to `hpc_launcher_hostlist.txt` | |

### `--launch-dir` Behavior:
- **No argument**: Creates timestamped launch directory
- **With argument**: Creates directory named `[LAUNCH_DIR]`
- **Argument = "."**: Creates launch script in current directory
- **Not set + blocking job**: Runs without creating files
- **Not set + non-blocking job**: Creates launch file and logs in current directory
- **Note**: Double dash `--` needed if this is the last argument

## System Options

Provide system parameters from CLI - overrides built-in system descriptions and autodetection.

| Option | Short Form | Description | Format |
|--------|------------|-------------|--------|
| `--system-params` | `-p` | Specify system parameters | `KEY=VALUE` pairs |

### System Parameter Examples:
```bash
-p cores_per_node=128 gpus_per_node=8 gpu_arch=ampere mem_per_gpu=80 numa_domains=4 scheduler=slurm
```

Available parameters:
- `cores_per_node`: Integer value for CPU cores per node
- `gpus_per_node`: Integer value for GPUs per node
- `gpu_arch`: String value for GPU architecture
- `mem_per_gpu`: Float value for memory per GPU
- `numa_domains`: Integer value for NUMA domains
- `scheduler`: String value for scheduler type

**Note**: Double dash `--` needed if this is the last argument

## Logging Options

Control output and error logging.

| Option | Description |
|--------|-------------|
| `--out` | Capture standard output to a log file (console only if not specified) |
| `--err` | Capture standard error to a log file (console only if not specified) |
| `--color-stderr` | Use terminal colors to color stderr in red (doesn't affect output files) |

## Usage Examples

### Basic Examples

```bash
# Simple single-node job
launch -N 1 -n 1 hostname

# Multi-node MPI job
launch -N 4 -n 2 ./mpi_application

# GPU job with 2 GPUs per process
launch -N 2 -n 2 --gpus-per-proc 2 ./gpu_application
```

### Resource Specification Examples

```bash
# Request specific number of GPUs total
launch -g 16 ./gpu_application

# Request allocation with at least 80GB GPU memory
launch --gpumem-at-least 80 ./memory_intensive_app

# Exclusive node access
launch -N 2 --exclusive ./exclusive_app

# Local execution without scheduler
launch -N 1 --local ./test_script.py
```

### Job Scheduling Examples

```bash
# Submit to specific queue with time limit
launch -q gpu_queue -t 120 -N 2 ./training_script.py

# Background job with custom name
launch --bg -J my_experiment -N 4 ./long_running_job

# Job with dependencies
launch --dependency afterok:12345 -N 1 ./dependent_job

# Use specific account/bank
launch --account project123 -N 2 ./billable_job
```

### Communication Backend Examples

```bash
# MPI backend
launch --comm-backend MPI -N 4 -n 4 ./mpi_app

# NCCL backend for GPU communication
launch --comm-backend NCCL -N 2 -n 2 --gpus-per-proc 2 ./gpu_training
```

### Script and Directory Management

```bash
# Create timestamped launch directory
launch -l -N 2 ./my_job

# Use specific launch directory
launch -l experiment_001 -N 2 ./my_job

# Run in current directory
launch -l . -N 2 ./my_job

# Generate script without running
launch --setup-only -l -o job_script.sh -N 4 ./my_application

# Dry run to see what would be executed
launch --dry-run -N 4 -n 8 ./my_application
```

### System Override Examples

```bash
# Override system detection
launch -p cores_per_node=64 gpus_per_node=4 -N 2 ./custom_job

# Multiple system parameters
launch -p gpu_arch=v100 mem_per_gpu=32 scheduler=slurm -N 2 ./gpu_job
```

### Scheduler-Specific Arguments

```bash
# Pass additional scheduler arguments
launch -x partition=debug -x qos=high -N 2 ./debug_job

# Remove specific arguments (leading tilde)
launch -x ~constraint -N 2 ./unconstrained_job

# Multiple xargs
launch --xargs key1=val1 --xargs key2=val2 -N 2 ./job
```

### Logging Examples

```bash
# Capture output and error to files
launch --out output.log --err error.log -N 2 ./my_job

# Colored error output in terminal
launch --color-stderr -N 1 ./verbose_job

# Verbose mode with saved hostlist
launch --verbose --save-hostlist -N 4 ./debug_job
```

### Complex Example

```bash
# Production job with all options
launch \
  --verbose \
  -N 8 \
  -n 4 \
  --gpus-per-proc 2 \
  -q production \
  -t 480 \
  --exclusive \
  --comm-backend NCCL \
  --bg \
  --scheduler slurm \
  -l production_run_001 \
  --account ml_project \
  -J "ResNet Training" \
  --save-hostlist \
  --out logs/output.log \
  --err logs/error.log \
  python train_resnet.py --epochs 100 --batch-size 256
```

## Environment Variables

The `launch` command may set or use various environment variables depending on the scheduler and communication backend:

- For MPI jobs: Standard MPI environment variables
- For GPU jobs: CUDA-related environment variables
- For NCCL/RCCL: Communication library environment variables

## Exit Status

- `0`: Successful execution
- Non-zero: Error occurred (specific codes depend on failure type)

## Tips and Best Practices

1. **Use `--dry-run`** first to verify your command before submitting large jobs
2. **Use `--verbose`** for debugging to see detailed information about job submission
3. **Save scripts with `--setup-only`** to review and reuse job configurations
4. **Use timestamped directories** (`-l` without argument) for experiment tracking
5. **Specify `--account`** for proper resource accounting on shared systems
6. **Use `--save-hostlist`** when you need to know which nodes were allocated
7. **Set appropriate time limits** with `-t` to avoid jobs being killed prematurely
8. **Use `--exclusive`** for performance-critical jobs to avoid interference

## Scheduler Detection

The launcher automatically detects the available scheduler. Override with `--scheduler` if needed:
- `local`: Run without a scheduler
- `slurm`: SLURM workload manager
- `lsf`: IBM Spectrum LSF
- `flux`: Flux resource manager

## See Also

- `torchrun-hpc` - PyTorch-specific distributed training launcher
- HPC-launcher documentation: https://github.com/LBANN/HPC-launcher
- LBANN documentation: https://lbann.readthedocs.io

---

*Generated from `launch -h` output*
