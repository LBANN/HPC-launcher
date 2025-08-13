# HPC-launcher `torchrun-hpc` Command Documentation

## Overview

The `torchrun-hpc` command is a wrapper script that launches and runs distributed PyTorch on HPC systems. It provides HPC-optimized functionality for PyTorch distributed training across various schedulers (SLURM, LSF, Flux) and handles the complexities of multi-node, multi-GPU training setups.

## Synopsis

```bash
torchrun-hpc [options] command [args...]
```

## Command Structure

```bash
torchrun-hpc [-h] [--verbose] [-N NODES] [-n PROCS_PER_NODE] [--gpus-per-proc GPUS_PER_PROC]
             [-q QUEUE] [-t TIME_LIMIT] [-g GPUS_AT_LEAST] [--gpumem-at-least GPUMEM_AT_LEAST]
             [--exclusive] [--local] [--comm-backend JOB_COMM_PROTOCOL]
             [-x KEY=VALUE [KEY=VALUE ...]] [--bg] [--batch-script BATCH_SCRIPT]
             [--scheduler {None,local,LocalScheduler,flux,FluxScheduler,slurm,SlurmScheduler,lsf,LSFScheduler}]
             [-l [LAUNCH_DIR]] [-o OUTPUT_SCRIPT] [--setup-only] [--dry-run]
             [--account ACCOUNT] [--dependency DEPENDENCY] [-J JOB_NAME]
             [--reservation RESERVATION] [--save-hostlist]
             [-p KEY=VALUE [KEY=VALUE ...]] [--out OUT_LOG_FILE] [--err ERR_LOG_FILE]
             [--color-stderr] [-r RDV] [--fraction-max-gpu-mem FRACTION_MAX_GPU_MEM]
             [-u] command [args...]
```

## Positional Arguments

| Argument | Description |
|----------|-------------|
| `command` | Command to be executed (typically a Python script) |
| `args` | Arguments to pass to the command |

## Optional Arguments

### General Options

| Option | Short Form | Description |
|--------|------------|-------------|
| `--help` | `-h` | Show help message and exit |
| `--verbose` | `-v` | Run in verbose mode. Also save the hostlist as if `--save-hostlist` is set |

### PyTorch-Specific Options

| Option | Short Form | Description | Values |
|--------|------------|-------------|--------|
| `--rdv` | `-r` | Specifies rendezvous protocol to use | `mpi` \| `tcp` |
| `--fraction-max-gpu-mem` | | Use `torch.cuda.set_per_process_memory_fraction` to limit GPU memory allocation | Float (0.0-1.0) |
| `--unswap-rocr-hip-vis-dev` | `-u` | Undo moving ROCR_VISIBLE_DEVICES into HIP_VISIBLE_DEVICES env variable | Flag |

#### Notes on PyTorch Options:
- **Rendezvous (`--rdv`)**: Controls how distributed processes discover and connect to each other
  - `mpi`: Use MPI for rendezvous (good for HPC environments)
  - `tcp`: Use TCP/IP for rendezvous (standard PyTorch default)
- **GPU Memory Fraction**: Useful for preventing OOM errors or sharing GPUs
- **AMD GPU Support**: The `-u` flag improves behavior with HuggingFace Accelerate and TorchTitan on AMD GPUs

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

### Basic PyTorch Training

```bash
# Single node, 4 GPUs
torchrun-hpc -N 1 -n 4 train.py --epochs 100

# Multi-node training (2 nodes, 8 GPUs each)
torchrun-hpc -N 2 -n 8 train.py --batch-size 256

# Local testing without scheduler
torchrun-hpc --local -n 2 test_script.py
```

### Rendezvous Configuration

```bash
# MPI rendezvous (recommended for HPC)
torchrun-hpc -r mpi -N 4 -n 8 train.py

# TCP rendezvous (standard PyTorch)
torchrun-hpc -r tcp -N 2 -n 4 train.py

# TCP is useful for cloud environments or mixed networks
torchrun-hpc --rdv tcp -N 4 -n 8 cloud_train.py
```

### GPU Memory Management

```bash
# Limit each process to 80% of GPU memory
torchrun-hpc --fraction-max-gpu-mem 0.8 -N 2 -n 8 train.py

# Useful for avoiding OOM errors
torchrun-hpc --fraction-max-gpu-mem 0.75 -N 1 -n 4 large_model.py

# Share GPUs between multiple jobs
torchrun-hpc --fraction-max-gpu-mem 0.5 -n 2 shared_gpu_train.py
```

### AMD GPU Support

```bash
# For AMD GPUs with HuggingFace Accelerate
torchrun-hpc -u -N 2 -n 8 accelerate_train.py

# TorchTitan on AMD GPUs
torchrun-hpc --unswap-rocr-hip-vis-dev -N 4 -n 8 torchtitan_train.py
```

### Resource Specification

```bash
# Request specific total GPU count
torchrun-hpc -g 32 distributed_train.py

# Request GPUs with minimum memory
torchrun-hpc --gpumem-at-least 80 -n 8 large_model_train.py

# Exclusive node access for performance
torchrun-hpc --exclusive -N 4 -n 8 performance_critical.py

# Multiple GPUs per process (model parallel)
torchrun-hpc -N 2 -n 2 --gpus-per-proc 4 model_parallel_train.py
```

### Job Scheduling

```bash
# Submit to specific queue with time limit
torchrun-hpc -q gpu_queue -t 480 -N 4 -n 8 long_train.py

# Background job with custom name
torchrun-hpc --bg -J "BERT_finetune" -N 2 -n 8 bert_train.py

# Job with dependencies
torchrun-hpc --dependency afterok:12345 -N 2 -n 4 continue_train.py

# Use specific account
torchrun-hpc --account ml_research -N 8 -n 8 research_train.py

# DAT reservation
torchrun-hpc --reservation dat_2024 -N 16 -n 8 dat_experiment.py
```

### Communication Backends

```bash
# NCCL for NVIDIA GPUs
torchrun-hpc --comm-backend NCCL -N 4 -n 8 nvidia_train.py

# RCCL for AMD GPUs
torchrun-hpc --comm-backend RCCL -N 4 -n 8 amd_train.py

# MPI backend
torchrun-hpc --comm-backend MPI -N 8 -n 4 mpi_train.py
```

### Script Management

```bash
# Generate script without running
torchrun-hpc --setup-only -o torch_job.sh -N 4 -n 8 train.py

# Dry run to preview
torchrun-hpc --dry-run -N 8 -n 8 train.py --lr 0.001

# Custom launch directory
torchrun-hpc -l experiment_001 -N 2 -n 8 experiment.py

# Save hostlist for debugging
torchrun-hpc --save-hostlist -N 4 -n 8 debug_train.py
```

### System Overrides

```bash
# Override GPU detection
torchrun-hpc -p gpus_per_node=8 gpu_arch=a100 -N 2 train.py

# Custom system configuration
torchrun-hpc -p cores_per_node=128 mem_per_gpu=80 -N 4 -n 8 custom_train.py

# Force specific scheduler
torchrun-hpc --scheduler slurm -N 2 -n 8 train.py
```

### Logging Configuration

```bash
# Separate output and error logs
torchrun-hpc --out output.log --err error.log -N 2 -n 8 train.py

# Colored error output for debugging
torchrun-hpc --color-stderr --verbose -N 1 -n 4 debug_train.py

# Full logging setup
torchrun-hpc \
  --verbose \
  --save-hostlist \
  --out logs/train_out.log \
  --err logs/train_err.log \
  -N 4 -n 8 train.py
```

### Complex Production Example

```bash
# Full production training setup
torchrun-hpc \
  --verbose \
  -N 16 \
  -n 8 \
  --gpus-per-proc 1 \
  -r mpi \
  --fraction-max-gpu-mem 0.9 \
  --comm-backend NCCL \
  -q production \
  -t 1440 \
  --exclusive \
  --bg \
  -l production_run_$(date +%Y%m%d_%H%M%S) \
  --account ai_training \
  -J "GPT_Training" \
  --save-hostlist \
  -p gpu_arch=a100 mem_per_gpu=80 \
  --out logs/gpt_out.log \
  --err logs/gpt_err.log \
  train_gpt.py \
    --model-size 13B \
    --batch-size 2048 \
    --learning-rate 1e-4 \
    --warmup-steps 1000 \
    --max-steps 100000
```

## Environment Variables Set by torchrun-hpc

The command sets standard PyTorch distributed environment variables:

| Variable | Description |
|----------|-------------|
| `WORLD_SIZE` | Total number of processes |
| `RANK` | Global rank of the process |
| `LOCAL_RANK` | Local rank on the node |
| `MASTER_ADDR` | Address of rank 0 node |
| `MASTER_PORT` | Port for communication |
| `NODE_RANK` | Rank of the current node |

## PyTorch Script Requirements

Your PyTorch script should handle distributed initialization:

```python
import os
import torch
import torch.distributed as dist

def init_distributed():
    # torchrun-hpc sets these environment variables
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    # Initialize the process group
    dist.init_process_group(backend='nccl')  # or 'gloo' for CPU

    # Set the device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank

def main():
    rank, world_size, local_rank = init_distributed()

    # Create model and move to device
    model = YourModel().cuda(local_rank)

    # Wrap with DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )

    # Your training code here
    train(model)

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

1. **NCCL Errors**: Try setting NCCL debug environment variables
   ```bash
   torchrun-hpc -x NCCL_DEBUG=INFO -N 2 -n 8 train.py
   ```

2. **OOM Errors**: Use memory fraction limiting
   ```bash
   torchrun-hpc --fraction-max-gpu-mem 0.8 -N 2 -n 8 train.py
   ```

3. **Rendezvous Failures**: Switch between MPI and TCP
   ```bash
   # Try MPI if TCP fails
   torchrun-hpc -r mpi -N 2 -n 8 train.py
   ```

4. **AMD GPU Issues**: Use the unswap flag
   ```bash
   torchrun-hpc -u -N 2 -n 8 train.py
   ```

## Tips and Best Practices

1. **Use MPI rendezvous** (`-r mpi`) for stable HPC environments
2. **Match processes to GPUs**: Set `-n` equal to GPUs per node
3. **Test locally first**: Use `--local` flag for debugging
4. **Save setup scripts**: Use `--setup-only` to review job configuration
5. **Monitor GPU memory**: Use `--fraction-max-gpu-mem` to prevent OOM
6. **Use exclusive nodes** for performance-critical training
7. **Enable verbose mode** (`-v`) for debugging distributed issues
8. **Save hostlists** for multi-node debugging
9. **Set appropriate time limits** to avoid job termination
10. **Use dry-run** (`--dry-run`) to verify complex commands

## Differences from Standard torchrun

- **HPC Scheduler Integration**: Native support for SLURM, LSF, Flux
- **Rendezvous Options**: Choice between MPI and TCP
- **Resource Management**: HPC-specific resource allocation
- **GPU Memory Control**: Built-in memory fraction limiting
- **AMD GPU Support**: Special handling for ROCm environments

## See Also

- `launch` - General purpose HPC job launcher
- PyTorch Distributed Documentation: https://pytorch.org/docs/stable/distributed.html
- HPC-launcher Repository: https://github.com/LBANN/HPC-launcher
- LBANN Documentation: https://lbann.readthedocs.io

---

*Generated from `torchrun-hpc -h` output*
