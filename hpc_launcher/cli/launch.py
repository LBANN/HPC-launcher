import argparse
import warnings
from hpc_launcher.cli import common_args
from hpc_launcher.systems import autodetect
from hpc_launcher.utils import ceildiv

# launch -N 2 -n 4 -- python -u ../PyTorch/pt-distconv.git/main.py

# -> flux run -N 2 -n 4 python -u main.py --mlperf --use-batchnorm --input-width=512 --depth-groups=8 --train-data=$TRAIN_DIR

def main():
    parser = argparse.ArgumentParser(
	description='Launches a distributed job on the current HPC cluster or cloud.')
    common_args.setup_arguments(parser)

    # Grab the rest of the command line to launch
    parser.add_argument('command',
			help='Command to be executed')
    parser.add_argument('args', nargs=argparse.REMAINDER,
			help='Arguments to the command that should be executed')

    args = parser.parse_args()

    print('Verbose:', args.verbose)
    system = autodetect.autodetect_current_system()
    print('Detected system:', type(system).__name__)
    system_params = system.system_parameters(args.queue)

    # If the user requested a specific number of process per node, honor that
    procs_per_node = args.procs_per_node

    # Otherwise ...
    # If there is a valid set of system parameters, try to fill in the blanks provided by the user
    if system_params is not None:
        procs_per_node = system_params.procs_per_node()
        if args.gpus_at_least > 0:
            args.nodes = ceildiv(args.gpus_at_least, procs_per_node)
        elif args.gpumem_at_least > 0:
            num_gpus = ceildiv(args.gpumem_at_least, system_params.mem_per_gpu)
            args.nodes = ceildiv(num_gpus, procs_per_node)
            if args.nodes == 1:
                procs_per_node = num_gpus

    common_args.validate_arguments(args)
    scheduler=system.preferred_scheduler(args.nodes, procs_per_node, partition=args.queue)

    if args.out:
        scheduler.out_log_file = f'{args.out}'
    if args.err:
        scheduler.err_log_file = f'{args.err}'

    print('Launch command:', scheduler.launch_command(False))
    print(f'system parameters: node={scheduler.nodes} ppn={scheduler.procs_per_node}')
    print('CMD:', args.command, args.args)

    scheduler.launch(system, args.command, args.args)

if __name__ == '__main__':
    main()
