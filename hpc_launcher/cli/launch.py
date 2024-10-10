import argparse
from hpc_launcher.systems import autodetect

# launch -N 2 -n 4 -- python -u ../PyTorch/pt-distconv.git/main.py

# -> flux run -N 2 -n 4 python -u main.py --mlperf --use-batchnorm --input-width=512 --depth-groups=8 --train-data=$TRAIN_DIR

def main():
    parser = argparse.ArgumentParser(
	description='Launches a distributed job on the current HPC cluster or cloud.')
    parser.add_argument('--verbose', '-v',
			action='store_true',
			default=False,
			help='Run in verbose mode')
    parser.add_argument('--nodes', '-N',
			default=1,
			help='Number of compute nodes for the job')
    parser.add_argument('--procs-per-node', '-n',
			default=1,
			help='Number of compute nodes for the job')
    parser.add_argument('--out',
			default=None,
			help='Capture stdout in a log file')
    parser.add_argument('--err',
			default=None,
			help='Capture stderr in a log file')
    # Grab the rest of the command line to launch
    parser.add_argument('command',
			help='command that should be executed')
    parser.add_argument('args', nargs=argparse.REMAINDER,
			help='arguments to the command that should be executed')

    args = parser.parse_args()

    print('Verbose:', args.verbose)
    system = autodetect.autodetect_current_system()
    print('Detected system:', type(system).__name__)
    scheduler=system.preferred_scheduler(args.nodes, args.procs_per_node)

    if args.out:
        scheduler.out_log_file = f'{args.out}'
    if args.err:
        scheduler.err_log_file = f'{args.err}'

    print('Launch command:', scheduler.launch_command(False))
    print(f'system parameters: node={scheduler.nodes} ppn={scheduler.procs_per_node}')
#    print('Launch script:', scheduler.launcher_script('tioga'))
    print('CMD:', args.command, args.args)

    scheduler.launch(system, args.command, args.args)
#    scheduler._system_params.print_params()

if __name__ == '__main__':
    main()
