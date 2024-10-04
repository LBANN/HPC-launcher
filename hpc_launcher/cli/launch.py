import argparse
from hpc_launcher.systems import autodetect


def main():
    parser = argparse.ArgumentParser(
	description='Launches a distributed job on the current HPC cluster or cloud.')
    parser.add_argument('--verbose', '-v',
			action='store_true',
			default=False,
			help='Run in verbose mode')

    args = parser.parse_args()

    print('Verbose:', args.verbose)
    system = autodetect.autodetect_current_system()
    print('Detected system:', type(system).__name__)


if __name__ == '__main__':
    main()
