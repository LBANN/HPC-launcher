try:
    import torch
except (ModuleNotFoundError, ImportError):
    print('PyTorch is not installed on this system, but is required for torchrun-hpc.')
    exit(1)

import argparse


def main():
    parser = argparse.ArgumentParser(
        description='A wrapper script that launches and runs distributed PyTorch on HPC systems.')
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        default=False,
                        help='Run in verbose mode')

    args = parser.parse_args()
    print('Verbose:', args.verbose)

if __name__ == '__main__':
    main()
