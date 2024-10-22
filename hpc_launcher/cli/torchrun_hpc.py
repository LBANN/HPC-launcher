from hpc_launcher.cli import common_args
import argparse


def main():
    parser = argparse.ArgumentParser(
        description=
        'A wrapper script that launches and runs distributed PyTorch on HPC systems.'
    )
    common_args.setup_arguments(parser)

    # Grab the rest of the command line to launch
    parser.add_argument('script', help='Python script to be executed')
    parser.add_argument('args',
                        nargs=argparse.REMAINDER,
                        help='Arguments to the Python script')

    args = parser.parse_args()
    common_args.validate_arguments(args)

    try:
        import torch
    except (ModuleNotFoundError, ImportError):
        print(
            'PyTorch is not installed on this system, but is required for torchrun-hpc.'
        )
        exit(1)

    print('Verbose:', args.verbose)


if __name__ == '__main__':
    main()
