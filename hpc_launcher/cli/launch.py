import argparse
from hpc_launcher.cli import common_args
from hpc_launcher.systems import autodetect

import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=
        'Launches a distributed job on the current HPC cluster or cloud.')
    common_args.setup_arguments(parser)

    # Grab the rest of the command line to launch
    parser.add_argument('command', help='Command to be executed')
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Arguments to the command that should be executed')

    args = parser.parse_args()
    common_args.validate_arguments(args)

    if args.verbose:
        # Another option: format='%(levelname)-7s: %(message)s',
        logging.basicConfig(level=logging.INFO,
                            format='hpc-launcher: %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING,
                            format='hpc-launcher: %(message)s')

    logger.info(f'Verbose: {args.verbose}')

    system = autodetect.autodetect_current_system()
    logger.info(f'Detected system: {type(system).__name__}')
    scheduler = system.preferred_scheduler(args.nodes, args.procs_per_node)

    if args.out:
        scheduler.out_log_file = f'{args.out}'
    if args.err:
        scheduler.err_log_file = f'{args.err}'

    logger.info(
        f'system parameters: node={scheduler.nodes} ppn={scheduler.procs_per_node}'
    )

    jobid = scheduler.launch(system, args.command, args.args, not args.bg,
                             args.output_script, args.setup_only,
                             args.color_stderr)

    if jobid:
        logger.info(f'Job ID: {jobid}')


if __name__ == '__main__':
    main()
