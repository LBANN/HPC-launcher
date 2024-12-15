import argparse
import pytest
from hpc_launcher.cli.common_args import setup_arguments, validate_arguments


def test_validate_arguments():
    parser = argparse.ArgumentParser()
    setup_arguments(parser)

    # Test valid arguments
    args = parser.parse_args(['--nodes', '2', '--procs-per-node', '4'])
    validate_arguments(args)
    assert args.nodes == 2 and args.procs_per_node == 4

    args = parser.parse_args(['--nodes', '3'])
    validate_arguments(args)
    assert args.nodes == 3
    assert not args.procs_per_node
    assert not args.gpus_at_least
    assert not args.gpumem_at_least

    args = parser.parse_args(['--gpumem-at-least', '16'])
    validate_arguments(args)
    assert args.gpumem_at_least == 16

    args = parser.parse_args(['--nodes', '1', '--local'])
    validate_arguments(args)
    assert args.local

    # Test invalid arguments
    with pytest.raises(ValueError):
        args = parser.parse_args(['--nodes', '2', '--gpus-at-least', '2'])
        validate_arguments(args)

    with pytest.raises(ValueError):
        args = parser.parse_args(
            ['--gpumem-at-least', '16', '--procs-per-node', '4'])
        validate_arguments(args)

    with pytest.raises(ValueError):
        args = parser.parse_args(['--local', '--bg'])
        validate_arguments(args)

    with pytest.raises(ValueError):
        args = parser.parse_args(['--local', '--scheduler', 'flux'])
        validate_arguments(args)

    with pytest.raises(ValueError):
        args = parser.parse_args(['--work-dir', '/tmp', '--run-from-dir'])
        validate_arguments(args)
