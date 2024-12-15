from unittest.mock import patch
from hpc_launcher.systems.lc.el_capitan_family import ElCapitan
from hpc_launcher.systems.system import GenericSystem
from hpc_launcher.systems.autodetect import system, autodetect_current_system


@patch('socket.gethostname', return_value='linux123')
def test_system(mock_gethostname):
    assert system() == 'linux'


@patch('socket.gethostname', return_value='tuolumne0001')
def test_autodetect(mock_gethostname):
    assert isinstance(autodetect_current_system(), ElCapitan)


@patch('socket.gethostname', return_value='linux')
def test_autodetect(mock_gethostname):
    assert system() == 'linux'
    assert isinstance(autodetect_current_system(), GenericSystem)


if __name__ == '__main__':
    test_system()
    test_autodetect()
