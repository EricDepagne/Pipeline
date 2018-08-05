import pytest
import configparser

Config = configparser.ConfigParser()
configfile = './tests/pipeline.ini.test'
Config.read(configfile)


def test_directories():
    assert(Config['Directory']['datadir'] == 'data')
    assert(Config['Directory']['savedir'] == 'save')

def test_loglevel():
    assert(Config['Logging']['log_level'] == 'log')
