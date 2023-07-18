from configparser import ConfigParser

def load_config():
    config = ConfigParser()
    config.read('./config.yaml')
    return config