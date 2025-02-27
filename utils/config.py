# utils/config.py
import json

def load_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config

def save_config(config, config_path):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)