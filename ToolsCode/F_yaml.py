# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 07:22:00 2025

@author: frederic.ros
"""
import sys
import yaml
import os
from argparse import ArgumentParser, Namespace
#...............YAML CONFIG......................................................
def load_yaml_if_exists(yaml_path: str, args: Namespace) -> Namespace:
    """Load YAML if exists and merge into args."""
    if not os.path.exists(yaml_path):
        print(f"[INFO] YAML file '{yaml_path}' not found. Using default config.")
        return args

    print(f"[INFO] Loading configuration from: {yaml_path}")

    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Replace args values with YAML ones
    for key, value in yaml_config.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            print(f"[WARNING] YAML parameter '{key}' ignored (not in argparse).")

    return args

