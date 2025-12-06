# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 07:15:11 2025

@author: frederic.ros
"""
import sys
# adding Folder_2 to the system path
#sys.path.insert(0, "..\SoftConcurrent")
sys.path.insert(0, "concurrents/deeplearningmethods")
sys.path.insert(0, "concurrents/deeplearningmethods/data")
sys.path.insert(0, "ToolsCode")
from argparse import ArgumentParser, Namespace
from F_yaml import load_yaml_if_exists
from train_icc_pstrides import launchiccdsprites
from train_icc_mnist import  launchiccmnist
from train_SCAN_mnist import launchscanmnist
from train_SCAN_dsprides import launchscandsprites
from train_depict_mnist import launchdepictmnist
from train_depict_pstrides import launchdepictdsprites

#...............DEFAULT CONFIG......................................................
def get_config_deeplearning(yaml_path: str = None) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--IIC", type=bool, default=True)
    parser.add_argument("--SCAN", type=bool, default=False)
    parser.add_argument("--DEPICT", type=bool, default=False)
    parser.add_argument("--dataset", type=str,default="MNIST")
        
    return parser.parse_args()

def launchdeeplearning():
    yaml_path = "yaml_deeplearning.txt"
    cfg = get_config_deeplearning(yaml_path)
    cfg = load_yaml_if_exists(yaml_path, cfg)
    print("launch deeplearning process from",yaml_path)
    if cfg.IIC == True:
        if cfg.dataset == "MNIST":
            print("ICC on MNIST")
            launchiccmnist()
        else:
            print("ICC on Dsprites")
            launchiccdsprites()
    if cfg.SCAN == True:
        if cfg.dataset == "MNIST":
            print("SCAN on MNIST")
            launchiccmnist()
        else:
            print("SCAN on Dsprites")
            launchiccdsprites()
    
    if cfg.DEPICT == True:
        if cfg.dataset == "MNIST":
            print("DEPICT on MNIST")
            launchdepictmnist()
        else:
            print("DEPICT on Dsprites")
            launchdepictdsprites
            
    
    #launchiccdsprites()

