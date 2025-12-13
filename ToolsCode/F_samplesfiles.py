# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 07:39:01 2025

@author: frederic.ros
"""

import os
import shutil
import random

def sample_txt_files(input_folder, output_folder, n):
    """
    Randomly select n .txt files from input_folder and copy them to output_folder.
    If n > number of files, copy all files.
    
    Parameters:
        input_folder (str): path to the folder containing input .txt files
        output_folder (str): path to the folder where sampled files will be copied
        n (int): number of files to sample
    """
    
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # List all .txt files in input folder
    all_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    
    if not all_files:
        print(f"No .txt files found in {input_folder}")
        return
    
    # Determine number of files to sample
    n_files = min(n, len(all_files))
    
    # Randomly sample files
    sampled_files = random.sample(all_files, n_files)
    
    # Copy sampled files to output folder
    for f in sampled_files:
        src_path = os.path.join(input_folder, f)
        dst_path = os.path.join(output_folder, f)
        shutil.copy2(src_path, dst_path)
    
    print(f"Copied {len(sampled_files)} files from {input_folder} to {output_folder}")

input_folder = "../savesyntheticdata/level3"
output_folder = "../savesyntheticdata/sampleslevel3"
sample_txt_files(input_folder, output_folder, 100)