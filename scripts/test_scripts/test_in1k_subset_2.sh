#!/bin/bash

# Exit on error
set -e

cd ..

## PLEASE NOTE: CHANGE THE PATH TO THE CONFIG FILE ACCORDING TO YOUR SETUP IN THE `in_subset_extraction.py` SCRIPT
python in_subset_extraction.py
