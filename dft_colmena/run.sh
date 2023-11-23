#!/bin/bash

# set the open thread as 10. 
# pyscf openmp is not very well built
# it will use 2x threads as set
export OMP_NUM_THREADS=10
python workflow.py -c local.yaml