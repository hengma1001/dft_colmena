# dft_colmena

This code is to run the pyscf DFT calculation with colmena to run large set of simulations on HPC. 

# Installation

The python environment to run the code can be easily set up with conda. 
```bash
conda env create -f env.yml
conda activate dft_colmena
```

# Run the code
The run configuration is managed with yaml format. The detail informations are available in the yaml files in the examples folder, where the `local.yml` is single node run with the local workstations, and the `polaris.yml` is multinode run on Polaris/ALCF machine. 

```bash
export OMP_NUM_THREADS=16
python -m dft_colmena.workflow -c ${YML_CFG_FILE}
```
