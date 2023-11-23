import json
from pathlib import Path
from typing import Optional, Type, TypeVar, Union

import h5py
import numpy as np
import pyscf
import yaml
from pydantic import BaseSettings as _BaseSettings
from pydantic import validator
from pyscf import dft, grad

_T = TypeVar("_T")

PathLike = Union[Path, str]


def _resolve_path_exists(value: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    p = value.resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def path_validator(field: str) -> classmethod:
    decorator = validator(field, allow_reuse=True)
    _validator = decorator(_resolve_path_exists)
    return _validator


class BaseSettings(_BaseSettings):
    """Base settings to provide an easier interface to read/write YAML files."""

    def dump_yaml(self, filename: PathLike) -> None:
        with open(filename, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: PathLike) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)  # type: ignore


def create_pyscf_mole(atomic_numbers, cartesian_coordinates):
    """Create a PySCF molecule object from cartesian coordinates and atomic numbers.
    Assumes the convention of the QM7 dataset, where an atomic number of 0 indicates
    the end of a molecule.

    Args:
        atomic_numbers (list): List of atomic numbers.
        cartesian_coordinates (list): List of cartesian coordinates.

    Returns:
        molecule (pyscf.gto.Mole): PySCF molecule object.
    """

    atomic_numbers = atomic_numbers.astype(int)
    nonzero_indices = np.nonzero(atomic_numbers)[0]
    atomic_numbers = atomic_numbers[nonzero_indices]
    cartesian_coordinates = cartesian_coordinates[nonzero_indices]

    # Create the molecule object
    molecule = pyscf.gto.Mole()

    # Set the cartesian coordinates and atomic numbers
    molecule.atom = list(zip(atomic_numbers, cartesian_coordinates))

    # Build the molecule
    molecule.build()

    return molecule


def compute_energy_gradients(mol, xc="wB97X"):
    mf_mol = dft.RKS(mol)
    mf_mol.xc = xc
    energy = mf_mol.kernel()
    grad_mf = grad.RKS(mf_mol)
    gradients = grad_mf.kernel()

    return energy, gradients


def save_result(h5_file, energy, gradients, coordinates, atomic_numbers):
    with h5py.File(h5_file, "w") as f:
        f.create_dataset("energy", data=energy)
        f.create_dataset("gradients", data=gradients)
        f.create_dataset("coordinates", data=coordinates)
        f.create_dataset("atomic_numbers", data=atomic_numbers)
