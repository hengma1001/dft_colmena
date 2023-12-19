"""Utilities to build Parsl configurations."""
import os
from abc import ABC, abstractmethod
from typing import Literal, Optional, Sequence, Tuple, Union

from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.providers import LocalProvider, PBSProProvider
from pydantic import validator

from dft_colmena.utils import BaseSettings, PathLike


class BaseComputeSettings(BaseSettings, ABC):
    """Compute settings (HPC platform, number of GPUs, etc)."""

    name: Literal[""] = ""
    """Name of the platform to use."""

    @abstractmethod
    def config_factory(self, run_dir: PathLike) -> Config:
        """Create a new Parsl configuration.
        Parameters
        ----------
        run_dir : PathLike
            Path to store monitoring DB and parsl logs.
        Returns
        -------
        Config
            Parsl configuration.
        """
        ...


class LocalSettings(BaseComputeSettings):
    name: Literal["local"] = "local"  # type: ignore[assignment]
    max_workers: int = 1
    cores_per_worker: float = 0.0001
    worker_port_range: Tuple[int, int] = (10000, 20000)
    label: str = "htex"

    def config_factory(self, run_dir: PathLike) -> Config:
        return Config(
            run_dir=str(run_dir),
            strategy=None,
            executors=[
                HighThroughputExecutor(
                    address="127.0.0.1",
                    label=self.label,
                    max_workers=self.max_workers,
                    cores_per_worker=self.cores_per_worker,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),  # type: ignore[no-untyped-call]
                ),
            ],
        )


class WorkstationSettings(BaseComputeSettings):
    name: Literal["workstation"] = "workstation"  # type: ignore[assignment]
    """Name of the platform."""
    available_accelerators: Union[int, Sequence[str]] = 8
    """Number of GPU accelerators to use."""
    worker_port_range: Tuple[int, int] = (10000, 20000)
    """Port range."""
    retries: int = 1
    label: str = "htex"

    def config_factory(self, run_dir: PathLike) -> Config:
        return Config(
            run_dir=str(run_dir),
            retries=self.retries,
            executors=[
                HighThroughputExecutor(
                    address="127.0.0.1",
                    label=self.label,
                    cpu_affinity="block",
                    available_accelerators=self.available_accelerators,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),  # type: ignore[no-untyped-call]
                ),
            ],
        )


class PolarisSettings(BaseComputeSettings):
    name: Literal["polaris"] = "polaris"  # type: ignore[assignment]
    label: str = "htex"

    num_nodes: int = 1
    """Number of nodes to request"""
    worker_init: str = ""
    """How to start a worker. Should load any modules and activate the conda env."""
    scheduler_options: str = ""
    """PBS directives, pass -J for array jobs"""
    account: str
    """The account to charge comptue to."""
    queue: str
    """Which queue to submit jobs to, will usually be prod."""
    walltime: str
    """Maximum job time."""
    cpus_per_node: int = 64
    """Up to 64 with multithreading."""
    strategy: str = "simple"

    def config_factory(self, run_dir: PathLike) -> Config:
        """Create a configuration suitable for running all tasks on single nodes of Polaris
        We will launch 4 workers per node, each pinned to a different GPU
        Args:
            num_nodes: Number of nodes to use for the MPI parallel tasks
            user_options: Options for which account to use, location of environment files, etc
            run_dir: Directory in which to store Parsl run files. Default: `runinfo`
        """

        return Config(
            retries=1,  # Allows restarts if jobs are killed by the end of a job
            executors=[
                HighThroughputExecutor(
                    label=self.label,
                    heartbeat_period=15,
                    heartbeat_threshold=120,
                    worker_debug=True,
                    available_accelerators=4,  # Ensures one worker per accelerator
                    address=address_by_hostname(),
                    cpu_affinity="alternating",
                    prefetch_capacity=0,  # Increase if you have many more tasks than workers
                    start_method="spawn",
                    provider=PBSProProvider(  # type: ignore[no-untyped-call]
                        launcher=MpiExecLauncher(
                            bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"
                        ),  # Updates to the mpiexec command
                        account=self.account,
                        queue=self.queue,
                        select_options="ngpus=4",
                        # PBS directives (header lines): for array jobs pass '-J' option
                        scheduler_options=self.scheduler_options,
                        worker_init=self.worker_init,
                        nodes_per_block=self.num_nodes,
                        init_blocks=1,
                        min_blocks=0,
                        max_blocks=1,  # Can increase more to have more parallel jobs
                        cpus_per_node=self.cpus_per_node,
                        walltime=self.walltime,
                    ),
                ),
            ],
            run_dir=str(run_dir),
            strategy=self.strategy,
            app_cache=True,
        )


ComputeSettingsTypes = Union[
    LocalSettings,
    WorkstationSettings,
    PolarisSettings,
]
