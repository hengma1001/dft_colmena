import logging
import sys
from argparse import ArgumentParser
from datetime import datetime
from functools import partial, update_wrapper
from pathlib import Path
from typing import Any, Dict, Optional

import scipy.io as sio
from colmena.models import Result
from colmena.queue.python import PipeQueues
from colmena.task_server import ParslTaskServer
from colmena.thinker import BaseThinker, agent, result_processor

# from esmfold import run_inference
from parsl_config import ComputeSettingsTypes
from proxystore.store import register_store
from proxystore.store.file import FileStore
from pydantic import root_validator
from utils import BaseSettings


def run_dft(coords, atomic_numbers, output_dir, ind):
    sys.path.append("/homes/heng.ma/Research/md_pkgs/dft_colmena")

    from dft_colmena.utils import (
        compute_energy_gradients,
        create_pyscf_mole,
        save_result,
    )

    mol = create_pyscf_mole(atomic_numbers, coords)
    energy, gradients = compute_energy_gradients(mol)

    h5_file = f"{output_dir}/run_{ind:05d}.h5"
    save_result(h5_file, energy, gradients, coords, atomic_numbers)


class Thinker(BaseThinker):  # type: ignore[misc]
    def __init__(
        self,
        mat_file: Path,
        result_dir: Path,
        output_dir: Path,
        num_parallel_tasks: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        result_dir.mkdir(exist_ok=True)
        self.result_dir = result_dir
        self.output_dir = output_dir
        self.task_idx = 0
        self.num_parallel_tasks = num_parallel_tasks
        data = sio.loadmat(mat_file)

        self.coordinates = data["R"]
        self.atomic_numbers = data["Z"]

        self.n_tasks = len(self.coordinates)
        logging.info(f"Processing {len(self.coordinates)} input files")

    def log_result(self, result: Result, topic: str) -> None:
        """Write a JSON result per line of the output file."""
        with open(self.result_dir / f"{topic}.json", "a") as f:
            print(result.json(exclude={"inputs", "value"}), file=f)

    def submit_task(self, topic: str, *inputs: Any) -> None:
        self.queues.send_inputs(
            *inputs, method=f"run_{topic}", topic=topic, keep_inputs=False
        )

    def submit_dft_task(self) -> None:
        # If we finished processing all the results, then stop
        if self.task_idx >= self.n_tasks:
            self.done.set()
            return

        coords = self.coordinates[self.task_idx]
        atomic_numbers = self.atomic_numbers[self.task_idx]
        self.submit_task(
            "dft",
            coords,
            atomic_numbers,
            self.output_dir,
            self.task_idx,
        )
        self.task_idx += 1

    @agent(startup=True)  # type: ignore[misc]
    def start_tasks(self) -> None:
        # Only submit num_parallel_tasks at a time
        for _ in range(self.num_parallel_tasks):
            self.submit_dft_task()

    @result_processor(topic="dft")  # type: ignore[misc]
    def process_dft_result(self, result: Result) -> None:
        """Handles the returned result of the dft function and log status."""
        self.log_result(result, "dft")
        if not result.success:
            logging.warning(f"Bad dft result: {result.json()}")

        # The old task is finished, start a new one
        self.submit_dft_task()


class WorkflowSettings(BaseSettings):
    """Provide a YAML interface to configure the workflow."""

    # Workflow setup parameters
    experiment_name: str = "experiment"
    """Name of the experiment to label the run directory."""
    runs_dir: Path = Path("runs")
    """Main directory to organize all experiment run directories."""
    run_dir: Path
    """Path this particular experiment writes to (set automatically)."""

    # Inference parameters
    mat_file: Path
    """Path to the mat file with initial qm7 conditions."""
    output_dir: Path
    """output path (set automatically)"""
    omp_threads: int = 1

    # num_data_workers: int = 16
    # """Number of cores to use for datalaoder."""
    num_parallel_tasks: int = 6
    """Number of parallel task to run (should be the total number of GPUs for GPU jobs)"""
    node_local_path: Optional[Path] = None
    """Node local storage option for writing output csv files."""

    compute_settings: ComputeSettingsTypes
    """The compute settings to use."""

    # validators
    # _input_dir_exists = path_validator("input_dir")
    # _script_path_exists = path_validator("script_path")
    # _checkpoint_dir_exists = path_validator("checkpoint_dir")

    def configure_logging(self) -> None:
        """Set up logging."""
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(self.run_dir / "runtime.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    @root_validator(pre=True)
    def create_output_dirs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Generate unique run path within run_dirs with a timestamp."""
        runs_dir = Path(values.get("runs_dir", "runs")).resolve()
        experiment_name = values.get("experiment_name", "experiment")
        timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
        run_dir = runs_dir / f"{experiment_name}-{timestamp}"
        run_dir.mkdir(exist_ok=False, parents=True)
        values["run_dir"] = run_dir
        # Specify task output directory
        values["output_dir"] = run_dir / "dft_output"
        values["output_dir"].mkdir(exist_ok=False, parents=True)
        return values


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    cfg = WorkflowSettings.from_yaml(args.config)
    cfg.dump_yaml(cfg.run_dir / "params.yaml")
    cfg.configure_logging()

    # Make the proxy store
    store = FileStore(name="file", store_dir=str(cfg.run_dir / "proxy-store"))
    register_store(store)

    # Make the queues
    queues = PipeQueues(
        serialization_method="pickle",
        topics=["dft"],
        proxystore_name="file",
        proxystore_threshold=10000,
    )

    # Define the parsl configuration (this can be done using the config_factory
    # for common use cases or by defining your own configuration.)
    parsl_config = cfg.compute_settings.config_factory(cfg.run_dir / "run-info")

    # Assign constant settings to each task function
    # my_run_inference = partial(
    #     run_inference, output_dir=cfg.output_dir, hf_dir=cfg.hf_dir
    # )
    # update_wrapper(my_run_inference, run_inference)

    doer = ParslTaskServer([run_dft], queues, parsl_config)

    thinker = Thinker(
        queue=queues,
        mat_file=cfg.mat_file,
        result_dir=cfg.run_dir / "result",
        output_dir=cfg.output_dir,
        num_parallel_tasks=cfg.num_parallel_tasks,
    )
    logging.info("Created the task server and task generator")

    try:
        # Launch the servers
        doer.start()
        thinker.start()
        logging.info("Launched the servers")

        # Wait for the task generator to complete
        thinker.join()
        logging.info("Task generator has completed")
    finally:
        queues.send_kill_signal()

    # Wait for the task server to complete
    doer.join()

    # Clean up proxy store
    store.close()
