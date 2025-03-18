# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import glob
import os
import shutil

import abutils


def process_boltz_output(
    original_path: str,
    processed_path: str,
    model_name: str,
    seed: int | str,
    stdout: str | None = None,
    stderr: str | None = None,
) -> None:
    """
    Process the outputs of a Boltz prediction. We'd like to restructure the output files
    into a standardized layout. The output path of each run (for Boltz, we need to do
    a separate run for each seed) is expected to be of the form:

    out_dir/
    ├── lightning_logs/                                            # Logs generated during training or evaluation
    ├── predictions/                                               # Contains the model's predictions
        ├── [input_file1]/
            ├── [input_file1]_model_0.cif                          # The predicted structure in CIF format, with the inclusion of per token pLDDT scores
            ├── confidence_[input_file1]_model_0.json              # The confidence scores (confidence_score, ptm, iptm, ligand_iptm, protein_iptm, complex_plddt, complex_iplddt, chains_ptm, pair_chains_iptm)
            ├-─ pae_[input_file1]_model_0.npz                      # The predicted PAE score for every pair of tokens
            ├── pde_[input_file1]_model_0.npz                      # The predicted PDE score for every pair of tokens
            ├── plddt_[input_file1]_model_0.npz                    # The predicted pLDDT score for every token
            ...
            └── [input_file1]_model_[diffusion_samples-1].cif      # The predicted structure in CIF format
            ...
        └── [input_file2]/
                ...
    └── processed/                                                 # Processed data used during execution

    """
    _build_output_directory_structure(processed_path)
    predictions_path = _get_predictions_path(processed_path)
    metrics_path = _get_metrics_path(processed_path)
    msas_path = _get_msas_path(processed_path)
    logs_path = _get_logs_path(processed_path)

    # copy predictions
    for model_path in glob.glob(
        os.path.join(original_path, "predictions", "*", "*_model_*.cif")
    ):
        model_num = os.path.basename(model_path).split("_")[-1].split(".")[0]
        shutil.copy(
            model_path,
            os.path.join(predictions_path, f"{seed}|{model_num}|{model_name}.cif"),
        )

    # copy metrics -- confidence
    for confidence_path in glob.glob(
        os.path.join(original_path, "predictions", "*", "confidence*.json")
    ):
        model_num = os.path.basename(confidence_path).split("_")[-1].split(".")[0]
        shutil.copy(
            confidence_path,
            os.path.join(
                confidence_path, "confidence", f"{seed}|{model_num}|{model_name}.json"
            ),
        )

    # copy metrics -- pde
    for pde_path in glob.glob(
        os.path.join(original_path, "predictions", "*", "pde*.npz")
    ):
        model_num = os.path.basename(pde_path).split("_")[-1].split(".")[0]
        shutil.copy(
            pde_path,
            os.path.join(metrics_path, "pde", f"{seed}|{model_num}|{model_name}.npz"),
        )

    # copy metrics -- pae
    for pae_path in glob.glob(
        os.path.join(original_path, "predictions", "*", "pae*.npz")
    ):
        model_num = os.path.basename(pae_path).split("_")[-1].split(".")[0]
        shutil.copy(
            pae_path,
            os.path.join(metrics_path, "pae", f"{seed}|{model_num}|{model_name}.npz"),
        )

    # copy metrics -- plddt
    for plddt_path in glob.glob(
        os.path.join(original_path, "predictions", "*", "plddt*.npz")
    ):
        model_num = os.path.basename(plddt_path).split("_")[-1].split(".")[0]
        shutil.copy(
            plddt_path,
            os.path.join(metrics_path, "plddt", f"{seed}|{model_num}|{model_name}.npz"),
        )

    # copy msas
    abutils.io.make_dir(os.path.join(msas_path, seed))
    for msa_path in glob.glob(os.path.join(original_path, "msa", "*.csv")):
        shutil.copy(
            msa_path,
            os.path.join(msas_path, seed, os.path.basename(msa_path)),
        )
    for msa_path in glob.glob(os.path.join(original_path, "msa", "*unpaired*", "*")):
        shutil.copy(
            msa_path,
            os.path.join(msas_path, seed, "unpaired", os.path.basename(msa_path)),
        )
    for msa_path in glob.glob(os.path.join(original_path, "msa", "*paired*", "*")):
        shutil.copy(
            msa_path,
            os.path.join(msas_path, seed, "paired", os.path.basename(msa_path)),
        )

    # write/copy logs
    abutils.io.make_dir(os.path.join(logs_path, seed))
    if stdout is not None:
        with open(os.path.join(logs_path, seed, "stdout.log"), "w") as f:
            f.write(stdout)
    if stderr is not None:
        with open(os.path.join(logs_path, seed, "stderr.log"), "w") as f:
            f.write(stderr)
    for lightning_logs_path in glob.glob(
        os.path.join(original_path, "*", "lightning_logs")
    ):
        shutil.copytree(
            lightning_logs_path,
            os.path.join(logs_path, seed),
            dirs_exist_ok=True,
        )


"""
PROTENIX PREDICTION OUTPUT:

├── <name>/  # specified in the input JSON file
│   ├── <seed>/  # specified via the `--seeds` flag in the inference script
│   │   ├── <name>_<seed>_sample_0.cif
│   │   ├── <name>_<seed>_summary_confidence_sample_0.json
│   │   └──... # the number of samples in each seed is specified via `--sample_diffusion.N_sample ` flag in the inference script
│   └──...
└── ...

"""


def _build_output_directory_structure(base_path: str) -> None:
    """
    Build the output directory structure.
    """
    abutils.io.make_dir(base_path)
    # predictions
    abutils.io.make_dir(_get_predictions_path(base_path))
    # metrics
    abutils.io.make_dir(_get_metrics_path(base_path))
    abutils.io.make_dir(os.path.join(_get_metrics_path(base_path), "confidence"))
    abutils.io.make_dir(os.path.join(_get_metrics_path(base_path), "pde"))
    abutils.io.make_dir(os.path.join(_get_metrics_path(base_path), "pae"))
    abutils.io.make_dir(os.path.join(_get_metrics_path(base_path), "plddt"))
    # msas
    abutils.io.make_dir(_get_msas_path(base_path))
    # logs
    abutils.io.make_dir(_get_logs_path(base_path))


def _get_predictions_path(base_path: str) -> str:
    """
    Get the path to the predictions directory.
    """
    return os.path.join(base_path, "predictions")


def _get_metrics_path(base_path: str) -> str:
    """
    Get the path to the metrics directory.
    """
    return os.path.join(base_path, "metrics")


def _get_msas_path(base_path: str) -> str:
    """
    Get the path to the MSAs directory.
    """
    return os.path.join(base_path, "msas")


def _get_logs_path(base_path: str) -> str:
    """
    Get the path to the logs directory.
    """
    return os.path.join(base_path, "logs")


def _get_raw_output_path(base_path: str) -> str:
    """
    Get the path to the raw output directory.
    """
    return os.path.join(base_path, "raw_output")
