# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import hashlib
import logging
import os
import random
import shutil
import tarfile
import time
from pathlib import Path
from typing import List, Literal, Mapping, Tuple

import abutils
import pandas as pd
import requests
from chai_lab.data.parsing.msas.aligned_pqt import (
    a3m_to_aligned_dataframe,
    expected_basename,
)
from chai_lab.data.parsing.msas.data_source import MSADataSource
from chai_lab.utils.typing import typecheck
from tqdm.auto import tqdm

from ..utils.inputs import StructurePredictionRun
from ..version import __version__

__all__ = [
    "msa",
    "run_mmseqs2",
    "process_a3ms_for_chai",
    "hash_sequence",
    "retrieve_msa_from_cache",
    "save_msa",
    "precompute_boltz_msas",
    "precompute_chai_msas",
]

logger = logging.getLogger(__name__)


TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
)


def msa(
    sequences: str | list[str],
    output_dir: str,
    prefix: str = "/tmp",
    use_env: bool = True,
    use_filter: bool = True,
    filter: str | None = None,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    msa_server_url: str = "https://api.colabfold.com",
    user_agent: str = f"mabmaker/{__version__} briney@scripps.edu",
    use_msa_cache: bool = True,
    msa_cache_dir: str = "~/.mabmaker/msa_cache",
    quiet: bool = True,
) -> str | list[str]:
    f"""
    Perform a multiple sequence alignment using the `ColabFold MMseqs2 API`_.

    Parameters
    ----------
    sequences : str | list[str]
        The sequences to align. Can be a single sequence (as a str) or a list of sequences.
        If a list of sequences is provided, the function will return a list of MSA file paths.
        If a single sequence is provided, the function will return a single MSA file path.

    output_dir: str
        The directory to save the output files. If `use_msa_cache` is `True` and MSAs are found
        in the cache, they will be copied into this directory.

    prefix : str, optional, default="/tmp"
        The prefix for temporary output files. Passed directly to ``run_mmseqs2``.

    use_env : bool, optional, default=True
        Whether to use environment-based filtering. Passed directly to ``run_mmseqs2``.

    use_filter : bool, optional, default=True
        Whether to use filtering. Passed directly to ``run_mmseqs2``.

    use_pairing : bool, optional, default=False
        Whether to use pairing. Passed directly to ``run_mmseqs2``.

    pairing_strategy : str, optional, default="greedy"
        The pairing strategy to use. Passed directly to ``run_mmseqs2``.

    msa_server_url : str, optional, default="https://api.colabfold.com"
        The URL of the MSA server. Passed to the ``host_url`` argument of ``run_mmseqs2``.

    user_agent : str, optional, default="mabmaker/{__version__} briney@scripps.edu"
        The user agent to use. Passed directly to ``run_mmseqs2``.

    use_msa_cache : bool, optional, default=True
        Whether to use the MSA cache.

    msa_cache_dir : str, optional, default="~/.mabmaker/msa_cache"
        The directory to save the MSA cache.

    quiet : bool, optional, default=False
        Whether to suppress progress bar. Passed directly to ``run_mmseqs2``.

    Returns
    -------
    str | List[str]
        A path to the MSA or a list of paths to the MSAs. If a single sequence is provided,
        the function will return a single path. If a list of sequences is provided, the
        function will return a list of paths.

    .. _ColabFold MMseqs2 API: https://github.com/sokrypton/ColabFold
    """
    if isinstance(sequences, str):
        sequences = [sequences]

    msa_paths = []
    for seq in sequences:
        # check the cache
        msa_cache_dir = os.path.expanduser(msa_cache_dir)
        a3m_string = (
            retrieve_msa_from_cache(seq, msa_cache_dir) if use_msa_cache else None
        )
        # if the MSA is not in the cache (or we're not using the cache), run MMseqs2
        if a3m_string is None:
            a3m_lines = run_mmseqs2(
                x=seq,
                prefix=prefix,
                use_env=use_env,
                use_filter=use_filter,
                use_templates=False,
                filter=filter,
                use_pairing=use_pairing,
                pairing_strategy=pairing_strategy,
                host_url=msa_server_url,
                user_agent=user_agent,
                quiet=quiet,
            )
            a3m_string = a3m_lines[0]
            # save the MSA to the cache
            if use_msa_cache:
                save_msa(a3m_string, msa_cache_dir)
        # copy the MSA from the cache to the output directory
        msa_path = save_msa(a3m_string, output_dir)
        msa_paths.append(msa_path)

    if len(msa_paths) == 1:
        return msa_paths[0]
    else:
        return msa_paths


# This is a modified version of the `run_mmseqs2` function from the `colabfold` package.
# https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py


def run_mmseqs2(
    x: str | list[str],
    prefix: str,
    use_env: bool = True,
    use_filter: bool = True,
    use_templates: bool = False,
    filter: str | None = None,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    host_url: str = "https://api.colabfold.com",
    user_agent: str = f"mabmaker/{__version__} briney@scripps.edu",
    quiet: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Run `MMseqs2`_ to generate multiple sequence alignments using the `ColabFold`_ API.

    Parameters
    ----------
    x : str | list[str]
        The protein sequences to align.

    prefix : str
        The prefix for the output files.

    use_env : bool, optional, default=True
        Whether to use environment-based filtering.

    use_filter : bool, optional, default=True
        Whether to use filtering.

    use_templates : bool, optional, default=False
        Whether to use templates.

    filter : str, optional, default=None
        The filter to use.

    use_pairing : bool, optional, default=False
        Whether to use pairing.

    pairing_strategy : str, optional, default="greedy"
        The pairing strategy to use.

    host_url : str, optional, default="https://api.colabfold.com"
        The URL of the MMseqs2 API.

    user_agent : str, optional, default=""
        The user agent to use.

    quiet : bool, optional, default=False
        Whether to suppress progress bar.

    Returns
    -------
    List[str] | Tuple[List[str], List[str]]
        A list of paths to the A3M files. If ``use_pairing`` is ``True``, the function will return a tuple of two lists.
        The first list contains the paths to the A3M files. The second list contains the template file paths.

    .. _MMseqs2: https://github.com/soedinglab/MMseqs2
    .. _ColabFold: https://github.com/sokrypton/ColabFold

    """
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    headers = {}
    if user_agent != "":
        headers["User-Agent"] = user_agent
    else:
        logger.warning(
            "No user agent specified. Please set a user agent (e.g., 'toolname/version contact@email') to help us debug in case of problems. This warning will become an error in the future."
        )

    def submit(seqs: list[str], mode: str, N: int = 101) -> dict:
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        while True:
            error_count = 0
            try:
                # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
                # "good practice to set connect timeouts to slightly larger than a multiple of 3"
                res = requests.post(
                    f"{host_url}/{submission_endpoint}",
                    data={"q": query, "mode": mode},
                    timeout=6.02,
                    headers=headers,
                )
            except requests.exceptions.Timeout:
                if not quiet:
                    logger.warning(
                        "Timeout while submitting to MSA server. Retrying..."
                    )
                continue
            except Exception as e:
                error_count += 1
                if not quiet:
                    logger.warning(
                        f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                    )
                    logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break

        try:
            out = res.json()
        except ValueError:
            if not quiet:
                logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ID: str) -> dict:
        while True:
            error_count = 0
            try:
                res = requests.get(
                    f"{host_url}/ticket/{ID}", timeout=6.02, headers=headers
                )
            except requests.exceptions.Timeout:
                if not quiet:
                    logger.warning(
                        "Timeout while fetching status from MSA server. Retrying..."
                    )
                continue
            except Exception as e:
                error_count += 1
                if not quiet:
                    logger.warning(
                        f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                    )
                    logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        try:
            out = res.json()
        except ValueError:
            if not quiet:
                logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ID: str, path: str) -> None:
        error_count = 0
        while True:
            try:
                res = requests.get(
                    f"{host_url}/result/download/{ID}", timeout=6.02, headers=headers
                )
            except requests.exceptions.Timeout:
                if not quiet:
                    logger.warning(
                        "Timeout while fetching result from MSA server. Retrying..."
                    )
                continue
            except Exception as e:
                error_count += 1
                if not quiet:
                    logger.warning(
                        f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                    )
                    logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        with open(path, "wb") as out:
            out.write(res.content)

    # process input x
    seqs = [x] if isinstance(x, str) else x

    # compatibility to old option
    if filter is not None:
        use_filter = filter

    # setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    if use_pairing:
        use_templates = False
        mode = ""
        # greedy is default, complete was the previous behavior
        if pairing_strategy == "greedy":
            mode = "pairgreedy"
        elif pairing_strategy == "complete":
            mode = "paircomplete"
        if use_env:
            mode = mode + "-env"

    # define path
    path = f"{prefix}_{mode}"
    if not os.path.isdir(path):
        os.mkdir(path)

    # call mmseqs2 api
    tar_gz_file = f"{path}/out.tar.gz"
    N, REDO = 101, True

    # deduplicate and keep track of order
    seqs_unique = []
    # TODO this might be slow for large sets
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    Ms = [N + seqs_unique.index(seq) for seq in seqs]
    # lets do it!
    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)
        with tqdm(
            total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT, disable=quiet
        ) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")

                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    if not quiet:
                        logger.error(
                            f"Sleeping for {sleep_time}s. Reason: {out['status']}"
                        )
                    # resubmit
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)

                if out["status"] == "ERROR":
                    raise Exception(
                        "MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later."
                    )

                if out["status"] == "MAINTENANCE":
                    raise Exception(
                        "MMseqs2 API is undergoing maintenance. Please try again in a few minutes."
                    )

                # wait for job to finish
                ID, TIME = out["id"], 0
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    if not quiet:
                        logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)
                    # if TIME > 900 and out["status"] != "COMPLETE":
                    #  # something failed on the server side, need to resubmit
                    #  N += 1
                    #  break

                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False

                if out["status"] == "ERROR":
                    REDO = False
                    raise Exception(
                        "MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later."
                    )

            # Download results
            download(ID, tar_gz_file)

    # prep list of a3m files
    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # templates
    if use_templates:
        templates = {}
        # print("seq\tpdb\tcid\tevalue")
        for line in open(f"{path}/pdb70.m8", "r"):
            p = line.rstrip().split()
            M, pdb, qid, e_value = p[0], p[1], p[2], p[10]
            M = int(M)
            if M not in templates:
                templates[M] = []
            templates[M].append(pdb)
            # if len(templates[M]) <= 20:
            #  print(f"{int(M)-N}\t{pdb}\t{qid}\t{e_value}")

        template_paths = {}
        for k, TMPL in templates.items():
            TMPL_PATH = f"{prefix}_{mode}/templates_{k}"
            if not os.path.isdir(TMPL_PATH):
                os.mkdir(TMPL_PATH)
                TMPL_LINE = ",".join(TMPL[:20])
                response = None
                while True:
                    error_count = 0
                    try:
                        # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
                        # "good practice to set connect timeouts to slightly larger than a multiple of 3"
                        response = requests.get(
                            f"{host_url}/template/{TMPL_LINE}",
                            stream=True,
                            timeout=6.02,
                            headers=headers,
                        )
                    except requests.exceptions.Timeout:
                        if not quiet:
                            logger.warning(
                                "Timeout while submitting to template server. Retrying..."
                            )
                        continue
                    except Exception as e:
                        error_count += 1
                        if not quiet:
                            logger.warning(
                                f"Error while fetching result from template server. Retrying... ({error_count}/5)"
                            )
                            logger.warning(f"Error: {e}")
                        time.sleep(5)
                        if error_count > 5:
                            raise
                        continue
                    break
                with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
                    tar.extractall(path=TMPL_PATH)
                os.symlink("pdb70_a3m.ffindex", f"{TMPL_PATH}/pdb70_cs219.ffindex")
                with open(f"{TMPL_PATH}/pdb70_cs219.ffdata", "w") as f:
                    f.write("")
            template_paths[k] = TMPL_PATH

    # gather a3m lines
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        for line in open(a3m_file, "r"):
            if len(line) > 0:
                if "\x00" in line:
                    line = line.replace("\x00", "")
                    update_M = True
                if line.startswith(">") and update_M:
                    M = int(line[1:].rstrip())
                    update_M = False
                    if M not in a3m_lines:
                        a3m_lines[M] = []
                a3m_lines[M].append(line)

    # return results

    a3m_lines = ["".join(a3m_lines[n]) for n in Ms]

    if use_templates:
        template_paths_ = []
        for n in Ms:
            if n not in template_paths:
                template_paths_.append(None)
                # print(f"{n-N}\tno_templates_found")
            else:
                template_paths_.append(template_paths[n])
        template_paths = template_paths_

    return (a3m_lines, template_paths) if use_templates else a3m_lines


# -----------------------------
#     hashing and caching
# -----------------------------


def hash_sequence(seq: str) -> str:
    """
    Hash a sequence (uppercased) using SHA-256.

    Parameters
    ----------
    seq : str
        The sequence to hash.

    Returns
    -------
    str
        The SHA-256 hash of the sequence.

    """
    hash_object = hashlib.sha256(seq.upper().encode())
    return hash_object.hexdigest()


def retrieve_msa_from_cache(
    seq: str,
    msa_cache_dir: str = "~/.mabmaker/msa_cache",
) -> str | None:
    """
    Check the MSA cache for a sequence.

    Parameters
    ----------
    seq : str
        The sequence to hash.

    msa_cache_dir : str, optional
        The path to the MSA cache directory. Default is "~/.mabmaker/msa_cache".

    Returns
    -------
    str | None
        The path to the MSA if it exists, otherwise None.

    """
    # hash the sequence
    seq_hash = hash_sequence(seq)

    # check the MSA cache
    cache_path = os.path.join(msa_cache_dir, f"{seq_hash}.a3m")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return f.read()

    return None


def save_msa(
    msa: str,
    destination_dir: str,
) -> str:
    """
    Save an MSA. The MSA file will be named after the SHA256 hash of the query sequence
    and deposited into the destination directory.

    Parameters
    ----------
    msa : str
        The MSA to save, in .a3m format. The name of the MSA file will be the SHA256 hash
        of the query sequence (the first sequence in the MSA).

    destination_dir : str
        The path to the directory to save the MSA.

    Returns
    -------
    str
        The path to the saved MSA.

    """
    # get the query sequence from the MSA
    seq = msa.split("\n")[1]

    # hash the sequence
    seq_hash = hash_sequence(seq)

    # save the MSA
    abutils.io.make_dir(destination_dir)
    msa_path = os.path.join(destination_dir, f"{seq_hash}.a3m")
    with open(msa_path, "w") as f:
        f.write(msa)

    return msa_path


# -----------------------------
#        Boltz-1 MSAs
# -----------------------------


def precompute_boltz_msas(
    runs: list[StructurePredictionRun],
    base_output_path: str,
    msa_server_url: str = "https://api.colabfold.com",
    use_msa_cache: bool = True,
    msa_cache_dir: str = "~/.mabmaker/msa_cache",
) -> list[StructurePredictionRun]:
    """
    Precompute MSAs for Boltz-1 runs.

    Parameters
    ----------
    runs : list[StructurePredictionRun]
        The runs to precompute MSAs for. MSAs will be computed for each protein chain in
        each run.

    output_path : str
        The path to the output directory. MSAs will be saved into a subdirectory called
        ``msas/precomputed``.

    use_msa_cache : bool, optional, default=True
        Whether to use the MSA cache. If ``True``, the cache will be checked for existing
        MSAs before running ``mmseqs2``. If a sequence is not present in the cache, the
        resulting MSA will be saved to the cache. If ``False``, ``mmseqs2`` will be run
        for each sequence and the resulting MSAs will not be cached.

    msa_cache_dir : str, optional, default="~/.mabmaker/msa_cache"
        The path to the MSA cache directory.

    Returns
    -------
    list[StructurePredictionRun]
        The run objects with precomputed MSA file paths added to the ``msa`` attribute
        of each protein chain.

    """
    for run in tqdm(
        runs,
        desc="precomputing MSAs: ",
        bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}",
    ):
        # make the run's precomputed MSA directory
        a3m_dir = os.path.join(base_output_path, run.name, "msas", "precomputed", "a3m")
        abutils.io.make_dir(a3m_dir)
        # get MSAs
        sequences = [chain.sequence for chain in run.protein_chains]
        msa_paths = msa(
            sequences=sequences,
            output_dir=a3m_dir,
            msa_server_url=msa_server_url,
            use_msa_cache=use_msa_cache,
            msa_cache_dir=msa_cache_dir,
        )
        for chain, msa_path in zip(run.protein_chains, msa_paths):
            chain.msa = msa_path

    return runs


# -----------------------------
#        Chai-1 MSAs
# -----------------------------


def precompute_chai_msas(
    runs: list[StructurePredictionRun],
    base_output_path: str,
    msa_server_url: str = "https://api.colabfold.com",
    use_msa_cache: bool = True,
    msa_cache_dir: str = "~/.mabmaker/msa_cache",
) -> list[StructurePredictionRun]:
    """
    Precompute MSAs for Chai-1 runs.

    Parameters
    ----------
    runs : list[StructurePredictionRun]
        The runs to precompute MSAs for. MSAs will be computed for each protein chain in
        each run.

    output_path : str
        The path to the output directory. MSAs will be saved into a subdirectory called
        ``msas/precomputed``. A3M files will be saved into a subdirectory called
        ``a3m``, and aligned parquet files will be saved into a subdirectory called
        ``parquet``.

    use_msa_cache : bool, optional, default=True
        Whether to use the MSA cache. If ``True``, the cache will be checked for existing
        MSAs before running ``mmseqs2``. If a sequence is not present in the cache, the
        resulting MSA will be saved to the cache. If ``False``, ``mmseqs2`` will be run
        for each sequence and the resulting MSAs will not be cached.

    msa_cache_dir : str, optional, default="~/.mabmaker/msa_cache"
        The path to the MSA cache directory.

    Returns
    -------
    list[StructurePredictionRun]
        The run objects with precomputed MSA file paths added to the ``msa`` attribute
        of each protein chain.

    """
    for run in tqdm(
        runs,
        desc="precomputing MSAs: ",
        bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}",
    ):
        # make the run's precomputed MSA directories
        a3m_dir = os.path.join(base_output_path, run.name, "msas", "precomputed", "a3m")
        pqt_dir = os.path.join(base_output_path, run.name, "msas", "precomputed", "pqt")
        abutils.io.make_dir(a3m_dir)
        abutils.io.make_dir(pqt_dir)
        # get MSAs and convert to Chai's aligned parquet format
        sequences = [chain.sequence for chain in run.protein_chains]
        msa_paths = msa(
            sequences=sequences,
            output_dir=a3m_dir,
            msa_server_url=msa_server_url,
            use_msa_cache=use_msa_cache,
            msa_cache_dir=msa_cache_dir,
        )
        process_a3ms_for_chai(msa_paths, pqt_dir)
        # set the MSA directory
        run.msa_directory = pqt_dir

    return runs


# the following functions are adapted from:
# https://github.com/chaidiscovery/chai-lab/blob/main/chai_lab/data/parsing/msas/aligned_pqt.py


@typecheck
def merge_multi_a3m_to_aligned_dataframe(
    msa_a3m_files: Mapping[Path, MSADataSource],
    insert_keys_for_sources: Literal["all", "none", "uniprot"] = "uniprot",
) -> pd.DataFrame:
    """Merge multiple a3ms from the same query sequence into a single aligned parquet."""
    dfs = {
        src: a3m_to_aligned_dataframe(
            a3m_path,
            src,
            insert_pairing_key=(
                src in (MSADataSource.UNIPROT, MSADataSource.UNIPROT_N3)
                if insert_keys_for_sources == "uniprot"
                else (insert_keys_for_sources == "all")
            ),
        )
        for a3m_path, src in msa_a3m_files.items()
    }
    # Check that all the dfs share the same query sequence
    queries = {df.iloc[0]["sequence"] for df in dfs.values()}
    assert len(queries) == 1
    # As a base, set the query sequence
    chunks = [next(iter(dfs.values())).iloc[0:1]]
    for df in dfs.values():
        # Take the non-query sequences for all sources
        chunks.append(df.iloc[1:])
    return pd.concat(chunks, ignore_index=True).reset_index(drop=True)


def process_a3ms_for_chai(
    a3m_files: str | Path | list[str | Path], output_directory: str | Path = None
) -> None:
    """
    Converts one or more a3m-formatted alignment files into a single aligned.pqt file.
    If multiple files are provided, they are expected to be derived from the same query sequence.

    Parameters
    ----------
    a3m_files : str | Path | list[str | Path]
        Path to a3m files. Can be any of the following:
        - A directory containing a3m files
        - A single a3m file path
        - A list of a3m file paths

    output_directory : str | Path
        The path to the output directory.

    Returns
    -------
    None

    """
    # process input
    if isinstance(a3m_files, (str, Path)):
        if os.path.isdir(a3m_files):
            a3m_dir = Path(a3m_files)
            a3m_files = list(a3m_dir.glob("*.a3m"))
            if not a3m_files:
                raise ValueError(f"No a3m files found in {a3m_files}")
        elif os.path.isfile(a3m_files):
            a3m_files = [Path(a3m_files)]
        else:
            raise ValueError(f"Invalid a3m path: {a3m_files}")
    else:
        a3m_files = [Path(a3m_file) for a3m_file in a3m_files]

    # map a3m files to their source database
    mapped_a3m_files = {}
    for a3m_file in a3m_files:
        dbname = a3m_file.stem.replace("_hits", "").replace("hits_", "")
        try:
            msa_src = MSADataSource(dbname)
        except ValueError:
            msa_src = MSADataSource.UNIREF90
        mapped_a3m_files[a3m_file] = msa_src

    # merge a3m files into a dataframe
    df = merge_multi_a3m_to_aligned_dataframe(
        mapped_a3m_files, insert_keys_for_sources="uniprot"
    )

    # get the query sequence and use it to determine where we save the file.
    query_seq: str = df.iloc[0]["sequence"]

    # output
    outdir = Path(output_directory)
    outdir.mkdir(exist_ok=True, parents=True)
    df.to_parquet(outdir / expected_basename(query_seq))
