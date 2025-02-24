import json
import os
import pickle
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import BinaryIO  # noqa: F401

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from modal import App, Image, asgi_app, gpu

# need larger memory GPU here
gpu_config = gpu.A100(count=1)
stub = App("colabfold")
web_app = FastAPI()

image = (
    Image.debian_slim(python_version="3.11")
    .micromamba()
    .apt_install("wget", "git")
    .pip_install(
        "colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold",
        "google-cloud-storage",
        "google-auth",
    )
    .micromamba_install(
        "kalign2=2.04",  # this is the version used by the model
        "hhsuite=3.3.0",  # this is the version used by the model
        channels=["conda-forge", "bioconda"],
    )
    .run_commands(
        'pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
        gpu=gpu_config,
    )  # this is the version used by the model
    .run_commands("python -m colabfold.download")
    .micromamba_install(
        "openmm=8.0.0", "pdbfixer==1.9", channels=["conda-forge", "bioconda"]
    )
    .pip_install("biopython")
)

with image.imports():
    from Bio import PDB
    from colabfold.batch import get_queries, run
    from colabfold.download import default_data_dir


@stub.function(image=image, gpu=gpu_config, timeout=30000)
def fold_monomer_with_colabfold(
        query_sequence: str,
        unique_filename: str,
):
    out_dir = tempfile.gettempdir()

    jobname = str(uuid.uuid4())
    queries_path = os.path.join(jobname, f"{jobname}.csv")
    with open(queries_path, "w") as text_file:  # noqa: FURB103
        text_file.write(f"id,sequence\n{jobname},{query_sequence}")
    queries, is_complex = get_queries(queries_path)
    run(
        queries=queries,
        result_dir=out_dir,
        use_templates=False,
        num_relax=1,
        msa_mode="MMseqs2 (UniRef+Environmental)",
        model_type="alphafold2_ptm",
        num_models=1,
        num_recycles=1,
        relax_max_iterations=None,
        recycle_early_stop_tolerance=None,
        num_seeds=1,
        use_dropout=False,
        model_order=[1],
        is_complex=is_complex,
        data_dir=default_data_dir,
        keep_existing_results=False,
        rank_by="pLDDT",
        pair_mode="unpaired",
        stop_at_score=float(100),
        zip_results=True,
        save_all=True,
        max_msa=None,
        use_cluster_profile=False,
        save_recycles=False,
        user_agent="colabfold/google-colab-main"
    )

    path = next(
        f for f in os.listdir(out_dir) if unique_filename in f and f.endswith(".zip")
    )
    z = zipfile.ZipFile(os.path.join(out_dir, path))
    score_file = [
        s for s in z.namelist() if ".json" in s if f"{unique_filename}_scores" in s
    ]
    return "wait"


@stub.function(
    image=image,
    gpu=gpu_config,
    timeout=30000,
    concurrency_limit=150,
)
def fold_multimer_with_colabfold(query_sequence: str, unique_filename: str) -> dict:  # noqa: PLR0915
    out_dir = tempfile.gettempdir()
    jobname = f"{unique_filename}"

    def check(folder):
        return not os.path.exists(folder)

    if not check(jobname):
        n = 0
        while not check(f"{jobname}_{n}"):
            n += 1
        jobname = f"{jobname}_{n}"

    # make directory to save results
    os.makedirs(jobname, exist_ok=True)
    queries_path = os.path.join(jobname, f"{jobname}.csv")
    with open(queries_path, "w") as text_file:  # noqa: FURB103
        text_file.write(f"id,sequence\n{jobname},{query_sequence}")

    queries, is_complex = get_queries(queries_path)
    run(
        queries=queries,
        result_dir=out_dir,
        use_templates=False,
        num_relax=1,
        msa_mode="MMseqs2 (UniRef+Environmental)",
        model_type="alphafold2_multimer_v3",
        num_models=1,
        num_recycles=3,
        relax_max_iterations=200,
        recycle_early_stop_tolerance=None,
        num_seeds=1,
        use_dropout=False,
        model_order=[1, 2, 3, 4, 5],
        is_complex=is_complex,
        data_dir=default_data_dir,
        keep_existing_results=False,
        rank_by="multimer",
        pair_mode="unpaired_paired",
        pairing_strategy="greedy",
        stop_at_score=float(100),
        zip_results=False,
        save_all=True,
        max_msa=None,
        use_cluster_profile=False,
        save_recycles=False,
        user_agent="colabfold/google-colab-main"
    )
    all_files = list(os.listdir(out_dir))

    pae_file = f"{unique_filename}_predicted_aligned_error_v1.json"
    with open(os.path.join(out_dir, pae_file)) as f:
        pae = np.array(json.load(f)["predicted_aligned_error"])

    binder_len = len(pae) - len(
        query_sequence.split(":")[1]
    )  # note that we are assuming that the binder always comes first in the sequence!
    pae_interaction = (
                              pae[:binder_len, binder_len:].mean() + pae[binder_len:, :binder_len].mean()
                      ) / 2
    scores = None
    for file in all_files:
        if file.startswith(f"{unique_filename}_scores_rank_001"):
            with open(os.path.join(out_dir, file)) as f:
                scores = json.load(f)
            break

    binder_chain, target_chain = "A", "B"  # Adjust based on your complex
    contact_cutoff = 3.5  # Contact definition in Å

    def get_residues_in_contact(structure, chain_a, chain_b, cutoff=3.5):
        """Find residues from chain_a and chain_b that are within a given cutoff distance."""
        model = structure[0]  # First model in PDB
        residues_a = [res for res in model[chain_a] if res.id[0] == " "]
        residues_b = [res for res in model[chain_b] if res.id[0] == " "]

        contact_residues_a, contact_residues_b = set(), set()

        for res_a in residues_a:
            for res_b in residues_b:
                try:
                    for atom_a in res_a:
                        for atom_b in res_b:
                            distance = atom_a - atom_b
                            if distance < cutoff:
                                contact_residues_a.add(res_a.id[1])
                                contact_residues_b.add(res_b.id[1])
                                break  # Move to next residue once contact is found
                except KeyError:
                    pass  # Skip residues without atomic coordinates

        return sorted(contact_residues_a), sorted(contact_residues_b)

    pdb_str = None
    mean_plddt = None
    i_pae = None
    temp_pdb_file = None
    for file in all_files:
        if file.endswith(".pdb") and "relaxed" in file:
            temp_pdb_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
                mode="w+", suffix=".pdb", delete=False
            )
            pdb_str = Path(os.path.join(out_dir, file)).read_text()
            temp_pdb_file.write(pdb_str)
            temp_pdb_file.flush()
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure("model", temp_pdb_file.name)
            receptor_res_in_contact, ligand_res_in_contact = get_residues_in_contact(
                structure, binder_chain, target_chain, contact_cutoff
            )
            receptor_res_in_contact = np.array(receptor_res_in_contact) - 1
            ligand_res_in_contact = np.array(ligand_res_in_contact) - 1
            interface_residues = np.unique(
                np.concatenate((receptor_res_in_contact, ligand_res_in_contact))
            )
            symmetric_contact_pae = pae[np.ix_(interface_residues, interface_residues)]
            i_pae = np.median(symmetric_contact_pae)

    if temp_pdb_file is not None:
        os.remove(temp_pdb_file.name)

    for file in all_files:
        if file.endswith(".pickle"):
            with open(os.path.join(out_dir, file), "rb") as fp:  # type: BinaryIO
                pick_date = pickle.load(fp)  # noqa: S301
                mean_plddt = pick_date["mean_plddt"]
            break

    iptm = scores["iptm"] if scores is not None else 1000.0
    shutil.rmtree(jobname)
    return {
        "plDDT": float(mean_plddt) if mean_plddt else None,
        "iPTM": float(iptm) if iptm else None,
        "iPAE": float(i_pae) if i_pae else None,
        "pae_interaction": float(pae_interaction) if pae_interaction else None,
        "pdb_str": pdb_str,
        "binder_sequence": query_sequence.split(":")[0],
    }


@stub.function(image=image, gpu=gpu_config, timeout=30000)
def fold_multimer_parallel(sequences: list[str], target_sequence: str) -> list[dict]:
    unique_filename = str(uuid.uuid4())
    arguments_list = []
    for idx, sequence in enumerate(sequences):
        arguments_list.append((
            f"{sequence}:{target_sequence}",
            f"{unique_filename}_{idx}",
        ))
    return list(fold_multimer_with_colabfold.starmap(arguments_list))


@web_app.post("/alphafold/{name}")
async def multifold_endpoint(json_data: dict, name: str):
    blob = {}
    if name == "multimer":
        blob = await fold_multimer_parallel.remote.aio(
            json_data["sequences"], json_data["target_sequence"]
        )
    elif name == "monomer":
        blob = await fold_monomer_with_colabfold.remote.aio(
            json_data["sequence"], model_type="alphafold2_ptm"
        )
    return JSONResponse(content={"results": blob})


@web_app.get("/")
async def root() -> dict:
    return {"message": "Multimer or Monomer AlphaFold2"}


@stub.function()
@asgi_app()
def app() -> FastAPI:
    return web_app
