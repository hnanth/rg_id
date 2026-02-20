#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:53:51 2025

@author: huyennhu
"""

from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import concurrent.futures
import logging
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from rgidp.utils import remove_hetero, remove_non_aa, remove_altloc
import tempfile
import zipfile
from path import Path
import re

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

GLYXGLY_SASA = {
    "ALA": 108.0,
    "ARG": 238.7,
    "ASN": 143.9,
    "ASP": 140.4,
    "CYS": 134.0,
    "GLN": 178.4,
    "GLU": 172.2,
    "GLY": 80.1,
    "HIS": 183.0,
    "ILE": 175.1,
    "LEU": 178.0,
    "LYS": 200.9,
    "MET": 194.0,
    "PHE": 199.5,
    "PRO": 136.2,
    "SER": 116.5,
    "THR": 139.2,
    "TRP": 249.0,
    "TYR": 212.7,
    "VAL": 151.3,
}

def format_residues(df, key=None):
    if key is None:
        surface = format_residues(df, key="surface")
        iface = format_residues(df, key="iface")
        core = format_residues(df, key="core")
        rim = format_residues(df, key="rim")

        return surface, iface, core, rim

    df = df[df[key]]
    last_start = None
    last_end = None
    formatted = []
    for row in df.itertuples():
        if last_start is None:
            last_start = row.resid
            last_end = row.resid
        elif row.resid == last_end + 1:
            last_end = row.resid
        else:
            formatted.append(f"{last_start}-{last_end}")
            last_start = row.resid
            last_end = row.resid
    if last_start is not None and last_end is not None:
        formatted.append(f"{last_start}-{last_end}")
    formatted = ":".join(formatted)

    return formatted

def calc_rasa(structure):
    # compute SASA for complex
    sr = ShrakeRupley()
    sr.compute(structure, level="R")
    # normalize the SASA
    for res in structure.get_residues():
        res.sasa /= GLYXGLY_SASA[res.resname]
        
def annotate_iface(model, idp_id, receptor_id):
    # select only the IDP and receptor chains
    chains = list(model.get_chains())
    for chain in chains:
        if chain.id not in [idp_id, receptor_id]:
            model.detach_child(chain.id)

    # clean the structure
    remove_hetero(model)
    remove_non_aa(model)
    remove_altloc(model)

    # compute SASA for complex
    calc_rasa(model)
    rasa_complex = pd.DataFrame(
        [(r.parent.id, r.id[1], r.sasa) for r in model.get_residues()],
        columns=["chain", "resid", "rasa"],
    )

    # compute SASA for IDP
    calc_rasa(model[idp_id])
    rasa_idp = pd.DataFrame(
        [(r.parent.id, r.id[1], r.sasa) for r in model[idp_id].get_residues()],
        columns=["chain", "resid", "rasa"],
    )

    # compute SASA for receptor
    calc_rasa(model[receptor_id])
    rasa_receptor = pd.DataFrame(
        [(r.parent.id, r.id[1], r.sasa) for r in model[receptor_id].get_residues()],
        columns=["chain", "resid", "rasa"],
    )

    # merge the SASA dataframes
    rasa_chains = pd.concat([rasa_idp, rasa_receptor], ignore_index=True)
    rasa = pd.merge(
        rasa_chains,
        rasa_complex,
        on=["chain", "resid"],
        suffixes=["_chain", "_complex"],
        how="inner",
    )
    # check if we have nan values
    assert not rasa.isnull().values.any(), "NaN values"

    # find the surface, interface, core, and the rim
    delta_rasa = rasa["rasa_chain"] - rasa["rasa_complex"]
    rasa["interior"] = np.isclose(delta_rasa, 0) & (rasa["rasa_complex"] < 0.25 - 1e-8)
    rasa["surface"] = np.isclose(delta_rasa, 0) & ~rasa["interior"]
    rasa["iface"] = delta_rasa > 1e-8
    rasa["rim"] = rasa["iface"] & (rasa["rasa_complex"] > 0.25 + 1e-8)
    rasa["core"] = rasa["iface"] & ~rasa["rim"]

    rasa.sort_values(by=["chain", "resid"], inplace=True)

    # convert interface to string (e.g 5-10:12-13:15:20-28)
    idp_surface, idp_iface, idp_core, idp_rim = format_residues(
        rasa[rasa["chain"] == idp_id]
    )
    receptor_surface, receptor_iface, receptor_core, receptor_rim = format_residues(
        rasa[rasa["chain"] == receptor_id]
    )
    
    return (
        idp_surface,
        idp_iface,
        idp_core,
        idp_rim,
        receptor_surface,
        receptor_iface,
        receptor_core,
        receptor_rim
    )

def annotate_iface_pdb(job_name, pdb_path, idp_id, receptor_id):
    
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure(job_name, pdb_path)
    model = structure[0]

    return annotate_iface(model, idp_id, receptor_id)

def process_job_zip(job_name, zip_path, idp_id, receptor_id):
    
    job_results = []
    
    with zipfile.ZipFile(zip_path, "r") as z:
        pdb_files = [f for f in z.namelist() if f.endswith(".pdb")]
            
        for file in pdb_files:
            # extract and process one file at a time
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                tmp.write(z.read(file))
                tmp_path = tmp.name

            try:
                pred_jobname, rank, model_name, model, seed = re.match(r"(.*)_unrelaxed_rank_(\d+)_(.*)_model_(\d+)_seed_(\d+).*",
                                                                       file).groups()
                
                iface = annotate_iface_pdb(job_name, tmp_path, idp_id, receptor_id)
                iface_results = (job_name, idp_id, receptor_id, rank, model, seed, file) + iface # does not include model_name
                job_results.append(iface_results)
            finally:
                os.unlink(tmp_path)
                
    return job_results
        


def main(
    idp_file,
    colabfold_dir,
    n_jobs,
    output,
    idp_chain,
    receptor_chain,
    is_zip
):
    # read the dataset
    df = pd.read_csv(idp_file, sep='\t')
    df = df.rename(columns={"jobname": "job_name"})
    df["idp_id"] = idp_chain
    df["receptor_id"] = receptor_chain 
    
    if not is_zip:
        df["pdb_path"] = colabfold_dir + "/" + df["af_fn"]
        # run the jobs - files are not contained in zips 
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                    executor.submit(
                        annotate_iface_pdb,
                        row.job_name,
                        row.pdb_path,
                        row.idp_id,  # IDP chain
                        row.receptor_id,  # Receptor chain
                    ): (
                        row.job_name,
                        row.idp_id,
                        row.receptor_id,
                        row.rank,
                        #row.model_name,
                        row.model,
                        row.seed,
                        row.af_fn
                    )
                    for row in df.itertuples()
            }
    
            # wait for the jobs to finish
            df_iface = []
            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Calculating Interfaces",
                unit="entries",
            ):
                info = futures[fut]
                try:
                    result = fut.result()
                    df_iface.append((*info, *result))
                except Exception as e:
                    logger.exception(e)
    
    else:
        df_jobs = df[["job_name", "idp_id", "receptor_id"]].drop_duplicates().reset_index(drop=True)
        df_jobs["zip_path"] = colabfold_dir + "/" + df_jobs["job_name"] + ".result.zip"
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                    executor.submit(
                        process_job_zip,
                        row.job_name,
                        row.zip_path, 
                        row.idp_id, 
                        row.receptor_id
                    ): row.job_name
                    for row in df_jobs.itertuples()
            }
            
            # wait for the jobs to finish
            df_iface = []
            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Calculating Interfaces (ZIP)",
                unit="jobs",
            ):
                job_name = futures[fut]
                try:
                    job_results = fut.result()
                    df_iface.extend(job_results)
                except Exception as e:
                    logger.exception(e)
    
    # save the results
    df_iface = pd.DataFrame(
        df_iface,
        columns=[
            "job_name",
            "idp_id",
            "receptor_id",
            "rank",
            #"model_name",
            "model",
            "seed",
            "af_fn",
            "idp_surface_pred",
            "idp_iface_pred",
            "idp_core_pred",
            "idp_rim_pred",
            "receptor_surface_pred",
            "receptor_iface_pred",
            "receptor_core_pred",
            "receptor_rim_pred"               
        ],
    )
    df_iface.to_csv(output, sep='\t', index=False)
            
if __name__ == "__main__":
    parser = ArgumentParser(description='Running interface analysis for ColabFold predictions')
    parser.add_argument("colabfold_dir",
                        help='directory containing prediction files')
    parser.add_argument("input_file", 
                        help='.csv/.tsv file containing the IDP dataset. Use the PyRosetta output')
    parser.add_argument("output_file", 
                        help='Output file name to save (format: .tsv)')
    parser.add_argument("--zip",
                        default=False,
                        action="store_true",
                        help="indicate if predictions are stored in .result.zip files")
    parser.add_argument("--idp_chain", 
                        default='A',
                        help="Set target chain. Default: A")
    parser.add_argument("--receptor_chain",
                        default='B',
                        help="Set binder chain. Default: B")
    parser.add_argument("--n_workers", type=int, default=5,
                        help="Number of parallel workers (default: 5). To set to sequential, set to 1")
    args = parser.parse_args()
    main(args.input_file, args.colabfold_dir, args.n_workers, args.output_file, args.idp_chain, args.receptor_chain, args.zip)