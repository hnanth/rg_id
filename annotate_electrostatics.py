from tqdm import tqdm
import pandas as pd
import concurrent.futures
import logging
from Bio.PDB.Polypeptide import one_to_index, index_to_three
from Bio.PDB import PDBParser
from rgidp.utils import remove_non_aa, remove_hetero, remove_altloc
import os
from argparse import ArgumentParser
import zipfile, tempfile

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def one_to_three(residue):
    return index_to_three(one_to_index(residue))

CHARGED_RESIDUES = [one_to_three(r) for r in "RKDEH"]
POLAR_RESIDUES = [one_to_three(r) for r in "QNSTC"]
HYDROPHOBIC_RESIDUES = [one_to_three(r) for r in "AILMFVWY"]
SMALL_RESIDUES = [one_to_three(r) for r in "PG"]

HYDROPATHY_VALUES = {
    "Ala": 1.8,
    "Arg": -4.5,
    "Asn": -3.5,
    "Asp": -3.5,
    "Cys": 2.5,
    "Gln": -3.5,
    "Glu": -3.5,
    "Gly": -0.4,
    "His": -3.2,
    "Ile": 4.5,
    "Leu": 3.8,
    "Lys": -3.9,
    "Met": 1.9,
    "Phe": 2.8,
    "Pro": -1.6,
    "Ser": -0.8,
    "Thr": -0.7,
    "Trp": -0.9,
    "Tyr": -1.3,
    "Val": 4.2,
}
HYDROPATHY_VALUES = {k.upper(): v for k, v in HYDROPATHY_VALUES.items()}


def calc_hydropathy(model, residue_ids):
    residues = list([r for r in model.get_residues() if r.id[1] in residue_ids])
    residues_with_hydropathy = [
        r for r in residues if r.resname.upper() in HYDROPATHY_VALUES
    ]
    hydropathy_values = [HYDROPATHY_VALUES[r.resname] for r in residues_with_hydropathy]
    if len(hydropathy_values) == 0:
        return pd.DataFrame(dict(hydropathy=[None]))
    return pd.DataFrame(
        dict(hydropathy=[sum(hydropathy_values) / len(hydropathy_values)])
    )


def count_electrostatics(model, residue_ids):
    residues = list([r for r in model.get_residues() if r.id[1] in residue_ids])
    charged = [r for r in residues if r.resname in CHARGED_RESIDUES]
    polar = [r for r in residues if r.resname in POLAR_RESIDUES]
    hydrophobic = [r for r in residues if r.resname in HYDROPHOBIC_RESIDUES]
    small = [r for r in residues if r.resname in SMALL_RESIDUES]

    n_charged = len(charged)
    n_polar = len(polar)
    n_hydrophobic = len(hydrophobic)
    n_small = len(small)

    return pd.DataFrame(
        dict(
            charged=[n_charged],
            polar=[n_polar],
            hydrophobic=[n_hydrophobic],
            small=[n_small],
        )
    )


def parse_residues(residues):
    try:
        residues = residues.split(":")
        residues = [r.split("-") for r in residues]
        residues = [[int(r) for r in r] for r in residues]
        residues = [list(range(r[0], r[1] + 1)) for r in residues]
        return sum(residues, [])
    except Exception as exc:
        return []


def annotate_electrostatics(
    model,
    idp_id,
    idp_surface,
    idp_iface,
    idp_core,
    idp_rim,
    receptor_id,
    receptor_surface,
    receptor_iface,
    receptor_core,
    receptor_rim,
):
    # clean the structure
    remove_hetero(model)
    remove_non_aa(model)
    remove_altloc(model)

    # parse residues
    idp_surface = set(parse_residues(idp_surface))
    idp_iface = set(parse_residues(idp_iface))
    idp_core = set(parse_residues(idp_core))
    idp_rim = set(parse_residues(idp_rim))
    receptor_surface = set(parse_residues(receptor_surface))
    receptor_iface = set(parse_residues(receptor_iface))
    receptor_core = set(parse_residues(receptor_core))
    receptor_rim = set(parse_residues(receptor_rim))

    idp_surface_df = count_electrostatics(model[idp_id], idp_surface) / len(idp_surface)
    idp_iface_df = count_electrostatics(model[idp_id], idp_iface) / len(idp_iface)
    idp_core_df = count_electrostatics(model[idp_id], idp_core) / len(idp_core)
    idp_rim_df = count_electrostatics(model[idp_id], idp_rim) / len(idp_rim)
    receptor_surface_df = count_electrostatics(
        model[receptor_id], receptor_surface
    ) / len(receptor_surface)
    receptor_iface_df = count_electrostatics(model[receptor_id], receptor_iface) / len(
        receptor_iface
    )
    receptor_core_df = count_electrostatics(model[receptor_id], receptor_core) / len(
        receptor_core
    )
    receptor_rim_df = count_electrostatics(model[receptor_id], receptor_rim) / len(
        receptor_rim
    )

    # hydropathy
    idp_surface_hydropathy = calc_hydropathy(model[idp_id], idp_surface)
    idp_iface_hydropathy = calc_hydropathy(model[idp_id], idp_iface)
    idp_core_hydropathy = calc_hydropathy(model[idp_id], idp_core)
    idp_rim_hydropathy = calc_hydropathy(model[idp_id], idp_rim)
    receptor_surface_hydropathy = calc_hydropathy(model[receptor_id], receptor_surface)
    receptor_iface_hydropathy = calc_hydropathy(model[receptor_id], receptor_iface)
    receptor_core_hydropathy = calc_hydropathy(model[receptor_id], receptor_core)
    receptor_rim_hydropathy = calc_hydropathy(model[receptor_id], receptor_rim)

    # concat
    idp_df = pd.concat(
        [
            idp_surface_df.assign(residue_type="surface"),
            idp_iface_df.assign(residue_type="interface"),
            idp_core_df.assign(residue_type="core"),
            idp_rim_df.assign(residue_type="rim"),
        ],
        ignore_index=True,
    )
    idp_hydropathy_df = pd.concat(
        [
            idp_surface_hydropathy.assign(residue_type="surface"),
            idp_iface_hydropathy.assign(residue_type="interface"),
            idp_core_hydropathy.assign(residue_type="core"),
            idp_rim_hydropathy.assign(residue_type="rim"),
        ],
        ignore_index=True,
    )
    idp_df["chain"] = "idp"
    idp_hydropathy_df["chain"] = "idp"
    receptor_df = pd.concat(
        [
            receptor_surface_df.assign(residue_type="surface"),
            receptor_iface_df.assign(residue_type="interface"),
            receptor_core_df.assign(residue_type="core"),
            receptor_rim_df.assign(residue_type="rim"),
        ],
        ignore_index=True,
    )
    receptor_hydropathy_df = pd.concat(
        [
            receptor_surface_hydropathy.assign(residue_type="surface"),
            receptor_iface_hydropathy.assign(residue_type="interface"),
            receptor_core_hydropathy.assign(residue_type="core"),
            receptor_rim_hydropathy.assign(residue_type="rim"),
        ],
        ignore_index=True,
    )
    receptor_df["chain"] = "receptor"
    receptor_hydropathy_df["chain"] = "receptor"

    # concat
    df = pd.concat(
        [idp_df, idp_hydropathy_df, receptor_df, receptor_hydropathy_df],
        ignore_index=True,
    )
    df = df.fillna(0)

    return df

def annotate_electrostatics_pdb(job_name, pdb_path, *args, **kwargs):
    # read the PDB file
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure(job_name, pdb_path)
    model = structure[0]

    return annotate_electrostatics(model, *args, **kwargs)

def process_job_zip(job_zip, df):
    
    results = []
    
    with zipfile.ZipFile(job_zip, "r") as z:
        for row in df.itertuples():
            # extract and process one pdb file at a time
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                tmp.write(z.read(row.af_fn))
                tmp_path = tmp.name
                df_annotate = annotate_electrostatics_pdb(row.job_name, 
                                                          tmp_path,
                                                          row.idp_id, 
                                                          row.idp_surface_pred,
                                                          row.idp_iface_pred,
                                                          row.idp_core_pred,
                                                          row.idp_rim_pred,
                                                          row.receptor_id,
                                                          row.receptor_surface_pred,
                                                          row.receptor_iface_pred,
                                                          row.receptor_core_pred,
                                                          row.receptor_rim_pred)
                df_annotate['af_fn'] = row.af_fn
                df_annotate['idp_id'] = row.idp_id
                df_annotate['receptor_id'] = row.receptor_id
                results.append(df_annotate)
                
    return results

def main(idp_file, colabfold_dir, n_jobs, output, idp_id, receptor_id, is_zip):
    
    df = pd.read_csv(idp_file, sep='\t')
    
    if is_zip:
        results = []
        for job_name in tqdm(df["job_name"].unique().tolist(), desc="Processing files (ZIP)"):
            job_zip = os.path.join(colabfold_dir, job_name+".result.zip")
            df_job = df[df["job_name"] == job_name]
            job_results = process_job_zip(job_zip, df_job)
            results.extend(job_results)
    else:
        df["pdb_path"] = colabfold_dir + "/" + df["af_fn"]
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    annotate_electrostatics_pdb,
                    row.job_name,
                    row.pdb_path, # PDB PATH
                    row.idp_id,
                    row.idp_surface_pred,
                    row.idp_iface_pred,
                    row.idp_core_pred,
                    row.idp_rim_pred,
                    row.receptor_id,
                    row.receptor_surface_pred,
                    row.receptor_iface_pred,
                    row.receptor_core_pred,
                    row.receptor_rim_pred,
                ): (
                    row.af_fn, 
                    idp_id, 
                    receptor_id
                    ) 
                for row in df.itertuples()
            }
    
            results = []
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Annotating",
                unit="entries",
            ):
                try:
                    af_fn, idp_id, receptor_id = futures[future]
                    this_df = future.result()
                    this_df["af_fn"] = af_fn
                    this_df["idp_id"] = idp_id
                    this_df["receptor_id"] = receptor_id
                    results.append(this_df)
                except Exception as exc:
                    logger.exception(exc)
    
    df = pd.concat(results, ignore_index=True)
    df = df.pivot_table(
        index=["af_fn", "idp_id", "receptor_id"],
        columns=["residue_type", "chain"],
        values=["charged", "polar", "hydrophobic", "small", "hydropathy"],
        )
    df.columns = ["_".join(col[::-1]).strip() for col in df.columns.values]
    df = df.reset_index()
    df.to_csv(output, sep='\t', index=False)

if __name__ == "__main__":
    parser = ArgumentParser(description='Running electrostatics analysis for ColabFold predictions')
    parser.add_argument("colabfold_dir",
                        help='directory containing prediction files')
    parser.add_argument("input_file", 
                        help='.csv/.tsv file containing the IDP dataset. Use the interfaces output')
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
