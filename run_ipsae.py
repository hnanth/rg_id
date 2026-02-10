import zipfile
import tempfile
import re
from collections import defaultdict
import subprocess
import os
import pandas as pd
from argparse import ArgumentParser

zip_path = ""
pae_cutoff = 10
dist_cutoff = 10
model_seed_re = re.compile(r"model[_-]?(\d+).*seed[_-]?(\d+)", re.IGNORECASE)

def extract_model_seed(name: str):
    m = model_seed_re.search(name)
    if not m:
        return int(m.group(1)), int(m.group(2))
    return None

def run_ipsae_zip(zip_path):
    with tempfile.TemporaryDirectory() as tmpdir:

        grouped_files = defaultdict(list)

        with zipfile.ZipFile(zip_path, "r") as z:
            # extract only pdb files and score files
            for file in z.namelist():
                if not file.endswith(".pdb") or not (file.endswith(".json") and "scores" in file):
                    continue
                ms = extract_model_seed(file)
                out_path = z.extract(file, path=tmpdir)
                grouped_files[ms].append(out_path)

        for (model, seed), files in grouped_files.items():
            if len(files) != 2:
                print("Check again")

            if files[0].endswith('.pdb'):
                pdb_path = os.path.join(tmpdir, files[0])
                pae_path = os.path.join(tmpdir, files[1])
            else:
                pdb_path = os.path.join(tmpdir, files[1])
                pae_path = os.path.join(tmpdir, files[0])

            output_path = pdb_path.split('.')[0] + f"_{pae_cutoff}_{dist_cutoff}.txt"
            #subprocess.run(['python', 'ipsae.py', pdb_path, pae_path, pae_cutoff, dist_cutoff])
            #ipsae_results = pd.read_csv(os.path.join(tmpdir, output_path), sep='\s+')
            print(files)

if __name__ == '__main__':
    parser = ArgumentParser(prog='Running ipSAE analysis')
    parser.add_argument('colabfold_dir',
                        help='directory containing prediction files')
    parser.add_argument('--zip',
                        action='store_true',
                        help='set if files are contained in .result.zip files')
    args = parser.parse_args()

    if args.zip:   
        files = [f for f in os.listdir(args.colabfold_dir) if f.endswith('.result.zip')]
        for f in files:
            zip_path = os.path.join(args.colabfold_dir, f)
            run_ipsae_zip(zip_path)
            break