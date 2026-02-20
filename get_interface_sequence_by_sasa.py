#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 11:44:36 2026

@author: huyennhu
"""

# new script to analyze interfaces for domains
import pandas as pd
from argparse import ArgumentParser
from Bio.PDB import FastMMCIFParser
from Bio.PDB.Polypeptide import index_to_one, three_to_index

def three_to_one(residue):
    return index_to_one(three_to_index(residue))

def annotate_iface_mmcif(pdb_id, mmcif_path, idp_id, receptor_id, idp_iface, receptor_iface):
    # read the mmCIF file
    parser = FastMMCIFParser(QUIET=True, auth_chains=True, auth_residues=False)
    structure = parser.get_structure(pdb_id, mmcif_path)
    model = structure[0]

    return annotate_iface(model, idp_id, receptor_id, idp_iface, receptor_iface)

def group_residues(model, chain):
    # list all residue_ids available
    residue_ids = [residue.get_id()[1] for residue in model[chain].get_residues()]
    last_start = None
    last_end = None
    formatted = []
    for res_id in residue_ids:
        if last_start is None:
            last_start = res_id
            last_end = res_id
        elif res_id == last_end + 1:
            last_end = res_id
        else:
            formatted.append((last_start, last_end))
            last_start = res_id
            last_end = res_id
            
    if last_start is not None and last_end is not None:
        formatted.append((last_start, last_end))

    return formatted
    
def get_individual_residues(residues):
    list_temp = residues.split(":")
    list_residues = []
    for residues in list_temp:
        start, end = residues.split("-")
        if start == end:
            list_residues.append(int(start))
        else:
            list_residues.extend(list(range(int(start), int(end)+1)))
    return sorted(list_residues)

def get_interface_sequence(model, iface, chain):
    iface_residues = set(get_individual_residues(iface))
    sequence = []

    prev_res_id = None

    for residue in model[chain].get_residues():
        residue_id = residue.get_id()[1]

        # fill gaps based on residue numbering
        if prev_res_id is not None and residue_id > prev_res_id + 1:
            sequence.extend("-" * (residue_id - prev_res_id - 1))

        try:
            aa = three_to_one(residue.get_resname())
        except Exception:
            aa = "?"

        if residue_id in iface_residues:
            if aa == "?":
                sequence.append("!")
            else:
                sequence.append(aa)
        else:
            sequence.append(aa.lower())

        prev_res_id = residue_id

    return "".join(sequence)


def annotate_iface(model, idp_id, receptor_id, idp_iface, receptor_iface):
    
    idp_iface_seq = get_interface_sequence(model, idp_iface, idp_id)
    receptor_iface_seq = get_interface_sequence(model, receptor_iface, receptor_id)

    return (idp_iface_seq, receptor_iface_seq)
    
    
def main(csv_file, mmcif_dir):
    
    df = pd.read_csv(csv_file)
    
    results = []
    for row in df.itertuples():
        idp, receptor = annotate_iface_mmcif(row.pdb_id,
                                             f"{mmcif_dir}/{row.pdb_id.lower()}.cif",
                                             row.idp_id,
                                             row.receptor_id,
                                             row.idp_iface,
                                             row.receptor_iface)
        print(row.pdb_id, idp)
        print(row.pdb_id, receptor)
        results.append({
            "pdb_id": row.pdb_id,
            "idp_id": row.idp_id,
            "receptor_id": row.receptor_id,
            "idp_iface": row.idp_iface,
            "idp_iface_seq": idp,
            "receptor_iface": row.receptor_iface,
            "receptor_iface_seq": receptor,
        })

    
    results_df = pd.DataFrame(results)
    results_df.to_csv("rg_id_iface_seq_v2.csv", index=False)
    
main("rg_id_interfaces_v2.csv", "./data/mmCIF")
