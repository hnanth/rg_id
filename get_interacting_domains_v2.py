# Get domains by aligning the interface sequence and the domain sequence
# check the % overlap (by number of interface residues and by overall)

import pandas as pd
from utils import check_range_relationship, get_start_end

df = pd.read_csv("rg_id_iface_seq_v3.csv")
df = df.dropna()
df_domains = pd.read_csv("./data/rg_id_all_pfam_domains.csv")

def max_consecutive_overlap(seq1, seq2):
    max_len = 0
    max_interacting_residues = 0
    
    for shift in range(-len(seq2)+1, len(seq1)):
        current_len = 0
        current_interacting_residues = 0
        for i in range(len(seq1)):
            j = i - shift
            if 0 <= j < len(seq2):
                if seq1[i] != "-" and seq2[j] != "-":
                    if seq1[i] == seq2[j]:
                        current_len += 1
                        current_interacting_residues += 1
                    elif seq1[i].upper() == seq2[j] or seq1[i] == "?": # add to overlap if one base is ? or one is lower case but same letter
                        current_len += 1
                        max_len = max(max_len, current_len)
                    else:
                        current_len = 0 # reset streak on mismatch
                        current_interacting_residues = 0
                elif (seq1[i] == "-" or seq2[j] == "-") and current_len >= 2:
                    current_len += 1
                else:
                    current_len = 0 # reset streak on mismatch
                    current_interacting_residues = 0

                max_len = max(max_len, current_len)
                max_interacting_residues = max(max_interacting_residues, current_interacting_residues)
            else:
                current_len = 0 # reset streack on mismatch or gap
                current_interacting_residues = 0
    return max_len, max_interacting_residues

def get_interacting_domains(all_domains: dict, uniprot_ranges, iface_seq):
    total_interact = len([x for x in iface_seq if x.isupper()])

    # results
    interacting_domains = []

    for domain in all_domains:
        domain_seq = domain["domain_sequence"]
        start, end = domain["domain_location"].split("-")

        # check if domain is within or has overlaps or contains the pdb seq
        range_relationships = check_range_relationship(int(start), int(end), uniprot_ranges)
        if len(range_relationships) == 0:
            continue

        max_len, max_interacting_residues = max_consecutive_overlap(iface_seq, domain_seq)
        perc_max_overlap = max_len / min([len(iface_seq), len(domain_seq)]) # percentage overlap based on the smaller sequence
        perc_max_interacting = max_interacting_residues / total_interact

        if max_interacting_residues >= 5:
            interacting_domains.append({
                "domain_name": domain["domain_name"],
                "domain_id": domain["domain_id"],
                "domain_location": domain["domain_location"],
                "max_sequence_overlap": max_len,
                "perc_sequence_overlap": perc_max_overlap,
                "max_interacting_residues": max_interacting_residues,
                "perc_interacting_residues": perc_max_interacting
            })

    return interacting_domains

results = []
for row in df.itertuples():
    idp_uniprot_ranges = get_start_end(row.idp_uniprot)
    receptor_uniprot_ranges = get_start_end(row.receptor_uniprot)

    idp_uniprot = next((r.split(":")[0] for r in row.idp_uniprot.split(";")), None)
    receptor_uniprot = next((r.split(":")[0] for r in row.receptor_uniprot.split(";")), None)

    # IDP
    idp_domains = df_domains[df_domains["uniprot"] == idp_uniprot]

    if not idp_domains.empty:
        idp_domains = idp_domains.to_dict(orient="records")
        idp_interacting_domains = get_interacting_domains(idp_domains, idp_uniprot_ranges, row.idp_iface_seq)
        for idp in idp_interacting_domains:
            results.append({
                "pdb_id": row.pdb_id,
                "idp_uniprot": row.idp_uniprot,
                "receptor_uniprot": row.receptor_uniprot,
                "chain": "idp",
                "chain_id": row.idp_id,
                **idp
            })
    else:
        idp_interacting_domains = []

    # receptor
    receptor_domains = df_domains[df_domains["uniprot"] == receptor_uniprot]

    if not receptor_domains.empty:
        receptor_domains = receptor_domains.to_dict(orient="records")
        receptor_interacting_domains = get_interacting_domains(receptor_domains, receptor_uniprot_ranges, row.receptor_iface_seq)
        for receptor in receptor_interacting_domains:
            results.append({
                "pdb_id": row.pdb_id,
                "idp_uniprot": row.idp_uniprot,
                "receptor_uniprot": row.receptor_uniprot,
                "chain": "receptor",
                "chain_id": row.receptor_id,
                **receptor
            })
    else:
        receptor_interacting_domains = []

results_df = pd.DataFrame(results)
results_df.to_csv("rg_id_interacting_domains_v2.tsv", sep='\t', index=False)
        