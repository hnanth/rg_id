import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm 
from utils import check_range_relationship, get_start_end, max_consecutive_overlap

def filter_complexes(df_pred, df_domains, df_iface, 
                     overlap_threshold, overlap_criteria, sequence_type,
                     chain1):

    for row in df_pred.itertuples():
        # analyze chain2
        interacting_domain2 = df_domains[(df_domains["idp_uniprot"].str.contains(row.uniprot_1)) 
                                      & (df_domains["receptor_uniprot"].str.contains(row.uniprot_2)) 
                                      & (df_domains["domain_name"] == row.domain_2)
                                      & (df_domains["domain_location"] == row.location_2)]
        
        # skip if the domain is not found in df_iface
        if not interacting_domain2.empty:
            interacting_domain2 = interacting_domain2.to_dict(orient="records")
            print(len(interacting_domain2))
        else:
            continue

        if interacting_domain2[0][overlap_criteria] >= overlap_threshold:
            interacting = True
            pdb_id = interacting_domain2[0]["pdb_id"]
        else:
            interacting = False
            pdb_id = None
        
        # analyze chain 1
        if sequence_type == "fragment" and interacting:
            start, end = row.location_1.split("-")
            iface = df_iface[df_iface["pdb_id"] == pdb_id].to_dict(orient="records")[0]

            uniprot_ranges = get_start_end(iface[f"{chain1}_uniprot"])
            range_relationship = check_range_relationship(int(start), int(end), uniprot_ranges)
            iface_interacting_residues = len([x for x in iface[f"{chain1}_iface_seq"] if x.isupper()])
            
            if len(range_relationship) == 0:
                interacting = False
                pdb_id = None

            max_len, max_interacting_residues = max_consecutive_overlap(row.sequence.split(":")[0], iface[f"{chain1}_iface_seq"])
            perc_interacting_residues = max_interacting_residues / iface_interacting_residues

            if (perc_interacting_residues >= overlap_threshold and 0 <= overlap_threshold <= 1) or \
            (max_interacting_residues >= overlap_threshold and overlap_threshold > 1):
                interacting = True
                pdb_id = interacting_domain2[0]["pdb_id"]
            else:
                interacting = False
                pdb_id = None

            print(max_len, max_interacting_residues, perc_interacting_residues)

    pass

if __name__ == "__main__":
    parser = ArgumentParser(prog="Filter interactions")
    parser.add_argument("predictions_csv",
                        help="CSV/TSV file containing all predictions")
    parser.add_argument("--domains_csv",
                        default="rg_id_interacting_domains_v2.tsv",
                        help="CSV/TSV file containing interacting domains")
    parser.add_argument("--interface_csv",
                        default="rg_id_iface_seq_v3.csv",
                        help="CSV/TSV file containing interface sequences")
    parser.add_argument("--threshold",
                        default=5,
                        help="Threshold for interacting domains.\nIf threshold < 1 -> search by overlap.\nIf threshold > 1 -> search by number of contacts. Default: 5 contacts.")
    parser.add_argument("--sequence_type",
                        choices=["full", "fragment"],
                        default="full",
                        help="Set whether sequence analyzed is full-length or fragment.\nIf full-length, only need to analyze second chain.\nIf fragment, both chains need to be analyzed.")
    parser.add_argument("--chain_1",
                        choices=["idp", "receptor"],
                        default="idp",
                        help="Set chain 1 as IDP or receptor. Default: idp")
    args = parser.parse_args()

    search_criteria = "perc_interacting_residues" if 0 <= args.threshold <= 1 else "max_interacting_residues"
    df_pred = pd.read_csv(args.predictions_csv, sep="\t" if args.predictions_csv.endswith(".tsv") else ",")
    df_domains = pd.read_csv(args.domains_tsv, sep="\t")
    if args.sequence_type == "full":
        filter_complexes(df_pred, df_domains, df_iface=None, overlap_threshold=args.threshold,
                         overlap_criteria=search_criteria, sequence_type="full", chain1=args.chain_1)
    else:
        df_iface = pd.read_csv(args.interface_csv)
        filter_complexes(df_pred, df_domains, df_iface=df_iface, overlap_threshold=args.threshold,
                         overlap_criteria=search_criteria, sequence_type="full", chain1=args.chain_1)