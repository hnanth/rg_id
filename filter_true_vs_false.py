import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm 
from utils import check_range_relationship, get_start_end, max_consecutive_overlap

def get_domain_entry(location, domain_dict, chain_key):
    domain_dict = {k: v.split(";") if isinstance(v, str) else [str(v)] for k, v in domain_dict.items() if chain_key in k}
    idx = domain_dict[f"{chain_key}_domain_locations"].index(location)
    domain_entry = {k: v[idx] if len(v) > 1 else v[0] for k, v in domain_dict.items()}
    return domain_entry

def trim_to_start_end_iface(s):
    import re
    first = re.search(r"[A-Z]", s)
    if not first:
        return ""
    last = None
    for match in re.finditer(r"[A-Z]", s):
        last = match
    return s[first.start():last.end()]

def filter_complexes(df_pred, df_domains, df_iface, 
                     overlap_threshold, overlap_criteria, sequence_type,
                     chain1):

    chain2 = "receptor" if chain1 == "idp" else "idp"
    df_pred["chain_1"] = chain1
    df_pred["chain_2"] = chain2
    
    # pre-initialize columns 
    df_pred[f"{chain1}_range"] = "-"
    df_pred[f"{chain1}_fragment_overlap"] = 0
    df_pred[f"{chain1}_interacting_residues"] = 0
    df_pred[f"{chain1}_perc_interacting_residues"] = 0.0
    df_pred[f"{chain1}_perc_overlap_iface"] = 0.0 # max overlap between iface and domain / length iface
    df_pred[f"{chain1}_perc_overlap_fragment"] = 0.0 # max overlap between iface and fragment / length fragment
    df_pred[f"{chain2}_fragment_overlap"] = 0
    df_pred[f"{chain2}_interacting_residues"] = 0
    df_pred[f"{chain2}_perc_interacting_residues"] = 0.0
    df_pred[f"{chain2}_perc_overlap_iface"] = 0.0 # max overlap between iface and domain / length iface
    df_pred[f"{chain2}_perc_overlap_domain"] = 0.0 # max overlap between iface and domain / domain iface
       
    df_pred["interacting_complex"] = False
    df_pred["pdb_id"] = None
    
    # convert to records once before loop 
    if not df_iface.empty:
        df_iface_dict = {row["pdb_id"]: row for row in df_iface.to_dict("records")}
        
    for row in tqdm(df_pred.itertuples(), total=len(df_pred), desc="Processing predictions"):
        # get row index
        i = row.Index
        
        interacting = False
        pdb_id = None
        
        # analyze chain2 by search interacting domain by domain name and location 
        interacting_domain_2 = df_domains[(df_domains[f"{chain1}_uniprot"].str.contains(row.uniprot_1)) 
                                      & (df_domains[f"{chain2}_uniprot"].str.contains(row.uniprot_2)) 
                                      & (df_domains[f"{chain2}_interacting_domains"].str.contains(row.domain_2))
                                      & (df_domains[f"{chain2}_domain_locations"].str.contains(row.location_2))]
        
        # skip if the domain is not found in df_iface
        if not interacting_domain_2.empty:
            interacting_domain_2_dict = interacting_domain_2.iloc[0].to_dict()
            domain_2 = get_domain_entry(row.location_2, interacting_domain_2_dict, chain2)
            
            chain2_max_interacting_residues = float(domain_2[f"{chain2}_max_interacting_residues"]) # max. no of interacting residues
            chain2_perc_interacting_residues = float(domain_2[f"{chain2}_perc_interacting_residues"]) # perc of interacting residues in domain / total num.
            overlap_value = float(domain_2[f"{chain2}_{overlap_criteria}"])
            
            if overlap_value >= overlap_threshold:
                interacting = True
                pdb_id = interacting_domain_2_dict["pdb_id"]
        
        if interacting:
            if sequence_type == "fragment":                
                start, end = row.location_1.split("-")
                fragment_seq = row.sequence.split(":")[0]
                iface = df_iface_dict.get(pdb_id)
                
                # check if fragment is within the uniprot ranges
                uniprot_ranges = get_start_end(iface[f"{chain1}_uniprot"])
                range_relationship = check_range_relationship(int(start), int(end), uniprot_ranges)
                
                # sequence
                full_iface_seq = iface[f"{chain1}_iface_seq"] # entire pdb seq with iface annotations
                trimmed_iface_seq = trim_to_start_end_iface(full_iface_seq) # trimmed iface defined by start and end interacting residues
                iface_interacting_residues = sum(1 for x in full_iface_seq if x.isupper())
                
                if len(range_relationship) > 0:
                    df_pred.loc[i, f"{chain1}_range"] = str(range_relationship)
                    
                    # fragment vs pdb
                    max_len, max_interacting_residues = max_consecutive_overlap(full_iface_seq, fragment_seq)
                    
                    # fragment vs iface
                    max_len_iface, _ = max_consecutive_overlap(trimmed_iface_seq, fragment_seq) 
                    
                    perc_interacting_residues = max_interacting_residues / iface_interacting_residues
                    
                    if overlap_criteria == "perc_interacting_residues":
                        if perc_interacting_residues >= overlap_threshold:
                            interacting = True
                            pdb_id = interacting_domain_2_dict["pdb_id"]
                        else:
                            interacting = False
                            pdb_id = None
                    else: # overlap_criteria == "max_interacting_residues")
                        if max_interacting_residues >= overlap_threshold:
                            interacting = True
                            pdb_id = interacting_domain_2_dict["pdb_id"]
                        else:
                            interacting = False
                            pdb_id = None
                else:
                    interacting = False
                    pdb_id = None
                
            else:
                df_pred.loc[i, f"{chain1}_fragment_overlap"] = -1 # all
                df_pred.loc[i, f"{chain1}_interacting_residues"] = interacting_domain_2_dict[f"{chain1}_num_interface_residues"]
                df_pred.loc[i, f"{chain1}_perc_interacting_residues"] = 1
                df_pred.loc[i, f"{chain1}_perc_overlap_iface"] = 1
                df_pred.loc[i, f"{chain1}_perc_fragment_iface"] = 1 # all 
                
        # update if value changes
        if interacting and pdb_id is not None:
            df_pred.loc[i, "interacting_complex"] = interacting
            df_pred.loc[i, "pdb_id"] = pdb_id
            df_pred.loc[i, f"{chain2}_fragment_overlap"] = float(domain_2[f"{chain2}_max_sequence_overlaps"])
            df_pred.loc[i, f"{chain2}_interacting_residues"] = chain2_max_interacting_residues
            df_pred.loc[i, f"{chain2}_perc_interacting_residues"] = chain2_perc_interacting_residues
            df_pred.loc[i, f"{chain2}_perc_overlap_iface"] = float(domain_2[f"{chain2}_perc_max_overlap_ifaces"]) # max overlap between iface and domain / length iface
            df_pred.loc[i, f"{chain2}_perc_overlap_domain"] = float(domain_2[f"{chain2}_perc_max_overlap_domain_ifaces"]) # max. overlap between iface and domain / length domain
            if sequence_type == "fragment":
                df_pred.loc[i, f"{chain1}_fragment_overlap"] = max_len
                df_pred.loc[i, f"{chain1}_interacting_residues"] = max_interacting_residues
                df_pred.loc[i, f"{chain1}_perc_interacting_residues"] = perc_interacting_residues # interacting residues found in fragment / total interacting residues
                df_pred.loc[i, f"{chain1}_perc_overlap_iface"] = max_len_iface / len(trimmed_iface_seq) # max. overlap between iface and fragment / length iface
                df_pred.loc[i, f"{chain1}_perc_overlap_fragment"] = max_len_iface / len(fragment_seq) # max. overlap between iface and fragment / length fragment 
        
    return df_pred

if __name__ == "__main__":
    parser = ArgumentParser(prog="Filter interactions")
    parser.add_argument("predictions_csv",
                        help="CSV/TSV file containing all predictions")
    parser.add_argument("output_csv",
                        help="CSV/TSV file for output")
    parser.add_argument("--domains_csv",
                        default="./rg_id/rg_id_interacting_domains_v3.tsv",
                        help="CSV/TSV file containing interacting domains")
    parser.add_argument("--interface_csv",
                        default="./rg_id/rg_id_iface_seq_v3_sasa.csv",
                        help="CSV/TSV file containing interface sequences")
    parser.add_argument("--threshold",
                        default=5,
                        type=float,
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
    df_domains = pd.read_csv(args.domains_csv, sep="\t" if args.domains_csv.endswith(".tsv") else ",")
    
    
    # only keep certain columns in df_pred
    cols = ["uniprot_1", "domain_1", "location_1",
            "uniprot_2", "domain_2", "location_2",
            "sequence", "job_name"]
    df_pred = df_pred[cols]
    if args.sequence_type == "full":
        results = filter_complexes(df_pred, df_domains, df_iface=pd.DataFrame(), overlap_threshold=args.threshold,
                         overlap_criteria=search_criteria, sequence_type=args.sequence_type, chain1=args.chain_1)
    else:
        df_iface = pd.read_csv(args.interface_csv)
        results = filter_complexes(df_pred, df_domains, df_iface=df_iface, overlap_threshold=args.threshold,
                         overlap_criteria=search_criteria, sequence_type=args.sequence_type, chain1=args.chain_1)
        
    results.to_csv(args.output_csv, sep="\t" if args.output_csv.endswith(".tsv") else ",", index=False)