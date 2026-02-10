import pandas as pd

df1 = pd.read_csv("./data/rg_id_interfaces.tsv", sep='\t') # obtained all interface: residue contacts < 5A between chains
df2 = pd.read_csv("rg_id_iface_seq_v2.csv") # defined by solvent accessible surface area

def count_interact(seq):
    interact = [x for x in seq if x.isupper()]
    return len(interact)

def find_first_matching_base(seq1, seq2, ignore="-"):
    for i in range(len(seq1) - 1):
        if seq1[i] == ignore or seq1[i + 1] == ignore:
            continue

        for j in range(len(seq2) - 1):
            if seq2[j] == ignore or seq2[j + 1] == ignore:
                continue

            if seq1[i] == seq2[j] and seq1[i + 1] == seq2[j + 1]:
                return i, j

    return None, None

for i in range(len(df1)):
    df2_match = df2[df2["pdb_id"] == df1.loc[i, 'pdb_id']]

    if df2_match.empty:
        print("no input for pdb", df1.loc[i, "pdb_id"])
        continue

    df2_match = df2_match.to_dict(orient="records")[0]

    print("PDB:", df1.loc[i, "pdb_id"])
    for chain in ["idp", "receptor"]:
        seq1 = df1.loc[i, f"{chain}_interface_sequence"]
        seq2 = df2_match[f"{chain}_iface_seq"]

        start_idx1, start_idx2 = find_first_matching_base(seq1, seq2)
        trim_seq1 = seq1[start_idx1:]
        trim_seq2 = seq2[start_idx2:]

        print("start1:", start_idx1)
        print("start2:", start_idx2)
        print("trim_seq1:", seq1[start_idx1:])
        print("trim_seq2:", seq2[start_idx2:])

        if start_idx1 != 0:
            print(seq1)
        
        if start_idx2 != 0:
            print(seq2)

        interface_overlap = 0
        overlap = 0
        for a, b in zip(trim_seq1, trim_seq2):
            if a == "-" or b == "-":
                overlap += 1
            elif a == b:
                interface_overlap += 1
                overlap += 1
        
        print("number of interacting residues in seq1:", count_interact(seq1))
        print("number of interacting residues in seq2:", count_interact(trim_seq2))
        print(f"{chain} interface overlap: {interface_overlap}, overlap in general: {overlap}")
        print(f"seq1 length: {len(seq1)}, seq2 length: {len(seq2)}")