def check_range_relationship(start, end, ranges):
    results = []

    for r_start, r_end in ranges:
        if r_start <= start and end <= r_end:
            results.append(("within", (r_start, r_end)))

        elif start <= r_start and r_end <= end:
            results.append(("contains", (r_start, r_end)))

        elif max(start, r_start) <= min(end, r_end):
            results.append(("overlaps", (r_start, r_end)))

    return results

def get_start_end(s):
    list_ranges = []
    for r in s.split(";"):
        _, pdb_start, length, start = r.split(":")
        start = int(start)
        end = start + int(length)
        list_ranges.append((start, end))
    return list_ranges

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