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

def max_consecutive_overlap(iface, seq, max_errors=2):
    """
    seq1: iface
    seq2: sequence of interest
    """
    max_len = 0
    max_interacting_residues = 0
    
    for shift in range(-len(seq)+1, len(iface)):
        current_len = 0
        current_interacting_residues = 0
        errors = 0
        for i in range(len(iface)):
            j = i - shift
            if 0 <= j < len(seq):
                if iface[i] != "-" and seq[j] != "-":
                    if iface[i] == seq[j]:
                        current_len += 1
                        current_interacting_residues += 1
                    elif iface[i].upper() == seq[j] or iface[i] == "?": # add to overlap if one base is ? or one is lower case but same letter
                        current_len += 1
                        max_len = max(max_len, current_len)
                    else:
                        current_len = 0 # reset streak on mismatch
                        current_interacting_residues = 0
                elif (iface[i] == "-" or seq[j] == "-") and current_len >= 2:
                    current_len += 1
                else:
                    errors += 1
                    if errors <= max_errors:
                        current_len += 1 # allow for mismatch (threshold set by max_errors)
                    else:
                        # reset streak if more than 
                        current_len = 0 
                        current_interacting_residues = 0

                max_len = max(max_len, current_len)
                max_interacting_residues = max(max_interacting_residues, current_interacting_residues)
            else:
                current_len = 0 # reset streak on mismatch or gap
                current_interacting_residues = 0
                
    return max_len, max_interacting_residues