#!/bin/bash

# input file name - will be used to access directories as well
INPUT_FILE=$1
FASTA_FILE=$2
COLABFOLD_DIR="./colabfold/${INPUT_FILE}"

# check if input argument was provided
if [ -z "$INPUT_FILE" ]; then
	echo "Error: No file name provided"
	echo "Script used for uniprot_idp_domain_receptor"
	echo "Usage: $0 <main_filename> <fasta_file> [--zip]"
	echo "List of flags that can be indicated:"
	echo "- --zip: set if files are saved in zip files"
	exit 1
fi

# check if directory exists
if [ ! -d "$COLABFOLD_DIR" ]; then
	echo "Error: Directory does not exist - ${COLABFOLD_DIR}"
	exit 1
fi

if [ ! -f "$FASTA_FILE" ]; then
	echo "Error: FASTA file does not exist"
	exit 1
fi

# Set flag variable - will be empty string if not provided
FLAG_OPTION=$3 # optional flag argument
PYTHON_FLAG=""
if [ ! -z "$FLAG_OPTION" ]; then
	PYTHON_FLAG="$FLAG_OPTION"
	echo "Running with flags: $PYTHON_FLAG"
fi

# 
# get confidence scores
CONFIDENCE_SCORES="./data/${INPUT_FILE}_confidence_scores.tsv"
if [ ! -f "$CONFIDENCE_SCORES" ]; then
	echo "Getting confidence scores..."
	python get_af_scores.py "$FASTA_FILE" "$COLABFOLD_DIR" "$CONFIDENCE_SCORES" $PYTHON_FLAG
else
	echo "Confidence scores already obtained for ${INPUT_FILE}"
fi

# get pyrosetta
PYROSETTA_SCORES="./data/${INPUT_FILE}_interaction_analysis.tsv"
if [ ! -f "$PYROSETTA_SCORES" ]; then
	echo "Running PyRosetta analysis..."
	python run_pyrosetta.py "$COLABFOLD_DIR" --output_file "$PYROSETTA_SCORES" --n_workers 10 $PYTHON_FLAG
else
	echo "PyRosetta analysis already completed for ${INPUT_FILE}"
fi

INTERFACE="./data/${INPUT_FILE}_interface.tsv"
if [ ! -f "$INTERFACE" ]; then
	echo "Running interface analysis..."
	python annotate_iface_af.py "$COLABFOLD_DIR" "$PYROSETTA_SCORES" "$INTERFACE" --n_workers 10 $PYTHON_FLAG
else
	echo "Interface analysis already completed for ${INPUT_FILE}"
fi

ELECTROSTATICS="./data/${INPUT_FILE}_electrostatics.tsv"
if [ ! -f "$ELECTROSTATICS" ]; then
	echo "Running electrostatics analysis..."
	python annotate_electrostatics.py "$COLABFOLD_DIR" "$INTERFACE" "$ELECTROSTATICS" --n_workers 10 $PYTHON_FLAG
else
	echo "Electrostatics analysis already completed for ${INPUT_FILE}"
fi
