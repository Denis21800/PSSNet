import argparse
import os
import sys
from config import ModelConfig
from pdb_processor import extract

parser = argparse.ArgumentParser(description='Extract SSS from PDB')
parser.add_argument('-pdb', type=str, help='Path to the folder with the PDB database', required=True)
parser.add_argument('-out', type=str, help='Path to the output folder', required=True)
parser.add_argument('-sss', type=str, help="""The type of supersecondary structure to be retrieved from the PDB
    aa-corner', 'a-hairpin', 'b-hairpin', 'bab""", default='b-hairpin', )
args = parser.parse_args()
pdb_path = args.pdb
if not os.path.exists(pdb_path):
    print(f'PDB folder not found: {pdb_path}')
    sys.exit(0)

output = args.out

if not os.path.exists(output):
    print(f'Output folder not found: {output}')
    sys.exit(0)


sss_type = args.sss
config = ModelConfig()

if sss_type not in config.models_types:
    print(f'Invalid SSS type: {sss_type} Valid types: {config.models_types}')
    sys.exit(0)

extract(
    pdb_path=pdb_path,
    out_dir=output,
    sss_type=sss_type
)