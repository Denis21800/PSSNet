import os.path
from pathlib import Path

import numpy as np
from Bio.PDB import Select, PDBIO
from Bio.PDB.PDBParser import PDBParser


class ChainSelect(Select):
    def __init__(self, position):
        start, end = position
        self.chain = start[0]
        self.start_idx = int(start[1:])
        self.end_idx = int(end[1:])

    def accept_chain(self, chain):
        if chain.get_id() == self.chain:
            return 1
        else:
            return 0

    def accept_residue(self, residue):
        _, res_id, _ = residue.get_id()
        if self.start_idx <= int(res_id) <= self.end_idx:
            return 1
        return 0


class PDBSelect(object):
    TRES_SUM = 20
    MAX_PAD = 2

    def __init__(self, pdb_path, positions, out_dir):
        assert pdb_path
        assert positions
        assert out_dir
        parser = PDBParser(PERMISSIVE=1)
        self.structure = parser.get_structure(pdb_path, pdb_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
        self.positions = positions
        filename = os.path.basename(pdb_path).upper()
        self.pdb_id = filename.replace('.PDB', '')

    def upload2pdb(self):
        for pos in self.positions:
            pdb_extract_file = self.out_dir / f'{self.pdb_id}_{pos[0]}_{pos[1]}.pdb'
            io_w_no_h = PDBIO()
            io_w_no_h.set_structure(self.structure)
            io_w_no_h.save(str(pdb_extract_file), ChainSelect(pos))


def find_index(prediction_pos, sequence, tres_sum=32, tres_pad=4):
    if np.sum(prediction_pos) < tres_sum:
        return None, None
    prediction_pos = np.array(prediction_pos)
    index_positions = np.argwhere(prediction_pos > 0)
    first_pos = index_positions[0][0]
    if not first_pos:
        return None, None
    total_seq = []
    seq_positions = []
    prev_pos = first_pos
    for i in range(1, len(index_positions)):
        next_pos = index_positions[i][0]

        if abs(next_pos - prev_pos) < tres_pad:
            if not seq_positions:
                seq_positions.append(prev_pos)
            seq_positions.append(next_pos)
        else:
            total_seq.append(seq_positions.copy())
            seq_positions.clear()
        prev_pos = next_pos

    total_seq.append(seq_positions.copy())
    result_positions = []
    result_index = []
    for item_seq in total_seq:
        if len(item_seq) > tres_sum:
            start_idx, end_idx = item_seq[0], item_seq[-1]
            start_seq = sequence[start_idx]
            end_seq = sequence[end_idx]
            result_positions.append((start_seq, end_seq))
            result_index.append((start_idx, end_idx))

    return result_positions, result_index
