import os
import warnings

import numpy as np
from Bio.PDB import PDBExceptions
from Bio.PDB import PDBParser

BACKBONE_IDS = ['CA', 'C', 'N']
warnings.filterwarnings("ignore", category=PDBExceptions.PDBConstructionWarning)

AT_ = []


class PDBBackbone(object):
    def __init__(self, pdb_path):
        self.filename = pdb_path
        pdb_id = os.path.basename(self.filename).replace('.pdb', "")
        self.structure = PDBParser().get_structure(pdb_id.upper(), self.filename)
        self.model = None

    def get_pdb_features(self, segments=None):
        atom_list = []
        seq_list = []
        seq_positions = []
        labels = []
        try:
            self.model = self.structure[0]
        except KeyError:
            return None, None, None, None

        for chain in self.model.get_list():
            for residue in chain.get_list():
                n_coord = None
                ca_coord = None
                c_coord = None
                o_coord = None
                for atom in residue:
                    full_id = atom.get_full_id()
                    _, _, seq_id, seq_pos, atom_id = full_id
                    seq_pos = seq_pos[1]
                    atom_id = atom_id[0]
                    if n_coord is not None and ca_coord \
                            is not None and c_coord is not None and o_coord is not None:
                        atom_list.append([n_coord, ca_coord, c_coord, o_coord])
                        seq_id = seq_id.lower()
                        seq_name = residue.get_resname()
                        label = self.get_segment_label(segments, seq_id, seq_pos)
                        seq_list.append(seq_name)
                        labels.append(label)
                        seq_positions.append(f'{seq_id.upper()}{seq_pos}')
                        break

                    if atom_id == 'N' and not n_coord:
                        n_coord = atom.get_coord()
                    elif atom_id == 'CA' and not ca_coord:
                        ca_coord = atom.get_coord()
                    elif atom_id == 'C' and not c_coord:
                        c_coord = atom.get_coord()
                    elif atom_id == 'O' and not o_coord:
                        o_coord = atom.get_coord()
                    else:
                        continue

        assert len(labels) == len(atom_list) == len(seq_list)
        labels = np.array(labels)
        if not atom_list:
            # print(f'Empty file: {self.filename}')
            return None, None, None, None
        if segments:
            if np.sum(labels) == 0:
                print(f'Without segments: {self.filename}')
                return None, None, None, None
        features = np.stack(atom_list)
        return features, seq_list, labels, seq_positions

    @staticmethod
    def get_segment_label(segments, seq_id, seq_pos):
        if not segments:
            return 0
        for seg in segments:
            start, end = seg
            liter = start[0]
            pos_start = int(start[1:])
            pos_end = int(end[1:])
            if liter == seq_id:
                if pos_start <= seq_pos <= pos_end:
                    return 1
        return 0
