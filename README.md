# PSSNet (Protein supersecondary structure segmentation) 
![alt text](https://github.com/Denis21800/PSSNet/blob/master/Logo/pssnet.png)


PSSNet – A new deep learning network architecture for semantic segmentation of supersecondary structures in proteins with small size, very high accuracy and performance.
**The PSSNet model made it possible to identify and classify more than 2.3 million supersecondary structures in whole protein structures annotated in the PDB and AlphaFold 2.0 banks and became the basis for filling the public database of supersecondary structures https://www.psskb.org**
## Model architecture

![alt text](https://github.com/Denis21800/PSSNet/blob/master/Logo/arcitecture.png)


## Run model
This repository contains a pretrained model for segmenting: 
-aa-corners
-a-hairpins 
-b-hairpins
-BaB
To extract the specified structures from the folder containing the PDB, you need to run the sss_extractor.py script.

usage: sss_extractor.py [-h] -pdb PDB -out OUT [-sss SSS]

Extract SSS from PDB

options:
-h, --help  show this help message and exit
-pdb PDB    Path to the folder with the PDB database
-out OUT    Path to the output folder
-sss SSS    The type of supersecondary structure to be extracted from the
              PDB aa-corner', 'a-hairpin', 'b-hairpin', 'bab


The output directory will contain files of pdb format with cut coordinates of supersecondary structures.


## Train model
To train the model, it is necessary to prepare training datasets:
- Set with positive examples
- Set with negative examples


Set with positive examples pdb format files containing the coordinates of the supersecondary structure in the file name
File name format: _pdb id chain code start pos chain code end pos.pdb_

**Example:** 1a0a_A7_A60.pdb

A set with negative examples is any pdb format files that do not contain super secondary structures.

The following parameters must be defined in the config.py file:
- root_dir = _root directory that contains the training datasets._
- positive_keys=_name of subdirectory containing positive examples._ 
- negative_keys= _name of subdirectory containing negative examples_
After specifying these parameters, you must run the command

**$python preprocessor.py**

The features subdirectory will be formed containing preprocessed files for model training.

To start the model training process, you need to execute the command:

**$python trainer.py**

The training process of the model and the metrics of each epoch of its training will be displayed in the console.

The model with the best measures based on training and validation results will be saved in the _models_ folder.
