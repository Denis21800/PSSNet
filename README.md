# PSSNet (Protein supersecondary structure segmentation) 
![alt text](https://github.com/Denis21800/PSSNet/blob/master/Logo/pssnet.png)

# Run model
This repository contains a pre-trained model for segmenting aa corners in pdb files. To run the model, you need to specify the following parameters in the config.py file:
pdb_base = <path to directory containing pdb files>
out_dir = <path to output directory>
After specifying these parameters, run the command:
$ python pdb_processor.py
