# sAMP-PFPDeep
A model for predicting short anti microbial peptides (length &lt;=30 residues) using multiple features and deep learning approaches

# To run sAMP-PFPDeep
1. Firstly download weights of the model from [HERE](https://drive.google.com/file/d/1o65-lXZ2Vvai1Jek9SrrIFXtgbtBhItm/view?usp=sharing), as size of weights is larger than 25MBs.
2. Put all sequences in FASTA format in a single file. 
3. Mention file name and correct file path in sAMP-PFPDeep.py in start at **FASTA_INPUT_FILE_NAME = "./Example.fasta"**
4. An example input file is given also. You can use this one for making predictions. 
5. The code requires Keras, BioPython, OpenCV and Pillow mainly to execute. If you face any issues with dependencies, ENV.yml has been provided so you can create new environment using this YAML file.
6. Thank you!
