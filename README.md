# DSMRI
Domain Shift analyzer for MRI

## Abstract



## Requirements
medpy  
pandas  
scikit-learn  
umap-learn  
numpy  
scipy  
matplotlib  
scikit-image  


To create a new conda environment: `conda create -n dsmri python=3.7`

To install all the packages from the requirements.txt file: `pip3 install -r requirements.txt`

## Datasets
ADNI, AIBL, PPMI and ABIDE datasets can be downloaded from [ADNI](http://adni.loni.usc.edu/) (Alzheimer’s Disease Neuroimaging Initiative)

CALSNIC dataset can be requested from [CALSNIC](https://calsnic.org/) (Canadian ALS Neuroimaging Consortium)


## Command
Run `python dsmri.py output_folder_name “input directory”`. It will generate `features.csv` and `results.tsv` in the `Results/output_folder_name` folder.



## Contact
Email at: kushol@ualberta.ca

## Acknowledgement
This basic structure of the code deeply relies on the project of [MRQy](https://github.com/ccipd/MRQy).



