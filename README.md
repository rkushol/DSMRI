# DSMRI
Domain Shift analyzer for MRI

(The complete code will be released after the acceptance of the paper)

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
seaborn  


To create a new conda environment: `conda create -n dsmri python=3.7`

To install all the packages from the requirements.txt file: `pip3 install -r requirements.txt`

## Datasets
ADNI, AIBL, PPMI and ABIDE datasets can be downloaded from [ADNI](http://adni.loni.usc.edu/) (Alzheimer’s Disease Neuroimaging Initiative)

CALSNIC dataset can be requested from [CALSNIC](https://calsnic.org/) (Canadian ALS Neuroimaging Consortium)


## Command
Run `python dsmri.py output_folder_name “input directory”`. For example, `python dsmri.py CALSNIC1 "D:\DS_MRI\Dataset\CALSNIC1"`. It will generate `features.csv` and `results.tsv` in the `Results/CALSNIC1` folder.

## Hyper-parameters
`-b`-> number of gap in consecutive slices, default= 5. The features will be extracted from an interval of five slices by default. Setting the value as `1` will consider every slice in the range.   
`-c`-> percent of central images, default= 70. The first 15% and last 15% slices will not be considered by default. Setting the value as `100` will start the slice range from first to last.  
`-p`-> perplexity for t-SNE method, default= 30. The t-SNE method requires the total samples to be greater than the value of perplexity. The perplexity is related to the number of nearest neighbours. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50.

## Contact
Email at: kushol@ualberta.ca

## Acknowledgement
This basic structure of the code deeply relies on the project of [MRQy](https://github.com/ccipd/MRQy).



