# DSMRI
DSMRI: Domain Shift Analyzer for Multi-Center MRI Datasets

The paper has been published in the Journal of Diagnostics.

Link: https://doi.org/10.3390/diagnostics13182947


## Abstract
In medical research and clinical applications, the utilization of MRI datasets from multiple centers has become increasingly prevalent. However, inherent variability between these centers presents challenges due to domain shift, which can impact the quality and reliability of the analysis. Regrettably, the absence of adequate tools for domain shift analysis hinders the development and validation of domain adaptation and harmonization techniques. To address this issue, this paper presents a novel Domain Shift analyzer for MRI (DSMRI) framework designed explicitly for domain shift analysis in multi-center MRI datasets. The proposed model assesses the degree of domain shift within an MRI dataset by leveraging various MRI-quality-related metrics derived from the spatial domain. DSMRI also incorporates features from the frequency domain to capture low- and high-frequency information about the image. It further includes the wavelet domain features by effectively measuring the sparsity and energy present in the wavelet coefficients. Furthermore, DSMRI introduces several texture features, thereby enhancing the robustness of the domain shift analysis process. The proposed framework includes visualization techniques such as t-SNE and UMAP to demonstrate that similar data are grouped closely while dissimilar data are in separate clusters. Additionally, quantitative analysis is used to measure the domain shift distance, domain classification accuracy, and the ranking of significant features. The effectiveness of the proposed approach is demonstrated using experimental evaluations on seven large-scale multi-site neuroimaging datasets.


![Proposed Workflow](https://github.com/rkushol/DSMRI/assets/76894940/ee57d137-5a1d-49ee-9758-fdbdbd74bf6b)


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
nibabel
PyWavelets 


To create a new conda environment: `conda create -n dsmri python=3.9`

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



