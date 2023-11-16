'''
The code deeply follows and taken from the MRQy (https://github.com/ccipd/MRQy)
@article{sadri2020mrqy,
  title={MRQy—An open-source tool for quality control of MR imaging data},
  author={Sadri, Amir Reza and Janowczyk, Andrew and Zhou, Ren and Verma, Ruchika and Beig, Niha and Antunes, Jacob and Madabhushi, Anant and Tiwari, Pallavi and Viswanath, Satish E},
  journal={Medical physics},
  volume={47},
  number={12},
  pages={6029--6038},
  year={2020},
  publisher={Wiley Online Library}
}
'''

import os
import numpy as np
import argparse
import datetime
import features
import time
import umap
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from medpy.io import load
from scipy.cluster.vq import whiten
from sklearn.manifold import TSNE
import warnings        
warnings.filterwarnings("ignore") 

nfiledone = 0
csv_report = None
first = True
headers = []

def subject_info(input_dir):
    files = [os.path.join(dirpath,filename) for dirpath, _, filenames in os.walk(input_dir) 
                for filename in filenames 
                if filename.endswith('.mha')
                or filename.endswith('.nii')
                or filename.endswith('.gz')]
                
    subjects = [os.path.basename(scan)[:os.path.basename(scan).index('.')] for scan in files]    
    print('The total number of MRI samples is {}'.format(len(subjects)))
    return files, subjects


def volume(scan, name):
    image_data, image_header = load(scan)
    print('The image <{}> shape is {}'.format(name, image_data.shape))
    #print(image_data.shape)  #(256, 256, 172)
    images = [image_data[:,:,i] for i in range(np.shape(image_data)[2])]
    return images, name, image_header      


def saveThumbnails(v, output, slice_gap, central_size):
    os.makedirs(output + os.sep + v[1])
    start = int(0.005 *len(v[0])*(100 - central_size))
    finish = int(0.005 *len(v[0])*(100 + central_size))
    cnt = 1
    for i in range(start, finish, slice_gap):
        cnt = cnt + 1
        plt.imsave(output + os.sep + v[1] + os.sep + v[1] + '(%d).png' % int(i+1), scipy.ndimage.rotate(v[0][i],270), cmap = cm.Greys_r)
    
    print('The %d selected 2D slices are saved to %s' % (cnt, output + os.sep + v[1]))


def worker_callback(s,fname_outdir):
    global csv_report, first, nfiledone
    if nfiledone  == 0:
        csv_report = open(fname_outdir + os.sep + "results" + ".tsv" , overwrite_flag, buffering=1)
        first = True

    if first and overwrite_flag == "w": 
        first = False
        csv_report.write("\n".join(["#" + s for s in headers])+"\n")
        csv_report.write("#dataset:"+"\t".join(s["output"])+"\n")
                         
    csv_report.write("\t".join([str(s[field]) for field in s["output"]])+"\n")
    csv_report.flush()
    nfiledone += 1
    print('The results are updated.')
    

def tsv_to_dataframe(tsvfileaddress):
    return pd.read_csv(tsvfileaddress, sep='\t', skiprows=2, header=0)


def data_whitening(dframe):
    dframe = dframe.fillna('N/A')
    df = dframe.copy()
    df = df.select_dtypes(exclude=['object'])
    ds = whiten(df)
    return ds


def tsne_umap(dataframe, per):
    ds = data_whitening(dataframe)
    ds_umap = ds.copy()
    tsne = TSNE(n_components=2, random_state=0, perplexity = per)
    tsne_obj = tsne.fit_transform(ds)
    dataframe['x'] = tsne_obj[:,0].astype(float)
    dataframe['y'] = tsne_obj[:,1].astype(float)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(ds_umap)
    dataframe['u'] = embedding[:,0]
    dataframe['v'] = embedding[:,1]


def cleanup(data_address, per):
    df = tsv_to_dataframe(data_address)
    tsne_umap(df, per)
    hf = pd.read_csv(data_address, sep='\t',  nrows=1)
    hf.to_csv(data_address, index = None, header=True, sep = '\t', mode = 'w')
    df.to_csv(data_address, index = None, header=True, sep = '\t', mode = 'a')
    return df


if __name__ == '__main__':
    start_time = time.time() 
    headers.append(f"start_time:\t{datetime.datetime.now()}")
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('output_folder_name',
                        help = "the name of output subfolder on '.\\Results' directory.",
                        type=str)
    parser.add_argument('inputdir',
                        help = "input folder consists of MRI(*.nii) files. For example: 'D:\\Dataset\\input_folder'",
                        nargs = "*")  
    parser.add_argument('-b', help="number of gap in consecutive slices", default=5, type = int)
    parser.add_argument('-c', help="percent of central images", default=70)
    parser.add_argument('-p', help="perplexity for t-SNE method", default=30)
   
    args = parser.parse_args() 
    output_dir = os.getcwd() + os.sep + 'Results' + os.sep + args.output_folder_name
    
        
    if args.b != 5:
        slice_gap = args.b
    else: 
        slice_gap = 5
        
    if args.c != 70:
        central_size = int(args.c)
    else: 
        central_size = 70
        
    if args.p != 30:
        perplexity = int(args.p)
    else: 
        perplexity = 30
    
       
    overwrite_flag = "w"        
    headers.append(f"outdir:\t{os.path.realpath(output_dir)}") 
    files, subject_names  = subject_info(args.inputdir[0])    

    if len(files) == 0:
        print('The input folder is empty or includes unsupported file formats!')
    
    for l,k in enumerate(files):
        v = volume(k, subject_names[l])
        saveThumbnails(v,output_dir, slice_gap, central_size)
        s = features.Extract_features(output_dir, v, l+1, slice_gap, central_size)
        worker_callback(s,output_dir)
        
    address = output_dir + os.sep + "results" + ".tsv" 
            
    if len(subject_names) < perplexity:
        print('Insufficient data for t-SNE and UMAP. The UMAP and t-SNE methods require the total samples ', 
        'greater than perplexity. The default value of perplexity is 30.')
        df = tsv_to_dataframe(address)
    else:        
        df = cleanup(address, perplexity)
        df = df.drop(['Name of Images'], axis=1)
        df = df.rename(columns={"#dataset:Patient": "Patient", 
                                "x":"TSNEX","y":"TSNEY", "u":"UMAPX", "v":"UMAPY" })
        df = df.fillna('N/A')
        
    df.to_csv(output_dir + os.sep +'features.csv',index=False)
    
    print("The features are saved in the {} file. ".format(output_dir + os.sep + "features.csv"))    
    print("DSMRI program took", format((time.time() - start_time)/60, '.2f'), \
          "minutes for {} MRI samples to run.".format(len(subject_names)))
    
    
    
