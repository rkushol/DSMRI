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

import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import scipy
import pywt
from scipy.fftpack import fft2, fftshift
from scipy.signal import convolve2d
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image,convex_hull_object
from skimage import exposure as ex
from skimage.filters import median
from skimage.morphology import square
from skimage.feature import graycomatrix, graycoprops


class Extract_features(dict):

    def __init__(self, output_dir, v, ol, slice_gap, central_size):
        dict.__init__(self)

        self["warnings"] = [] 
        self["output"] = []
        self.addToPrintList("Patient", v[1], v, ol, 170)
        self["outdir"] = output_dir
        self["os_handle"] = v[0]
        self.addToPrintList("Name of Images", os.listdir(output_dir + os.sep + v[1]), v, ol, 100)
        self.addToPrintList("MEAN", vol(v, slice_gap, central_size, "Mean", output_dir), v, ol, 1)
        self.addToPrintList("RNG", vol(v, slice_gap, central_size, "Range", output_dir), v, ol, 2)
        self.addToPrintList("VAR", vol(v, slice_gap, central_size, "Variance", output_dir), v, ol, 3)
        self.addToPrintList("CV", vol(v, slice_gap, central_size, "CV", output_dir), v, ol, 4)
        self.addToPrintList("PSNR", vol(v, slice_gap, central_size, "PSNR", output_dir), v, ol, 5)
        self.addToPrintList("SNR1", vol(v, slice_gap, central_size, "SNR1", output_dir), v, ol, 6)
        self.addToPrintList("SNR2", vol(v, slice_gap, central_size, "SNR2", output_dir), v, ol, 7)
        self.addToPrintList("CNR", vol(v, slice_gap, central_size, "CNR", output_dir), v, ol, 8)
        self.addToPrintList("CJV", vol(v, slice_gap, central_size, "CJV", output_dir), v, ol, 9)
        self.addToPrintList("EFC", vol(v, slice_gap, central_size, "EFC", output_dir), v, ol, 10)
        self.addToPrintList("SNRF", vol(v, slice_gap, central_size, "SNRF", output_dir), v, ol, 11)
        self.addToPrintList("HFR", vol(v, slice_gap, central_size, "HFR", output_dir), v, ol, 12)
        self.addToPrintList("LFR", vol(v, slice_gap, central_size, "LFR", output_dir), v, ol, 13)
        self.addToPrintList("WQS", vol(v, slice_gap, central_size, "WQS", output_dir), v, ol, 14)
        self.addToPrintList("WCE", vol(v, slice_gap, central_size, "WCE", output_dir), v, ol, 15)
        self.addToPrintList("WCS", vol(v, slice_gap, central_size, "WCS", output_dir), v, ol, 16)
        self.addToPrintList("Contrast", vol(v, slice_gap, central_size, "Contrast", output_dir), v, ol, 17)
        self.addToPrintList("Correlation", vol(v, slice_gap, central_size, "Correlation", output_dir), v, ol, 18)
        self.addToPrintList("Energy", vol(v, slice_gap, central_size, "Energy", output_dir), v, ol, 19)
        self.addToPrintList("Homogeneity", vol(v, slice_gap, central_size, "Homogeneity", output_dir), v, ol, 20)
        self.addToPrintList("Dissimilarity", vol(v, slice_gap, central_size, "Dissimilarity", output_dir), v, ol, 21)
        self.addToPrintList("ASM", vol(v, slice_gap, central_size, "ASM", output_dir), v, ol, 22)
        
    def addToPrintList(self, name, val, v, ol, il):
        self[name] = val
        self["output"].append(name)
        if name != 'Name of Images' and il != 170:
            print('%s-%s. The %s of the subject with the name of <%s> is %s' % (ol, il, name, v[1], val))


def vol(v, slice_gap, central_size, feature_name, output_dir):
    switcher={
            'Mean': mean,
            'Range': rang,
            'Variance': variance, 
            'CV': percent_coefficient_variation,
            'PSNR': fpsnr,
            'SNR1': snr1,
            'SNR2': snr2,
            'CNR': cnr,
            'CJV': cjv,
            'EFC': efc,
            'SNRF': snrf,
            'HFR': hfr,
            'LFR': lfr,
            'WQS': wqs,
            'WCE': wce,
            'WCS': wcs,
            'Contrast': contrast,
            'Correlation': correlation,
            'Energy': energy,
            'Homogeneity': homogeneity,
            'Dissimilarity': dissimilarity,
            'ASM': asm,
            }
    func=switcher.get(feature_name)
    M = []
    start = int(0.005 *len(v[0])*(100 - central_size))
    finish = int(0.005 *len(v[0])*(100 + central_size))

    for i in range(start, finish, slice_gap):
        I = v[0][i]
        F, B, c, f, b = foreground(I, output_dir, v, i)
        #print(F.shape) #(256, 256)
        glcm = glcm_features(F)
        if np.std(F) == 0: 
            continue
        measure = func(F, B, c, f, b, glcm)
        if np.isnan(measure):
            continue
        if np.isinf(measure):
            continue

        M.append(measure)
    return np.mean(M)
       

def foreground(img, save_folder, v, inumber):
    try:
        h = ex.equalize_hist(img[:,:])*255
        oi = np.zeros_like(img, dtype=np.uint16)
        oi[(img > threshold_otsu(img)) == True] = 1
        oh = np.zeros_like(img, dtype=np.uint16)
        oh[(h > threshold_otsu(h)) == True] = 1
        nm = img.shape[0] * img.shape[1]
        w1 = np.sum(oi)/(nm)
        w2 = np.sum(oh)/(nm)
        ots = np.zeros_like(img, dtype=np.uint16)
        new =( w1 * img) + (w2 * h)
        ots[(new > threshold_otsu(new)) == True] = 1 
        conv_hull = convex_hull_image(ots)
        conv_hull = convex_hull_image(ots)
        ch = np.multiply(conv_hull, 1)
        fore_image = ch * img
        back_image = (1 - ch) * img
    except Exception: 
        fore_image = img.copy()
        back_image = np.zeros_like(img, dtype=np.uint16)
        conv_hull = np.zeros_like(img, dtype=np.uint16)
        ch = np.multiply(conv_hull, 1)
    
    return fore_image, back_image, conv_hull, img[conv_hull], img[conv_hull==False]


# Spatial domain

def mean(F, B, c, f, b, glcm):
    return np.nanmean(f)

def rang(F, B, c, f, b, glcm):
    return np.ptp(f)

def variance(F, B, c, f, b, glcm):
    return np.nanvar(f)

def percent_coefficient_variation(F, B, c, f, b, glcm):
    return (np.nanstd(f)/np.nanmean(f))*100

def psnr(img1, img2):
    mse = np.square(np.subtract(img1, img2)).mean()
    return 20 * np.log10(np.nanmax(img1) / np.sqrt(mse))

def fpsnr(F, B, c, f, b, glcm):
    I_hat = median(F/np.max(F), square(5))
    return psnr(F, I_hat)

def snr1(F, B, c, f, b, glcm):
    return np.nanstd(f) / np.nanstd(b)

def patch(img, patch_size):
    h = int(np.floor(patch_size / 2))
    U = np.pad(img, pad_width=5, mode='constant')
    [a,b]  = np.where(img == np.max(img))
    a = a[0]
    b = b[0]
    return U[a:a+2*h+1,b:b+2*h+1]

def snr2(F, B, c, f, b, glcm):
    fore_patch = patch(F, 5)
    return np.nanmean(fore_patch) / np.nanstd(b)

def cnr(F, B, c, f, b, glcm):
    fore_patch = patch(F, 5)
    back_patch = patch(B, 5)
    return np.nanmean(fore_patch-back_patch) / np.nanstd(back_patch)

def cjv(F, B, c, f, b, glcm):
    return (np.nanstd(f) + np.nanstd(b)) / abs(np.nanmean(f) - np.nanmean(b))

def efc(F, B, c, f, b, glcm):
    n_vox = F.shape[0] * F.shape[1]
    efc_max = 1.0 * n_vox * (1.0 / np.sqrt(n_vox)) * \
        np.log(1.0 / np.sqrt(n_vox))
    cc = (F**2).sum()
    b_max = np.sqrt(abs(cc))
    return float((1.0 / abs(efc_max)) * np.sum(
        (F / b_max) * np.log((F + 1e16) / b_max)))


# Frequency domain
# Signal-to-Noise Ratio in the Frequency domain   
def snrf(F, B, c, f, b, glcm):
    # Perform 2D fast Fourier transform (FFT) on the image
    image_fft = np.fft.fft2(F) 
    image_fft_shifted = np.fft.fftshift(image_fft)
    # Calculate the power spectrum
    power_spectrum = np.abs(image_fft_shifted)**2
    # Calculate the noise power spectrum
    noise_power_spectrum = power_spectrum[power_spectrum < np.mean(power_spectrum)]
    # Calculate the signal power spectrum
    signal_power_spectrum = power_spectrum[power_spectrum >= np.mean(power_spectrum)]
    # Calculate the SNR in the frequency domain
    #snr = np.mean(signal_power_spectrum) / np.mean(noise_power_spectrum)
    snr = 10*np.log10(np.mean(signal_power_spectrum) / np.mean(noise_power_spectrum))
    return snr

# High Frequency Response
def hfr(F, B, c, f, b, glcm):
    # Create a high-pass filter
    high_pass_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # Convolve the image with the high-pass filter
    high_pass_image = convolve2d(F, high_pass_filter, mode='same')
    # Calculate the amplitude spectrum of the high-pass image
    high_pass_image_fft = np.abs(np.fft.fft2(high_pass_image))
    # Normalize the amplitude spectrum
    #high_pass_image_fft_norm = high_pass_image_fft / np.max(high_pass_image_fft)
    # Calculate the SFR as the square root of the amplitude spectrum
    cal_hfr = np.sqrt(high_pass_image_fft)
    return np.nanmean(cal_hfr)
    
# Low Frequency Response
def lfr(F, B, c, f, b, glcm):
    # Define the 3x3 Gaussian filter
    gaussian_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
    low_pass_filter = gaussian_filter
    # Convolve the image with the low-pass filter
    low_pass_image = convolve2d(F, low_pass_filter, mode='same')
    # Calculate the amplitude spectrum of the high-pass image
    low_pass_image_fft = np.abs(np.fft.fft2(low_pass_image))
    # Normalize the amplitude spectrum
    #low_pass_image_fft_norm = low_pass_image_fft / np.max(low_pass_image_fft)
    # Calculate the LFR as the square root of the amplitude spectrum
    cal_lfr = np.sqrt(low_pass_image_fft)
    return np.nanmean(cal_lfr)   
    
    
# Wavelet domain    
# Wavelet Coefficient Energy
def wce(F, B, c, f, b, glcm, wavelet_type='coif4'):
    # Decompose the image into wavelet coefficients
    coeffs = pywt.wavedec2(F, wavelet_type)
    
    # Calculate the sum of the squares of the coefficients
    energy = 0
    for coeff in coeffs:
        energy += np.sum(np.abs(coeff))
    
    return energy / len(coeffs)   
    
# Wavelet Coefficient Sparsity
def wcs(F, B, c, f, b, glcm, wavelet='coif4'):
    # Decompose the image using the specified wavelet
    #The PyWavelets library provides several wavelets to choose from, including 'db1', 'db2', 'db3', 'db4', 'db5', and 'db6'.
    coefficients = pywt.wavedec2(F, wavelet)
    
    # Calculate the sparsity of the coefficients
    sparsity = 0
    for coeff in coefficients:
        sparsity += np.sum(np.abs(coeff) < np.mean(np.abs(coeff)))
    
    return sparsity / len(coefficients)
           
# Wavelet-based Quality Score
def wqs(F, B, c, f, b, glcm):
    # Decompose the image using the specified wavelet
    wavelet = 'coif4'
    coefficients = pywt.wavedec2(F, wavelet)
    
    # Calculate the magnitude and phase of the coefficients
    magnitude = [np.abs(coeff) for coeff in coefficients]
    phase = [np.angle(coeff) for coeff in coefficients]
    
    # Calculate the quality score based on the magnitude and phase
    score = 0
    for i in range(len(magnitude)):
        score += np.sum(magnitude[i] * np.cos(phase[i]))
    
    return score 


# Texture domain
def glcm_features(image, levels=256):
    # Define the distances and angles to consider for the GLCM
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]    
    glcm = graycomatrix(image.astype(np.uint8), distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    return glcm      
        
def contrast(F, B, c, f, b, glcm):    
    cont = graycoprops(glcm, 'contrast')
    return np.mean(cont)

def correlation(F, B, c, f, b, glcm):    
    corr = graycoprops(glcm, 'correlation')
    return np.mean(corr)
    
def energy(F, B, c, f, b, glcm):    
    ener = graycoprops(glcm, 'energy')
    return np.mean(ener)    
    
def homogeneity(F, B, c, f, b, glcm):   
    homoge = graycoprops(glcm, 'homogeneity')
    return np.mean(homoge)
       
def dissimilarity(F, B, c, f, b, glcm):    
    dissi = graycoprops(glcm, 'dissimilarity')
    return np.mean(dissi)
        
def asm(F, B, c, f, b, glcm):    
    asm = graycoprops(glcm, 'ASM')
    return np.mean(asm)

  