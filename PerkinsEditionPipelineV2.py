#Importing packages
#System libraries

### TODOS:
### Make datafolder a part of the call for the script rather than a predefined variable -- DONE
### Add filter input option (or way to check for all filters not just BVR) -- DONE
### Quadruple check cross-correlation and probably do centroiding for galaxy images
### Implement PSF matching to deal with diff filters and potential focus issues, do this before aligning master frames to each other.
### Add a parameter to skip doing the calibration steps.
### Add a testing script/mode for the pipeline

#IF BREAK REPLACE TQDM.write WITH print

  
import os
import os.path
import glob
import numpy as np
from tqdm import tqdm
import math
import warnings
import argparse
import time
import shutil

#Numpy,matplotlib,etc.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from IPython.display import Image

#Astropy
#from astropy.visualization import ZScaleInterval
import astropy
from astropy.io import fits
import astropy.stats as stat
from astropy.stats import mad_std
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from astropy.convolution import convolve
from astropy.visualization import SimpleNorm
from astropy.stats import SigmaClip


from scipy.ndimage import shift
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
import scipy.signal

import photutils.psf_matching as psf
from photutils.utils import calc_total_error
from photutils.psf import CircularGaussianSigmaPRF
from photutils.psf_matching import make_kernel, make_wiener_kernel, SplitCosineBellWindow
from photutils.aperture import aperture_photometry, ApertureStats, CircularAperture, CircularAnnulus
from photutils.detection import DAOStarFinder
from photutils.profiles import CurveOfGrowth
from photutils.background import Background2D, MedianBackground


from skimage.registration import phase_cross_correlation

warnings.filterwarnings("ignore", category=RuntimeWarning)
terminal_width = os.get_terminal_size().columns
textbar = terminal_width * "-"


parser = argparse.ArgumentParser(description="Set default options")
# parser.add_argument('-d', '--datafolder', \
#                         default=None, help='File path to folder containing the night\'s images')
parser.add_argument('-n', '--objname', \
                        default=None, help='Name of object to run code on')
parser.add_argument('-w', '--writefiles', default=True, \
                        help='Whether to save intermediate files other than master images')
parser.add_argument('-r', '--rsubtraction', default=False, \
                        help='Whether to use r subtraction instead of Ha (subtraction)')
parser.add_argument('-s', '--skipreduction', \
                        help = 'Whether to skip data reduction and start from  shift/stacking (True/False)', default=False)
parser.add_argument('-o', '--omitstacking', \
                        help = 'Whether to skip stacking and shifting or to just use existing master ims', default = False)


parser.add_argument('datafolder')
                        
opt = parser.parse_args()


#Define the master bias code (median of bias frames)

def master_bias_generator(filelist, savepath):
    '''
    This function generates the master bias by taking the median of an input of bias frames
    Inputs:
    filelist (array): list of files to take the median of and generate their master bias
    savepath (String): the filepath to save the generated masterbias to
    Outputs:
    master_bias (array): the master bias in array form
    '''
    pbar = tqdm(total = len(filelist), desc = "Master Bias Generation", leave = False)
    # Setting the 3rd dimensions of the array to the number of input files
    n = len(filelist)
    #It sets the standard frame to be the first one in the list
    first_frame_data = fits.getdata(filelist[0])
    #Defines the dimension for the median array to be the same as the standard frame
    imsize_y, imsize_x = first_frame_data.shape
    #Creates an array of zeroes to store all the others,
    #with size determined by the standard frame and the number of files in the list
    fits_stack = np.zeros((imsize_y, imsize_x , n))
    #Sets the zeros in the previous array to each value of the filelist arrays
    for ii in range(0, n):
        fits_stack[:,:,ii] = fits.getdata(filelist[ii])
        pbar.update(1)
    #Finds the median of all the arrays in the combined array and sets that to a variable
    master_bias = np.median(fits_stack, axis = 2)
    fits.writeto(savepath + '/master_bias.fit', master_bias, fits.getheader(filelist[0]), overwrite=True)
    pbar.close()
    return master_bias

#Define the master flat file

def master_dark_generation(filelist, path_to_bias, savepath):
    '''
    Subtracts the master bias from a given file, then divides by exposure time to generate a single dark frame and adds it to a 3d array,
    then takes the median of this array
    Inputs:
    filelist (array): list of files to take the median of and generate their master bias
    path_to_bias (string): the file path of the master bias
    savepath (string): the file path to save the master dark frame to
    Outputs:
    master_dark (array): the master dark in array form
    '''
    n = len(filelist) 
    pbar = tqdm(total = n, desc = "Master Dark Generation", leave = False)
    #Imports the data from the first dark to retrieve the header and image size
    #Determing the shape of the overall frame
    first_frame_data = fits.getdata(filelist[0])
    imsize_y, imsize_x = first_frame_data.shape
    fits_stack = np.zeros((imsize_y, imsize_x,n))
    #Takes in the files, header, and master bias from their respective paths
    rawheader = fits.getheader(filelist[0])
    master_bias = fits.getdata(path_to_bias)

    for ii in range(0, n):
        im_exptime = fits.getheader(filelist[ii])['EXPTIME']
        fits_stack[:,:,ii] = (fits.getdata(filelist[ii]) - master_bias)/im_exptime
        pbar.update(1)

    #Computes the overall median
    master_dark = np.median(fits_stack, axis=2)
    #Writes that to an output in the same directory
    fits.writeto(savepath +  '/master_dark.fit', master_dark, rawheader, overwrite=True)
    return master_dark

#Define the flat fielding median function

def flatfieldingALL(filelist,path_to_master_bias,savepath):
    '''
    Creates a master flat field file given a filelist, master bias, and master dark.
    Inputs:
    filelist (list,str): list of filepaths for flat files
    path_to_master_bias (str): filepath to master bias
    path_to_master_dark (str): filepath to master dark
    Outputs: None
    Creates a fits file of the normalized master flat data.
    '''
    # Gets header of first file to find filter and exposure time
    n = len(filelist)
    #print(n)
    header = fits.getheader(filelist[0])
    filter = header['FILTNME1']
    if filter == "Empty":
        filter = header['FILTNME3']

    if filter == 'New H Alpha (On)' or filter == 'H-Alpha On (New)':
        color = 'magenta'
    if filter == 'New H Alpha (Off)' or filter == 'H-Alpha Off(New)':
        color = 'cyan'
    if filter == "B":
        color = 'blue'
    if filter == "V":
        color = 'green'
    if filter == "R":
        color = 'red'

    pbar = tqdm(total = n, desc = filter+" Master Flat Generation", colour = color)#, leave = False)

    # Creates data array to store all data from images
    datafiller = fits.getdata(filelist[0])
    imsize_y, imsize_x = datafiller.shape
    fits_stack = np.zeros((imsize_y, imsize_x,n))

    # Gets data from the master bias/dark fits files.
    databias = fits.getdata(path_to_master_bias)
    #datadark = fits.getdata(path_to_master_dark)
                    
    # Iterates through all files in filelist, getting data, then bias/dark subtraction and accounting for expt before adding to data array
    for ii in np.arange(0,n):
        expt = fits.getheader(filelist[ii])['EXPTIME']
        fits_stack[:,:,ii] = ((fits.getdata(filelist[ii]) - databias))
        pbar.update(1)

    # Creates unnormalized median file of all flats.
    unnorm = np.median(fits_stack,axis=2)
    # Normalizes the unnormalized file
    master_flat = unnorm/np.median(unnorm)
    
    # Writes the fits files out, including the expt and filter.
    fits.writeto(savepath + '/master_flat_' + str(filter)+'_'+ str(expt)+'.fit', master_flat, header, overwrite = True)
    return

#Define the image calibration function 

def reducing(im,path_to_master_bias,path_to_flat_folder,output_path):
    '''
    Creates a master image file given a master bias, master dark, and folder containing the master flat
    Inputs:
    im (str): filepath to science images
    path_to_master_bias (str): filepath to master bias
    path_to_master_dark (str): filepath to master dark
    path_to_flat_folder (str): filepath to the folder of flats (which will have the master flats)
    output_path (str): the output path for the reduced image (should just be the image path)
    Outputs: None
    Creates a fits file of the reduced science image with filepath of the file with fdb_ at the start of the filename.
    '''

    # Gets header from image to find exposure time and filter
    header = fits.getheader(im)
    expt = header['EXPTIME']
    filter = header['FILTNME1']
    if filter == "Empty":
        filter = header['FILTNME3']

    # Uses the filter of given images to find and extract the data for the given master flat
    if filter == 'New H Alpha (On)' or filter == 'H-Alpha On (New)':
        flatdata = fits.getdata(glob.glob(path_to_flat_folder+'/HaON/master_flat*')[0])
    if filter == 'New H Alpha (Off)' or filter == 'H-Alpha Off(New)':
        flatdata = fits.getdata(glob.glob(path_to_flat_folder+'/HaOFF/master_flat*')[0])
    if filter == "B":
        flatdata = fits.getdata(glob.glob(path_to_flat_folder+'/B/master_flat*')[0])
    if filter == "V":
        flatdata = fits.getdata(glob.glob(path_to_flat_folder+'/V/master_flat*')[0])
    if filter == "R":
        flatdata = fits.getdata(glob.glob(path_to_flat_folder+'/R/master_flat*')[0])

    #Gets data from image and master bias, dark, and flats.
    imdata = fits.getdata(im)
    biasdata = fits.getdata(path_to_master_bias)
    #darkdata = fits.getdata(path_to_master_dark)
    #imbias = imdata - biasdata

    # Reduces the image by bias subtracting, dark subtracting and accounting for expt, then flatfielding.
    imreduced = ((imdata - biasdata)/flatdata)
    imreduced = imreduced[::,55:2100]

    
    #Writes data to a fits file with the same filepath of image name + _fdb at the start of the filename.

    fits.writeto(os.path.dirname(output_path)+'/fdb_' + os.path.basename(output_path), imreduced, header, overwrite=True)

    return



def timenorm(im,output_path):
    '''
    Normalizes inputted image by exposure time
    Inputs: 
    im (str): filepath to image
    output_path (str): output path for normalized image (should be same as im)
    Outputs: None
    Creates fits file with n_ prefix
    '''
    #extracts exposure time
    header = fits.getheader(im)
    expt = header['EXPTIME']
    #gets image data and divides by expt
    imdata = fits.getdata(im)
    im_norm = imdata/expt

    #outputs image with n_ at front
    fits.writeto(os.path.dirname(output_path)+'/n_' + os.path.basename(output_path), im_norm, header, overwrite=True)


#Define the centroid function 
def cross_image(im1, im2):
    
    """
    Calculates the x- and y-shifts between two images using cross-correlation

    Parameters:
    - im1 (np.ndarray): First image (reference)
    - im2 (np.ndarray): Second image (to be aligned)

    Returns:
    - xshift (float): The x-shift needed to align im2 w/ im1
    - yshift (float): The y-shift needed to align im2 w/ im1
    
    """

    # Ensure images have the same shape
    im1 = fits.getdata(im1)
    xshape1, yshape1 = im1.shape
    im2 = fits.getdata(im2)
    xshape2, yshape2 = im2.shape
    #plt.figure()
    #plt.imshow(im1,vmin=0,vmax=3)
    #plt.figure()
    #plt.imshow(im2,vmin=0,vmax=3)

    im1 = im1[200:xshape1-200,200:yshape1-200]
    im2 = im2[200:xshape2-200,200:yshape2-200]

    # if objname == "standard":
    #     im1 = im1[900:xshape1-900,900:yshape1-900]
    #     im2 = im2[900:xshape2-900,900:yshape2-900]        

    if im1.shape != im2.shape:
        raise ValueError(f"Image dimensions do not match: {im1.shape} vs {im2.shape}")

    # Replace NaNs with zeros and create masks
    im1_mask = ~np.isnan(im1)
    im2_mask = ~np.isnan(im2)
    if im1_mask.all == True:
        print("NO MASK (1)")
    if im2_mask.all == True:
        print("NO MASK (2)")
    im1 = np.nan_to_num(im1)
    im2 = np.nan_to_num(im2)

    # Apply Gaussian high-pass filter to remove low-frequency variations
    im1_hp = im1 - gaussian_filter(im1, sigma=2) #is usually 10
    im2_hp = im2 - gaussian_filter(im2, sigma=2)

    # Normalize intensity ranges
    im1_hp = (im1_hp - np.mean(im1_hp)) / np.std(im1_hp)
    im2_hp = (im2_hp - np.mean(im2_hp)) / np.std(im2_hp)

    # Compute shift using phase cross-correlation with masks
    shifts, error, diffphase = phase_cross_correlation(
        im1_hp, im2_hp, reference_mask=im1_mask, moving_mask=im2_mask
    )

    xshift, yshift = shifts[1], shifts[0]

    fits.writeto(datafolder + '/cross_corr1.fits', im1_hp, overwrite=True)   
    fits.writeto(datafolder + '/cross_corr2.fits', im2_hp, overwrite=True)
    # print("******")
    # print(xshift,yshift)
    # print("******")
    return [xshift, yshift]


def centroid_one_star(ref_image,target_image,star_bound,background_bound):
    '''
    This function takes the centroid of the area specified in a reference image (intended to be one star), 
    then takes the centroid of the same area in the specified list of images, 
    and calculates the shifts required to map them to the reference image.
    Inputs:
    ref_image (str): the filepath of the reference image
    target_image (str): the filepath of the target image
    star_bound (np array): the bounds for the box of which to centroid, in [[xmin,xmax],[ymin,ymax]] form
    background_bound (np array): the bounds for the box of which to calculate the background from, in [[xmin,xmax],[ymin,ymax]] form
    Outputs:
    xy_shift (list): the x and y shifts for the target image
    '''
    ref_fits = fits.getdata(ref_image)
    #Extracts reference star bounds and background
    #star_bound = star_bound.astype(int)
    #background_bound = background_bound.astype(int)
    ref_background = ref_fits[background_bound[1][0]:background_bound[1][1],background_bound[0][0]:background_bound[0][1]]
    ref_star = ref_fits[star_bound[1][0]:star_bound[1][1],star_bound[0][0]:star_bound[0][1]]
    
    #First find the background of the reference
    ref_med = np.median(ref_background)
    ref_std = np.std(ref_background)
    ref_threshhold = ref_med + 3*ref_std    
    #Now set up the array for our weighted/unweighted pixels
    ref_x_weight = []
    ref_y_weight = []
    ref_no_weight = []

     #Nested for loops for x/y weighted pixels
    for y in np.arange(0,ref_star.shape[1]):
        for x in np.arange(0,ref_star.shape[0]):
            if ref_star[y,x] > ref_threshhold:
                x_weight = (ref_star[y,x]-ref_med)*x
                y_weight = (ref_star[y,x]-ref_med)*y
                no_weight= ref_star[y,x]-ref_med
                ref_x_weight.append(x_weight)
                ref_y_weight.append(y_weight)
                ref_no_weight.append(no_weight)


    #Calculate the reference image's centroid 
    ref_x_centroid = sum(ref_x_weight)/sum(ref_no_weight)
    ref_y_centroid = sum(ref_y_weight)/sum(ref_no_weight)
    ref_centroid = [ref_y_centroid,ref_x_centroid]
    
    #Calculate the shifts as compared to the reference centroid
    
    #Set our sum arrays to nothing
    img_x_weight = []
    img_y_weight = []
    img_no_weight = []
        
    #Converting the fits to a np.array
    img_fits = fits.getdata(target_image)

    #Extracing star+background patches
    img_background = img_fits[background_bound[1][0]:background_bound[1][1],background_bound[0][0]:background_bound[0][1]]
    img_star = img_fits[star_bound[1][0]:star_bound[1][1],star_bound[0][0]:star_bound[0][1]]

    img_med = np.median(img_background)
    img_std = np.std(img_background)
    img_threshhold = img_med + 3*img_std
        
    #Nested for loops for x/y weighted pixels
    for y in np.arange(0,img_star.shape[1]):
        for x in np.arange(0,img_star.shape[0]):
            if img_star[y,x] > img_threshhold:
                x_weight = (img_star[y,x]-img_med)*x
                y_weight = (img_star[y,x]-img_med)*y
                no_weight= img_star[y,x]-img_med
                img_x_weight.append(x_weight)
                img_y_weight.append(y_weight)
                img_no_weight.append(no_weight)
                
    #Calculate the image's centroid
    # print(sum(img_x_weight))
    # print(sum(img_no_weight))
    img_x_centroid = sum(img_x_weight)/sum(img_no_weight)
    img_y_centroid = sum(img_y_weight)/sum(img_no_weight)
    img_centroid = [img_y_centroid,img_x_centroid]
        
    #Calculate the x/y shifts, print them, and add them to the list
    x_shift = ref_centroid[1] - img_centroid[1]
    y_shift = ref_centroid[0] - img_centroid[0]
    xy_shift = [x_shift,y_shift]
    return xy_shift

#Define the image registration/stacking function

def simple_image_combination(img_list, img_offset, pad_size, save_path):
    '''
    This function uses the inputted offsets and padding size to align the given images in imglist, then writes the new images as 
    registered ones and calculates the stacked median. Finally, it writes the registered image to the specified file path and returns it as an array.
    Inputs:
    imglist (list): the list of filepaths for the images to be shifted
    imgoffset (list): the list of x/y offsets for a given image
    pad_size (float): the number of pixels to pad each image so as to allow for shifting
    save_path (str): the path to save the overall registered image
    Outputs:
    final_median (array): the final registered image as an numpy array
    '''
    #Ensuring the padding size is an integer
    pad_size = int(pad_size)
    
    #Determining the filter type for our image
    image_header = fits.getheader(img_list[0])
    image_type = image_header['FILTNME1']
    if image_type == "Empty":
        image_type = image_header['FILTNME3']
    
    if filter == 'New H Alpha (On)' or filter == 'H-Alpha On (New)':
        color = 'magenta'
    if filter == 'New H Alpha (Off)' or filter == 'H-Alpha Off(New)':
        color = 'cyan'
    if filter == "B":
        color = 'blue'
    if filter == "V":
        color = 'green'
    if filter == "R":
        color = 'red'
    
    pbar = tqdm(total = len(img_list), desc = image_type +" Image Stacking", colour = color)#, leave = False)

    #Setting up our median
    median_stack = []
    
    #For loop for shifting each image
    for n in np.arange(0,len(img_list)):
        file = img_list[n]
        image = fits.getdata(file)
        
        #Removing artifact values
        image[np.isinf(image)] = 0.0
        image[np.isnan(image)] = 0.0
        
        #Padding + shifting
        padded_image = np.pad(image, pad_size, 'constant', constant_values = 0.001)
        new_image = shift(padded_image, (float(img_offset[n,0]),float(img_offset[n,1])), cval = 0.001)
        
        #Appending this image to our median list
        median_stack.append(new_image)
        
        #Writing the new image with the prefix reg_ (for registered)
        fits.writeto(os.path.dirname(file)+'/reg_' + os.path.basename(file), 
                     new_image, fits.getheader(file),overwrite=True)
        #print("Registered: " + str(file))
        pbar.update(1)

    #Median stacking/combination
    standard_image = fits.getdata(img_list[0])
    y, x = standard_image.shape
    median_array = np.array(median_stack)
   
    final_median = np.nanmedian(median_array, axis=0)

    #Writing the registered image to the specified path, with name corresponding to the filter type
    fits.writeto(save_path+'/registered_'+image_type+'.fits', final_median, image_header,overwrite=True)

    #Output the image
    return final_median


def reg_and_stack(file_list, offsets, padding_size, stacked_path):

    '''
    Register and stack a list of FITS images with given offsets and padding
    
    Parameters:
    -----------
    file_list : list of str
        List of file paths to input FITS images to be registered and stacked
    offsets : list of tuples
        List of (y_offset, x_offset) values for each image
    padding_size : int
        Size of padding to be applied around images before shifting
    stacked_path : str
        File path where final stacked FITS image will be saved

    Returns:
    --------
    None
    
    '''

    registered_images = []

    # Load first image to get original dimensions
    ref_image = fits.getdata(file_list[0])
    image_header = fits.getheader(file_list[0])
    original_shape = ref_image.shape
    image_type = image_header['FILTNME1']
    if image_type == "Empty":
        image_type = image_header['FILTNME3']

    if image_type == 'New H Alpha (On)' or image_type == 'H-Alpha On (New)':
        color = 'magenta'
    if image_type == 'New H Alpha (Off)' or image_type == 'H-Alpha Off(New)':
        color = 'cyan'
    if image_type == "B":
        color = 'blue'
    if image_type == "V":
        color = 'green'
    if image_type == "R":
        color = 'red'
    
    pbar = tqdm(total = len(file_list), desc = image_type + " Image Stacking", colour = color)#, leave = False)
    

    for i, file in enumerate(file_list):
        header = fits.getheader(file)
        im_now = fits.getdata(file)
        shift_y, shift_x = offsets[i]
        shift_y = shift_y
        shift_x = shift_x

        # Handle NaNs
        im_now[np.isnan(im_now)] = 0

        # Pad image
        median_value = np.median(im_now[np.isfinite(im_now)])
        padded_image = np.pad(im_now, padding_size, mode='constant', constant_values=median_value)

        # Shift image
        shifted_image = shift(padded_image, (shift_y, shift_x), cval=median_value)

        # Crop back to original size
        pad_y, pad_x = padding_size, padding_size
        cropped_image = shifted_image[pad_y: pad_y + original_shape[0], pad_x: pad_x + original_shape[1]]
        
        fits.writeto(os.path.dirname(file)+'/newreg_' + os.path.basename(file), cropped_image, header, overwrite=True)
        # Add to list of registered images
        registered_images.append(cropped_image)
        pbar.update(1)

    # Stack images
    stack_array = np.stack(registered_images, axis=2)
    stacked_median = np.median(stack_array, axis=2)

    # Replace NaNs in final stacked image
    stacked_median[np.isnan(stacked_median)] = np.median(stacked_median[np.isfinite(stacked_median)])

    # Save stacked image
    header = fits.getheader(file_list[0])
    save_path = stacked_path
    fits.writeto(save_path+'/registered_'+image_type+'.fits', stacked_median, header, overwrite=True)
    return

def get_psf(target_image,star_bound,background_bound,size):
    '''
    This function takes the centroid of the area specified in a reference image (intended to be one star), 
    then takes the centroid of the same area in the specified list of images, 
    and calculates the shifts required to map them to the reference image.
    Inputs:
    ref_image (str): the filepath of the reference image
    target_image (str): the filepath of the target image
    star_bound (np array): the bounds for the box of which to centroid, in [[xmin,xmax],[ymin,ymax]] form
    background_bound (np array): the bounds for the box of which to calculate the background from, in [[xmin,xmax],[ymin,ymax]] form
    Outputs:
    xy_shift (list): the x and y shifts for the target image
    '''
    
    #Calculate the shifts as compared to the reference centroid
    
    #Set our sum arrays to nothing
    img_x_weight = []
    img_y_weight = []
    img_no_weight = []
        
    #Converting the fits to a np.array
    img_fits = fits.getdata(target_image)

    #Extracing star+background patches
    img_background = img_fits[background_bound[1][0]:background_bound[1][1],background_bound[0][0]:background_bound[0][1]]
    img_star = img_fits[star_bound[1][0]:star_bound[1][1],star_bound[0][0]:star_bound[0][1]]
    #plt.imshow(img_star)
    #plt.xlim(150,200)
    #plt.ylim(100,150)
    #Calculating the background
    bg_med = np.median(img_background)
    bg_std = np.std(img_background)
    threshhold = bg_med + 3*bg_std
    # img_reduced = img_star - bg_med

    #plt.imshow(img_star,vmin=-1,vmax=5)
    #plt.colorbar()
        
    #Nested for loops for x/y weighted pixels
    for x in np.arange(0,img_star.shape[1]):
        for y in np.arange(0,img_star.shape[0]):
            if img_star[x,y] > threshhold:
                x_weight = (img_star[x,y]-bg_med)*x
                y_weight = (img_star[x,y]-bg_med)*y
                no_weight= img_star[x,y]-bg_med
                img_x_weight.append(x_weight)
                img_y_weight.append(y_weight)
                img_no_weight.append(no_weight)
                
    #Calculate the image's centroid
    print(sum(img_x_weight))
    print(sum(img_no_weight))
    img_x_centroid = sum(img_x_weight)/sum(img_no_weight)
    img_y_centroid = sum(img_y_weight)/sum(img_no_weight)
    img_centroid = [img_y_centroid,img_x_centroid]
    print(img_centroid)
    starbox = img_star[int(img_x_centroid)-size:int(img_x_centroid)+size+1, int(img_y_centroid)-size:int(img_y_centroid)+size+1]
    plt.imshow(starbox)
    plt.scatter(size+1,size+1, color='red')
    plt.colorbar



    y, x, = np.mgrid[:size*2+1, :size*2+1]
    print(np.shape(starbox))
    f_init = models.Gaussian2D(amplitude = starbox[size+1,size+1]-np.median(img_background),x_mean=int(size+1), y_mean = int(size+1))
    fit_f = fitting.LevMarLSQFitter()

    f = fit_f(f_init, x, y, starbox-np.median(img_background))
    std_devs = np.array([f.x_stddev[0],f.y_stddev[0]])
    fwhm = 2.335*std_devs
    fwhm = np.sqrt(fwhm[0]**2+fwhm[1]**2)


    psf_fin = f(x,y)/np.sum(f(x,y))


    return img_centroid, psf_fin, starbox, fwhm

def background_subtract(img):
    """
    Subtracts out the background from an image
    Inputs:
    img (str): filepath to image to be subtracted

    Outputs:
    writes subtracted image out!
    """
    data = fits.getdata(img)
    hdr = fits.getheader(img)

    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (15, 15), filter_size=(3, 3),
                   sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    
    new_data = data-bkg.background

    fits.writeto(os.path.dirname(img)+'/s_' + os.path.basename(img), new_data, hdr, overwrite=True)
    return new_data



#Error estimation function
def bg_error_estimate(fitsfile):
    """
    Summary: Calculates the background error from an image.
    Input: fitsfile for measuring background. String.
    Output: Error of the background in counts.
    """
    fitsdata = fits.getdata(fitsfile)
    hdr = fits.getheader(fitsfile)
    
    # Removes data sigma number of standard deviations away from the mean
    filtered_data = sigma_clip(fitsdata, sigma=6.,copy=False)
    
    #Calculates error for given pixels
    bkg_values_nan = filtered_data.filled(fill_value=np.nan)
    bkg_error = np.sqrt(bkg_values_nan)
    bkg_error[np.isnan(bkg_error)] = np.nanmedian(bkg_error)
    
    print("Writing the background-only error image: ", fitsfile.split('.')[0]+"_bgerror.fit")
    fits.writeto(fitsfile.split('.')[0]+"_bgerror.fit", bkg_error, hdr, overwrite=True)
    
    effective_gain = 1.4 # electrons per ADU
    
    error_image = calc_total_error(fitsdata, bkg_error, effective_gain)  
    
    print("Writing the total error image: ", fitsfile.split('.')[0]+"_error.fit")
    fits.writeto(fitsfile.split('.')[0]+"_error.fit", error_image, hdr, overwrite=True)
    
    return error_image

# Star extraction function 
def starExtractor(fitsfile, nsigma_value, fwhm_value):
    """
    Summary:
    Uses a given fwhm and sigma detection threshold to find potential star candidates using centroiding and returns the arrays of those centroid coordinates
    Inputs:
    fitsfile (str): the filepath to the file to extract star positions from
    nsigma_value (float): the number of standard deviations of noise above the background which a star needs to be to be counted/detected
    fwhm_value (float): the approximate FWHM value for potential star candidates 
    Outputs:
    xpos (array): array of the x positions of the star centroids
    ypos (array): array of the y positions of the star centroids
    """
    
    # First, check if the region file exists yet, so it doesn't get overwritten
    regionfile = fitsfile.split(".")[0] + ".reg"
     
    if os.path.exists(regionfile) == True:
        print(regionfile, "already exists in this directory. Rename or remove the .reg file and run again.")
        return
    
    # *** ea ***
    image = fits.getdata(fitsfile)
    
    # *** Measure the median absolute standard deviation of the image: ***
    bkg_sigma = np.nanmedian(bg_error_estimate(fitsfile))
    
    # *** Define the parameters for DAOStarFinder ***
    daofind = DAOStarFinder(fwhm=fwhm_value, threshold = bkg_sigma*nsigma_value)
    
    # Apply DAOStarFinder to the image
    sources = daofind(image)
    nstars = len(sources)
    print("Number of stars found in ",fitsfile,":", nstars)
    
    # Define arrays of x-position and y-position
    xpos = np.array(sources['xcentroid'])
    ypos = np.array(sources['ycentroid'])
    
    # Write the positions to a .reg file based on the input file name
    if os.path.exists(regionfile) == False:
        f = open(regionfile, 'w') 
        for i in range(0,len(xpos)):
            f.write('circle '+str(xpos[i])+' '+str(ypos[i])+' '+str(fwhm_value)+'/n')
        f.close()
        print("Wrote ", regionfile)
        return xpos, ypos # Return the x and y positions of each star as variables

#Measures the photometry at the given points with the specified aperture size

def measurePhotometry(fitsfile, star_xpos, star_ypos, aperture_radius, sky_inner, sky_outer, error_array):
    """
    Summary: Uses aperture photometry to determine star counts for the stars at the specified coordinates
    Inputs:
    fitsfile (str): the path to the target image
    star_xpos (array): the x coordinates of the target stars
    star_ypos (array): the y coordinates of the target stars
    aperture_radius (float): the desired radius for the aperture
    sky_inner (float): the inner radius of the sky annulus
    sky_outer (float: the outer radius of the sky annulus
    error_array (array): the array of the errors (effectively noise) at each point
    Outputs:
    phot_table (array): the array of the photon counts for the given stars
    """
    # *** Read in the data from the fits file:
    image = fits.getdata(fitsfile)

    #Creates an array of the star positions
    star_pos = np.vstack([star_xpos, star_ypos]).T
    
    #Creates apertures around the stars
    starapertures = CircularAperture(star_pos,r = aperture_radius)
    skyannuli = CircularAnnulus(star_pos, r_in = sky_inner, r_out = sky_outer)
    phot_apers = [starapertures, skyannuli]
    
    #Creates the photometry table
    phot_table = aperture_photometry(image, phot_apers, error=error_array)
        
    # Calculate mean background in annulus and subtract from aperture flux
    bkg_mean = phot_table['aperture_sum_1'] / skyannuli.area
    bkg_starap_sum = bkg_mean * starapertures.area
    final_sum = phot_table['aperture_sum_0']-bkg_starap_sum
    phot_table['bg_subtracted_star_counts'] = final_sum
    
    # Divides the total error in the annulus by the its area
    bkg_mean_err = phot_table['aperture_sum_err_1'] / skyannuli.area
    # Caculates the total error in the star aperture based on the mean error
    bkg_sum_err = bkg_mean_err * starapertures.area
    
    # Calculates the error-corrected values for the star photon counts
    phot_table['bg_sub_star_cts_err'] = np.sqrt((phot_table['aperture_sum_err_0']**2)+(bkg_sum_err**2)) 
    
    return phot_table

#Finally, the zp calc function for the zeropoint function
def zpcalc(magzp, magzp_err, filtername, dataframe):
    '''
    Calculates the calibrated magnitudes for the stars based on the provided zero  
    Inputs:
    magzp (float): the zero point value
    magzp (error): the error in the zero point magnitude, used for error propagation
    filtername (str): the name of the filter for the given zeropoint value
    dataframe (dataframe): the photometry pandas dataframe
    Outputs:
    None (just modifies the pandas dataframe)
    '''
    #Sets up an if statement filtering based on the band of the calculated zeropoi 
    if filtername == "V":
        dataframe["Vmag"] = dataframe['Vinst'] + magzp
        dataframe["Vmag_err"] = np.sqrt((dataframe['Vinst_err'])**2 + magzp_err**2)
    if filtername == "R":
        dataframe["Rmag"] = dataframe['Rinst'] + magzp
        dataframe["Rmag_err"] = np.sqrt((dataframe['Rinst_err'])**2 + magzp_err**2)
    if filtername == "B":
        dataframe["Bmag"] = dataframe['Binst'] + magzp
        dataframe["Bmag_err"] = np.sqrt((dataframe['Binst_err'])**2 + magzp_err**2)
    return
#Setup the final reduction function

def reduction(datafolder,objname):
    '''
    Runs through the reduction pipeline using previously defined functions. The inputs are the master datafolder and 
    the folder with the science images (lights) you wish to reduce.
    Inputs:
    datafolder (str): the filepath to the master datafolder
    objname (str): the folder inside the master datafolder with the desired science images to reduce
    '''
    tqdm.write(textbar)
    tqdm.write(f"Running DRP on {objname}!")
    tqdm.write(textbar)

    # FILTER CHECKER
    image_HaON = glob.glob(datafolder+'/'+objname+'/HaON/202*')
    image_HaOFF = glob.glob(datafolder+'/'+objname+'/HaOFF/202*')
    image_B = glob.glob(datafolder+'/'+objname+'/B/202*')
    image_V = glob.glob(datafolder+'/'+objname+'/V/202*')
    image_R = glob.glob(datafolder+'/'+objname+'/R/202*')
    
    flag_B = False
    flag_V = False
    flag_R = False
    flag_HaON = False
    flag_HaOFF = False
    if len(image_B) > 0:
        flag_B = True
    if len(image_V) > 0:
        flag_V = True    
    if len(image_R) > 0:
        flag_R = True  
    if len(image_HaON) > 0:
        flag_HaON = True
    if len(image_HaOFF) > 0:
        flag_HaOFF = True    
    

    if skipred == True:
        tqdm.write("Skipped Data Reduction!")
    else:
        #First create master calibration files
        #First master bias
        biaslist = glob.glob(datafolder + '/calibration/biasframes/202*')
        bias_check = glob.glob(datafolder + '/calibration/biasframes/master_bias.fit')
        if len(bias_check) == 1:
            master_bias_path = bias_check[0]
            tqdm.write("Master Bias Already Exists!")
        else:
            master_bias_generator(biaslist, datafolder + '/calibration/biasframes')
            master_bias_path = datafolder + '/calibration/biasframes/master_bias.fit'
            tqdm.write("Master Bias Created!")
            pbar.close()

        #Next master dark
        #darklist = glob.glob(datafolder + '/calibration/darks/*')
        #master_dark_generation(darklist, master_bias_path, datafolder + '/calibration/darks')
        #master_dark_path = datafolder + '/calibration/darks/master_dark.fit'
        #print("Master Dark Created")

        #Then the master flats

        #B
        B_flats = glob.glob(datafolder + '/calibration/flats/B/202*')
        B_master_check = glob.glob(datafolder + '/calibration/flats/R/master_flat*')
        if len(B_flats) > 0:
            if len(B_master_check) ==1:
                tqdm.write("Master B Flat Already Exists!")
            else:
                flatfieldingALL(B_flats, master_bias_path, datafolder + '/calibration/flats/B')
                tqdm.write("Blue Flat Created")
                pbar.close()

        #V
        V_flats = glob.glob(datafolder + '/calibration/flats/V/202*')
        V_master_check = glob.glob(datafolder + '/calibration/flats/R/master_flat*')
        if len(V_flats) > 0:
            if len(V_master_check) ==1:
                tqdm.write("Master V Flat Already Exists!")
            else:
                flatfieldingALL(V_flats, master_bias_path, datafolder + '/calibration/flats/V')
                tqdm.write("V Flat Created")
                pbar.close()

        #R
        R_flats = glob.glob(datafolder + '/calibration/flats/R/202*')
        R_master_check = glob.glob(datafolder + '/calibration/flats/R/master_flat*')
        if len(R_flats) > 0:
            if len(R_master_check) ==1:
                tqdm.write("Master R Flat Already Exists!")
            else:
                flatfieldingALL(R_flats, master_bias_path, datafolder + '/calibration/flats/R')
                tqdm.write("R Flat Created")
                pbar.close()
        
        #HaON
        HaON_flats = glob.glob(datafolder + '/calibration/flats/HaON/202*')
        HaON_master_check = glob.glob(datafolder + '/calibration/flats/HaON/master_flat*')
        if len(HaON_flats) > 0:
            if len(HaON_master_check) ==1:
                tqdm.write("Master HaON Flat Already Exists!")
            else:
                flatfieldingALL(HaON_flats, master_bias_path, datafolder + '/calibration/flats/HaON')
                tqdm.write("HaON Flat Created")
                pbar.close()

        #HaOFF
        HaOFF_flats = glob.glob(datafolder + '/calibration/flats/HaOFF/202*')
        HaOFF_master_check = glob.glob(datafolder + '/calibration/flats/HaOFF/master_flat*')
        if len(HaOFF_flats) > 0:
            if len(HaOFF_master_check) ==1:
                tqdm.write("Master HaOFF Flat Already Exists!")
            else:
                flatfieldingALL(HaOFF_flats, master_bias_path, datafolder + '/calibration/flats/HaOFF')
                tqdm.write("HaOFF Flat Created")
                pbar.close()

        #Now, it is time to register the image files
        #Set up the filelists for each filter and run a loop

        #Note the usage of the SAW prefix in the image str; this is to ensure 
        #if we need to run the code again we don't use images that have already been registered
        flat_path = datafolder + '/calibration/flats'


        if flag_B == True:
            pbar = tqdm(total = len(image_B), desc = "B Image Reduction", colour='blue')#, leave = False)
            for im in image_B:
                reducing(im,master_bias_path,flat_path,im)
                pbar.update(1)
            tqdm.write("B Images Reduced")
            pbar.close()

        if flag_V == True:
            pbar = tqdm(total = len(image_V), desc = "V Image Reduction", colour='green')#, leave = False)
            for im in image_V:
                reducing(im,master_bias_path,flat_path,im)
                pbar.update(1)
            tqdm.write("V Images Reduced")
            pbar.close()

        if flag_R == True:
            pbar = tqdm(total = len(image_R), desc = "R Image Reduction", colour='red')#, leave = False)
            for im in image_R:
                reducing(im,master_bias_path,flat_path,im)
                pbar.update(1)
            tqdm.write("R Images Reduced")
            pbar.close()

        if flag_HaON == True:
            pbar = tqdm(total = len(image_HaON), desc = "HaON Image Reduction", colour = 'magenta')#, leave = False)
            for im in image_HaON:
                reducing(im,master_bias_path,flat_path,im)
                pbar.update(1)
            tqdm.write("HaON Images Reduced")
            pbar.close()

        if flag_HaOFF == True:
            pbar = tqdm(total = len(image_HaOFF), desc = "HaOFF Image Reduction", colour = 'cyan')#, leave = False)
            for im in image_HaOFF:
                reducing(im,master_bias_path,flat_path,im)
                pbar.update(1)
            tqdm.write("HaOFF Images Reduced")
            pbar.close()

        #Next, normalize each image by exposure time before shifting and stacking
        image_fdb_HaON = glob.glob(datafolder+'/'+objname+'/HaON/fdb*')
        image_fdb_HaOFF = glob.glob(datafolder+'/'+objname+'/HaOFF/fdb*')
        image_fdb_B = glob.glob(datafolder+'/'+objname+'/B/fdb*')
        image_fdb_V = glob.glob(datafolder+'/'+objname+'/V/fdb*')
        image_fdb_R = glob.glob(datafolder+'/'+objname+'/R/fdb*')

        if flag_B == True:
            pbar = tqdm(total = len(image_fdb_B), desc = "B Image Normalization", colour = "blue")#, leave = False)
            for im in image_fdb_B:
                timenorm(im,im)
                pbar.update(1)
            tqdm.write("B Images Normalized")
            pbar.close()

        if flag_V == True:
            pbar = tqdm(total = len(image_fdb_V), desc = "V Image Normalization", colour = "green")#, leave = False)
            for im in image_fdb_V:
                timenorm(im,im)
                pbar.update(1)
            tqdm.write("V Images Normalized")
            pbar.close()

        if flag_R == True:
            pbar = tqdm(total = len(image_fdb_R), desc = "R Image Normalization", colour = "red")#, leave = False)
            for im in image_fdb_R:
                timenorm(im,im)
                pbar.update(1)
            tqdm.write("R Images Normalized")
            pbar.close()

        if flag_HaON == True:
            pbar = tqdm(total = len(image_fdb_HaON), desc = "HaON Image Normalization", colour = 'magenta')#, leave = False)
            for im in image_fdb_HaON:
                timenorm(im,im)
                pbar.update(1)
            tqdm.write("HaON Images Normalized")
            pbar.close()

        if flag_HaOFF == True:
            pbar = tqdm(total = len(image_fdb_HaOFF), desc = "HaOFF Image Normalization", colour = 'cyan')#, leave = False)
            for im in image_fdb_HaOFF:
                timenorm(im,im)
                pbar.update(1)
            tqdm.write("HaOFF Images Normalized")
            pbar.close()

    image_n_fdb_HaON = glob.glob(datafolder+'/'+objname+'/HaON/n_fdb*')
    image_n_fdb_HaOFF = glob.glob(datafolder+'/'+objname+'/HaOFF/n_fdb*')
    image_n_fdb_B = glob.glob(datafolder+'/'+objname+'/B/n_fdb*')
    image_n_fdb_V = glob.glob(datafolder+'/'+objname+'/V/n_fdb*')
    image_n_fdb_R = glob.glob(datafolder+'/'+objname+'/R/n_fdb*')

    if flag_B == True:
        pbar = tqdm(total = len(image_n_fdb_B), desc = "B Image Background Subtraction", colour = "blue")#, leave = False)
        for im in image_n_fdb_B:
            background_subtract(im)
            pbar.update(1)
        tqdm.write("B Backgrounds Subtracted")
        pbar.close()

    if flag_V == True:
        pbar = tqdm(total = len(image_n_fdb_V), desc = "V Image Background Subtraction", colour = "green")#, leave = False)
        for im in image_n_fdb_V:
            background_subtract(im)
            pbar.update(1)
        tqdm.write("V Backgrounds Subtracted")
        pbar.close()

    if flag_R == True:
        pbar = tqdm(total = len(image_n_fdb_R), desc = "R Image Background Subtraction", colour = "red")#, leave = False)
        for im in image_n_fdb_R:
            background_subtract(im)
            pbar.update(1)
        tqdm.write("R Backgrounds Subtracted")
        pbar.close()

    if flag_HaON == True:
        pbar = tqdm(total = len(image_n_fdb_HaON), desc = "HaON Image Background Subtraction", colour = 'magenta')#, leave = False)
        for im in image_n_fdb_HaON:
            background_subtract(im)
            pbar.update(1)
        tqdm.write("HaON Backgrounds Subtracted")
        pbar.close()

    if flag_HaOFF == True:
        pbar = tqdm(total = len(image_n_fdb_HaOFF), desc = "HaOFF Image Background Subtraction", colour = 'cyan')#, leave = False)
        for im in image_n_fdb_HaOFF:
            background_subtract(im)
            pbar.update(1)
        tqdm.write("HaOFF Backgrounds Subtracted")
        pbar.close()


    image_s_n_fdb_HaON = glob.glob(datafolder+'/'+objname+'/HaON/s_n_fdb*')
    image_s_n_fdb_HaOFF = glob.glob(datafolder+'/'+objname+'/HaOFF/s_n_fdb*')
    image_s_n_fdb_B = glob.glob(datafolder+'/'+objname+'/B/s_n_fdb*')
    image_s_n_fdb_V = glob.glob(datafolder+'/'+objname+'/V/s_n_fdb*')
    image_s_n_fdb_R = glob.glob(datafolder+'/'+objname+'/R/s_n_fdb*')



    #We've already found the bounds for each filter, but depending on the what the objname is,
    #we need to specify which bounds we are using, along with the reference image index
    if objname == 'standard':
        HaON_star_bound = [[1060,1160],[1030,1130]]
        HaON_background_bound = [[1060,1160],[830,930]]
        HaON_index = 1
        HaOFF_star_bound = [[1060,1160],[1030,1130]]
        HaOFF_background_bound = [[1060,1160],[830,930]]
        HaOFF_index = 1
        #v_star_bound = [[2000,2080],[2190,2270]]
        #v_background_bound = [[2120,2160],[2210,2250]]
        #v_index = 4
        #Final alignment bounds
        HaOFF_star_bound = [[1060,1160],[1030,1130]]
        HaOFF_background_bound = [[1060,1160],[830,930]]
 
    if objname == 'target':
        r_star_bound = [[2327,2487],[364,524]]
        r_background_bound = [[2487,2527],[424,464]]
        r_index = 14
        b_star_bound = [[2334,2494],[387,547]]
        b_background_bound = [[2494,2534],[447,487]]
        b_index = 12
        v_star_bound = [[2307,2467],[353,513]]
        v_background_bound = [[2486,2507],[413,453]]
        v_index = 5
        #Final alignment bounds
        star_bound = [[2300,2500],[350,550]]
        bg_bound = [[2520,2540],[450,470]]

    if objname == 'NGC 2785':
        HaON_star_bound = [[900,1000],[560,660]]
        HaON_background_bound = [[900,1000],[400,500]]
        HaON_index = 1
        HaOFF_star_bound = [[900,1000],[560,660]]
        HaOFF_background_bound = [[900,1000],[400,500]]
        HaOFF_index = 1
        #Final alignment bounds
        star_bound = [[900,1000],[560,660]]
        bg_bound = [[900,1000],[400,500]]

    if objname == 'NGC 5297':
        HaON_star_bound = [[1325,1385],[1480,1540]]
        HaON_background_bound = [[1410,1470],[1480,1540]]
        HaON_index = 1
        HaOFF_star_bound = [[1325,1385],[1480,1540]]
        HaOFF_background_bound = [[1410,1470],[1480,1540]]
        HaOFF_index = 1
        #Final alignment bounds
        star_bound = [[1325,1385],[1480,1540]]
        bg_bound = [[1410,1470],[1480,1540]]

    #Initialize shift lists
    xy_HaOFF = []
    xy_HaON = []
    xy_B = []
    xy_V = []
    xy_R = []

    #Actually execute the centroiding for loops
    # pbar = tqdm(total = len(image_R), desc = "Red Image Alignment:")
    # for im in image_fdb_R:
    #    xy_shift = centroid_one_star(image_fdb_R[r_index], im, np.array(r_star_bound), np.array(r_background_bound))
    #    xy_R.append(xy_shift)
    #    pbar.update(1)
    # print("Red Shifts Computed")

    # pbar = tqdm(total = len(image_B), desc = "Blue Image Alignment:")
    # for im in image_fdb_B:
    #    xy_shift = centroid_one_star(image_fdb_B[b_index], im, np.array(b_star_bound), np.array(b_background_bound))
    #    xy_B.append(xy_shift)
    #    pbar.update(1)
    # print("Blue Shifts Computed")

    # pbar = tqdm(total = len(image_V), desc = "Visual Image Alignment:")
    # for im in image_fdb_V:
    #    xy_shift = centroid_one_star(image_fdb_V[v_index], im, np.array(v_star_bound), np.array(v_background_bound))
    #    xy_V.append(xy_shift)
    #    pbar.update(1)
    #print("Visual Shifts Computed")
    if omitstack == True:
        tqdm.write("Skipping Shifting/Stacking and Using Existing Master Images")
    else:
        if flag_B == True:
            pbar = tqdm(total = len(image_s_n_fdb_B), desc = "B Image Alignment", colour = 'blue')#, leave = False)
            for im in image_s_n_fdb_B:
                xy_shift = cross_image(image_s_n_fdb_B[0], im)
                xy_B.append(xy_shift)
                pbar.update(1)
            tqdm.write("B Shifts Computed")
            pbar.close()

        if flag_V == True:
            pbar = tqdm(total = len(image_s_n_fdb_V), desc = "V Image Alignment", colour = 'green')#, leave = False)
            for im in image_s_n_fdb_B:
                xy_shift = cross_image(image_s_n_fdb_V[0], im)
                xy_V.append(xy_shift)
                pbar.update(1)
            tqdm.write("V Shifts Computed")
            pbar.close()

        if flag_R == True:
            pbar = tqdm(total = len(image_s_n_fdb_R), desc = "R Image Alignment", colour = 'red')#, leave = False)
            for im in image_s_n_fdb_R:
                xy_shift = cross_image(image_s_n_fdb_R[0], im)
                xy_R.append(xy_shift)
                pbar.update(1)
            tqdm.write("R Shifts Computed")
            pbar.close()
        
        if flag_HaON == True:
            pbar = tqdm(total = len(image_s_n_fdb_HaON), desc = "HaON Image Alignment", colour = 'magenta')#, leave = False)
            tqdm.write("Aligning to: " + image_s_n_fdb_HaON[0])
            for im in image_s_n_fdb_HaON:
                tqdm.write(im)
                tqdm.write(image_s_n_fdb_HaON[0])
                xy_shift = cross_image(image_s_n_fdb_HaON[0], im)
                tqdm.write(str(xy_shift))
                xy_HaON.append(xy_shift)
                pbar.update(1)
            tqdm.write("HaON Shifts Computed")
            pbar.close()

        if flag_HaOFF == True:
            pbar = tqdm(total = len(image_s_n_fdb_HaOFF), desc = "HaOFF Image Alignment", colour = 'cyan')#, leave = False)
            for im in image_s_n_fdb_HaOFF:
                xy_shift = cross_image(image_s_n_fdb_HaOFF[0], im)
                xy_HaOFF.append(xy_shift)
                pbar.update(1)
            tqdm.write("HaOFF Shifts Computed")
            pbar.close()



        #pbar = tqdm(total = len(image_HaON), desc = "HaON Image Alignment:")
        #for im in image_fdb_HaON:
        #    xy_shift = centroid_one_star(image_n_fdb_HaON[HaON_index], im, np.array(HaON_star_bound), np.array(HaON_background_bound))
        #    xy_HaON.append(xy_shift)
        #    pbar.update(1)
        #print("HaON Shifts Computed")
        #print(xy_HaON)

        #pbar = tqdm(total = len(image_HaOFF), desc = "HaOFF Image Alignment:")
        #for im in image_fdb_HaOFF:
        #    xy_shift = centroid_one_star(image_n_fdb_HaOFF[HaOFF_index], im, np.array(HaOFF_star_bound), np.array(HaOFF_background_bound))
        #    xy_HaOFF.append(xy_shift)
        #    pbar.update(1)
        #print("HaOFF Shifts Computed")

        #Next, align + stack the images in those arrays
        #Red first
        #simple_image_combination(image_fdb_red, np.array(xy_red), np.max(xy_red), datafolder+'/'+objname+'/red')
        #print("Red Images Stacked")
        #Then blue
        #simple_image_combination(image_fdb_blue, np.array(xy_blue), np.max(xy_blue), datafolder+'/'+objname+'/blue')
        #print("Blue Images Stacked")
        #Then visual
        if flag_B == True:
            reg_and_stack(image_s_n_fdb_B, np.array(xy_B), math.ceil(np.max(xy_B)), datafolder+'/'+objname+'/B')
            tqdm.write("B Images Stacked")
            pbar.close()

        if flag_V == True:
            reg_and_stack(image_s_n_fdb_V, np.array(xy_V), math.ceil(np.max(xy_V)), datafolder+'/'+objname+'/V')
            tqdm.write("V Images Stacked")
            pbar.close()

        if flag_R == True:
            reg_and_stack(image_s_n_fdb_R, np.array(xy_R), math.ceil(np.max(xy_R)), datafolder+'/'+objname+'/R')
            tqdm.write("R Images Stacked")
            pbar.close()

        if flag_HaON == True:
            reg_and_stack(image_s_n_fdb_HaON, np.array(xy_HaON), math.ceil(np.max(xy_HaON)), datafolder+'/'+objname+'/HaON')
            tqdm.write("HaON Images Stacked")
            pbar.close()

        if flag_HaOFF == True:
            reg_and_stack(image_s_n_fdb_HaOFF, np.array(xy_HaOFF), math.ceil(np.max(xy_HaOFF)), datafolder+'/'+objname+'/HaOFF')
            tqdm.write("HaOFF Images Stacked")
            pbar.close()
    
    #Finally, align all the images 
    #We will align relative to the red band
    #First, the alignment coordinates themselves


    ######################################################################################################
    if flag_B == True:
        B_path = glob.glob(datafolder+'/'+objname+'/B/registered*')[0]
    if flag_V == True:
        V_path = glob.glob(datafolder+'/'+objname+'/V/registered*')[0]
    if flag_R == True:
        R_path = glob.glob(datafolder+'/'+objname+'/R/registered*')[0]
    if flag_HaON == True:
        HaON_path = glob.glob(datafolder+'/'+objname+'/HaON/registered*')[0]
    if flag_HaOFF == True:
        HaOFF_path = glob.glob(datafolder+'/'+objname+'/HaOFF/registered*')[0]
    ######################################################################################################
    if flag_HaOFF == True:
        ref_path = HaOFF_path
    elif flag_HaON == True:
        ref_path = HaON_path
    elif flag_B == True:
        ref_path = B_path
    elif flag_V == True:
        ref_path = V_path
    elif flag_R == True:
        ref_path == R_path

    #red_shift = centroid_one_star(red_path, red_path, np.array(general_star_bound), np.array(general_background_bound))
    #blue_shift = centroid_one_star(red_path, blue_path, np.array(general_star_bound), np.array(general_background_bound))
    #visual_shift = centroid_one_star(red_path, visual_path, np.array(general_star_bound), np.array(general_background_bound))

### INSERT PSF CONVOLUTION HERE

    if objname == "standard":
        star_bound = [[800,1100],[890,1190]]
        bg_bound = [[1100,1400],[800,1100]]
    if objname == "NGC 5297":
        star_bound = [[1600,1900],[550,850]] #1740 696, 1723, 698, 1720, 704
        bg_bound = [[1700,1900],[750,950]]
    if objname == "NGC 2785":
        star_bound = [[1530,1770],[1200,1440]]
        bg_bound = [[1240,1480],[1360,1600]]

    if flag_B == True:
        B_shift = centroid_one_star(ref_path, B_path, star_bound, bg_bound)
    if flag_V == True:
        V_shift = centroid_one_star(ref_path, V_path, star_bound, bg_bound)
    if flag_R == True:
        R_shift = centroid_one_star(ref_path, R_path, star_bound, bg_bound)
    if flag_HaON == True:
        HaON_shift = centroid_one_star(ref_path, HaON_path, star_bound, bg_bound)
        #tqdm.write("HaON SHIFT")
        #tqdm.write(str(HaON_shift))
    if flag_HaOFF == True:
        HaOFF_shift = centroid_one_star(ref_path, HaOFF_path, star_bound, bg_bound)
        #tqdm.write("HaOFF SHIFT")
        #tqdm.write(str(HaOFF_shift))
    #Then shift the actual images
    median_images = []
    registered_shifts0 = []
    registered_shifts1 = []
    if flag_B == True:
        median_images.append(B_path)
        registered_shifts0.append(B_shift[0])
        registered_shifts1.append(B_shift[1])
    if flag_V == True:
        median_images.append(V_path)
        registered_shifts0.append(V_shift[0])
        registered_shifts1.append(V_shift[1])
    if flag_R == True:
        median_images.append(R_path)
        registered_shifts0.append(R_shift[0])
        registered_shifts1.append(R_shift[1])
    if flag_HaON == True:
        median_images.append(HaON_path)
        registered_shifts0.append(HaON_shift[0])
        registered_shifts1.append(HaON_shift[1])
    if flag_HaOFF == True:
        median_images.append(HaOFF_path)
        registered_shifts0.append(HaOFF_shift[0])
        registered_shifts1.append(HaOFF_shift[1])


    #print(registered_shifts0)
    #print(registered_shifts1)


    #For loop for shifting each image (NOT stacking them)
    print("Shifting Master Images")
    for m in np.arange(0,len(median_images)):
        file = median_images[m]
        image = fits.getdata(file)
        image_header = fits.getheader(file)
        original_shape = image.shape
        #Removing artifact values
        image[np.isinf(image)] = 0.0
        image[np.isnan(image)] = 0.0
        #print(image.shape[0])
        #Padding + shifting
        #Need to pad up to a specific value (so that the final arrays all have the same shape)
        max0 = np.max(registered_shifts0)
        max1 = np.max(registered_shifts1)
        #print(max0,max1)
        maxall = max(max0,max1)
        
        pad_specific = math.ceil(maxall)

        #pad_specific = int((2200 - image.shape[0])/2)
        #print(registered_shifts0)
        #print(registered_shifts1)
        padded_image = np.pad(image, pad_specific, 'constant', constant_values = 0.001)
        new_image = shift(padded_image, (-1*float(registered_shifts1[m]),-1*float(registered_shifts0[m])), cval = 0.001)

        pad_y, pad_x = pad_specific, pad_specific
        cropped_image = new_image[pad_y: pad_y + original_shape[0], pad_x: pad_x + original_shape[1]]

        #Writing the new image with the prefix align_ (for aligned to the standard image)
        fits.writeto(os.path.dirname(file)+'/align_' + os.path.basename(file), 
                    cropped_image, image_header, overwrite=True)
        tqdm.write("Aligned: " + str(file))

######### PSF STUFF!
    tqdm.write("Calculating PSFs")
    
        
    if flag_B == True:
        B_path = glob.glob(datafolder+'/'+objname+'/B/align*')[0]
    if flag_V == True:
        V_path = glob.glob(datafolder+'/'+objname+'/V/align*')[0]
    if flag_R == True:
        R_path = glob.glob(datafolder+'/'+objname+'/R/align*')[0]
    if flag_HaON == True:
        HaON_path = glob.glob(datafolder+'/'+objname+'/HaON/align*')[0]
    if flag_HaOFF == True:
        HaOFF_path = glob.glob(datafolder+'/'+objname+'/HaOFF/align*')[0]

    keydata = []
    ###Extract PSFs and FWHMs
    if flag_B == True:
        centB, psfB, starboxB, fwhmB = get_psf(B_path, star_bound, bg_bound,50)
        keydata.append(['B',fwhmB,psfB,B_path])
    if flag_V == True:
        centV, psfV, starboxV, fwhmV = get_psf(V_path, star_bound, bg_bound,50)
        keydata.append(['V',fwhmV,psfV,V_path])
    if flag_R == True:
        centR, psfR, starboxR, fwhmR = get_psf(R_path, star_bound, bg_bound,50)
        keydata.append(['R',fwhmR,psfR,R_path])
    if flag_HaON == True:
        centHaON, psfHaON, starboxHaON, fwhmHaON = get_psf(HaON_path, star_bound, bg_bound,50)
        keydata.append(['HaON',fwhmHaON,psfHaON,HaON_path])
    if flag_HaOFF == True:
        centHaOFF, psfHaOFF, starboxHaOFF, fwhmHaOFF = get_psf(HaOFF_path, star_bound, bg_bound,50)
        keydata.append(['HaOFF',fwhmHaOFF,psfHaOFF,HaOFF_path])

    df = pd.DataFrame(keydata, columns=['filt', 'fwhm', 'psf','filepath'])
    print(df)
    max_row = df.loc[df['fwhm'].idxmax()]
    tqdm.write(f"Convolving Images to match PSF of {max_row.filt}!")
    pbar = tqdm(total = len(df), desc = "Convolving Images", leave = False)
    for idx, row in df.iterrows():
        # if row.filt == max_row.filt:
        #     tqdm.write("Filters Match! Skipping Convolution")
        #     newdata = fits.getdata(row.filepath)
        # else:
        kernellap = make_wiener_kernel(row.psf,max_row.psf, penalty="laplacian", regularization=1.082e-5)
        newdata = fftconvolve(fits.getdata(row.filepath), kernellap)
        header = fits.getheader(row.filepath)
        fits.writeto(datafolder+'/'+objname+'/' +row.filt+  f'/convolved_{row.filt}.fits', newdata, header, overwrite=True)
        pbar.update(1)
    pbar.close()
    tqdm.write("CONVOLVED ALL IMAGES!")

    #REALIGN IMAGES
    
    if flag_B == True:
        B_path = glob.glob(datafolder+'/'+objname+'/B/convolved*')[0]
    if flag_V == True:
        V_path = glob.glob(datafolder+'/'+objname+'/V/convolved*')[0]
    if flag_R == True:
        R_path = glob.glob(datafolder+'/'+objname+'/R/convolved*')[0]
    if flag_HaON == True:
        HaON_path = glob.glob(datafolder+'/'+objname+'/HaON/convolved*')[0]
    if flag_HaOFF == True:
        HaOFF_path = glob.glob(datafolder+'/'+objname+'/HaOFF/convolved*')[0]

    if flag_HaOFF == True:
        ref_path = HaOFF_path
    elif flag_HaON == True:
        ref_path = HaON_path
    elif flag_B == True:
        ref_path = B_path
    elif flag_V == True:
        ref_path = V_path
    elif flag_R == True:
        ref_path == R_path

    tqdm.write(f'Ref Image: {ref_path}')
    if flag_B == True:
        B_shift = centroid_one_star(ref_path, B_path, star_bound, bg_bound)
        tqdm.write(f'B SHIFT: {B_shift}')
    if flag_V == True:
        V_shift = centroid_one_star(ref_path, V_path, star_bound, bg_bound)
        tqdm.write(f'V SHIFT: {V_shift}')
    if flag_R == True:
        R_shift = centroid_one_star(ref_path, R_path, star_bound, bg_bound)
        tqdm.write(f'R SHIFT: {R_shift}')
    if flag_HaON == True:
        HaON_shift = centroid_one_star(ref_path, HaON_path, star_bound, bg_bound)
        tqdm.write(f'HaON SHIFT: {HaON_shift}')
    if flag_HaOFF == True:
        HaOFF_shift = centroid_one_star(ref_path, HaOFF_path, star_bound, bg_bound)
        tqdm.write(f'HaOFF SHIFT: {HaOFF_shift}')

    #Then shift the actual images
    median_images = []
    registered_shifts0 = []
    registered_shifts1 = []

    if flag_B == True:
        median_images.append(B_path)
        registered_shifts0.append(B_shift[0])
        registered_shifts1.append(B_shift[1])
    if flag_V == True:
        median_images.append(V_path)
        registered_shifts0.append(V_shift[0])
        registered_shifts1.append(V_shift[1])
    if flag_R == True:
        median_images.append(R_path)
        registered_shifts0.append(R_shift[0])
        registered_shifts1.append(R_shift[1])
    if flag_HaON == True:
        median_images.append(HaON_path)
        registered_shifts0.append(HaON_shift[0])
        registered_shifts1.append(HaON_shift[1])
    if flag_HaOFF == True:
        median_images.append(HaOFF_path)
        registered_shifts0.append(HaOFF_shift[0])
        registered_shifts1.append(HaOFF_shift[1])

    print(median_images)
    tqdm.write("Shifting Convolved Images")
    for m in np.arange(0,len(median_images)):
        file = median_images[m]
        image = fits.getdata(file)
        image_header = fits.getheader(file)
        original_shape = image.shape
        #Removing artifact values
        image[np.isinf(image)] = 0.0
        image[np.isnan(image)] = 0.0
        #print(image.shape[0])
        #Padding + shifting
        #Need to pad up to a specific value (so that the final arrays all have the same shape)
        max0 = np.max(registered_shifts0)
        max1 = np.max(registered_shifts1)
        #print(max0,max1)
        maxall = max(max0,max1)
        
        pad_specific = math.ceil(maxall)

        #pad_specific = int((2200 - image.shape[0])/2)
        #print(registered_shifts0)
        #print(registered_shifts1)
        padded_image = np.pad(image, pad_specific, 'constant', constant_values = 0.001)
        new_image = shift(padded_image, (float(registered_shifts1[m]),float(registered_shifts0[m])), cval = 0.001)

        pad_y, pad_x = pad_specific, pad_specific
        cropped_image = new_image[pad_y: pad_y + original_shape[0], pad_x: pad_x + original_shape[1]]

        #Writing the new image with the prefix align_ (for aligned to the standard image)
        fits.writeto(os.path.dirname(file)+'/final_' + os.path.basename(file), 
                    cropped_image, image_header, overwrite=True)
        tqdm.write("Aligned: " + str(file))
    

    

    


    












######### PSF STUFF!


### SUBTRACT HA ON FROM OFF
    HaON_aligned = glob.glob(os.path.dirname(HaON_path)+'/final*')[0]
    header = fits.getheader(HaON_aligned)
    HaOFF_aligned = glob.glob(os.path.dirname(HaOFF_path)+'/final*')[0]
    HaON_data = fits.getdata(HaON_aligned)
    HaOFF_data = fits.getdata(HaOFF_aligned)

    Ha_DIFF = HaOFF_data - HaON_data
    
    fits.writeto(datafolder +'/'+ objname + '/Ha_difference.fits', Ha_DIFF, header, overwrite = True)
    print("Subtracted HaON from HaOFF!")


#Actually run the reduction function
if __name__ == "__main__":


######################################################################################################
    
    if opt.datafolder == None:
        raise(NameError("No datafolder specified!"))
        #datafolder = '/Users/samkleiman/Desktop/PERKINS_DATA_TESTING/20250320'
    else:
        datafolder = opt.datafolder
    if opt.objname == None:
        warnings.warn("No object name specified! Input name now.")
        objname = input("Object name: ")
    else:
        objname = opt.objname
    if opt.skipreduction == "True":
        skipred = True
    else:
        skipred = False

    if opt.omitstacking == "True":
        omitstack = True
    else:
        omitstack = False

    
######################################################################################################
    
    reduction(datafolder, objname)

#Time to run the photometry code


#########################################################################################################################################################
#Specify file paths to all the relevant images
#Standard star images
#std_V = "/Users/samkleiman/Desktop/Grant_Treadway_Kleiman_sorted/datafolder/standard/visual/align_registered_Visual.fit"
#std_R = "/Users/samkleiman/Desktop/Grant_Treadway_Kleiman_sorted/datafolder/standard/red/align_registered_Red.fit"
#std_B = "/Users/samkleiman/Desktop/Grant_Treadway_Kleiman_sorted/datafolder/standard/blue/align_registered_Blue.fit"
#Cluster
#cls_V = "/Users/samkleiman/Desktop/Grant_Treadway_Kleiman_sorted/datafolder/target/visual/align_registered_Visual.fit"
#cls_R = "/Users/samkleiman/Desktop/Grant_Treadway_Kleiman_sorted/datafolder/target/red/align_registered_Red.fit"
#cls_B = "/Users/samkleiman/Desktop/Grant_Treadway_Kleiman_sorted/datafolder/tartarget/blue/align_registered_Blue.fit"
##########################################################################################################################################################

#Calculating the positions of stars (so as to propogate)
#Based off the visual images
#std_xpos, std_ypos = starExtractor(std_V, nsigma_value=25, fwhm_value=10)

#cls_xpos, cls_ypos = starExtractor(cls_V, nsigma_value=25, fwhm_value=10)

#Extracting the photometry tables for the standard star V, R and B bands
#std_V_bgerror = bg_error_estimate(std_V)
#std_V_phottable = measurePhotometry(std_V, star_xpos=std_xpos, star_ypos=std_ypos, 
#                                    aperture_radius=35, sky_inner=40, sky_outer=45, error_array=std_V_bgerror)
#std_R_bgerror = bg_error_estimate(std_R)
#std_R_phottable = measurePhotometry(std_R, star_xpos=std_xpos, star_ypos=std_ypos, 
#                                    aperture_radius=35, sky_inner=40, sky_outer=45, error_array=std_R_bgerror)
#std_B_bgerror = bg_error_estimate(std_B)
#std_B_phottable = measurePhotometry(std_B, star_xpos=std_xpos, star_ypos=std_ypos, 
#                                    aperture_radius=35, sky_inner=40, sky_outer=45, error_array=std_B_bgerror)
#Extracting the photometry tables for the cluster V, R and B bands
#cls_V_bgerror = bg_error_estimate(cls_V)
#cls_V_phottable = measurePhotometry(cls_V, star_xpos=cls_xpos, star_ypos=cls_ypos, 
#                                   aperture_radius=15, sky_inner=25, sky_outer=30, error_array=cls_V_bgerror)
#cls_R_bgerror = bg_error_estimate(cls_R)
#cls_R_phottable = measurePhotometry(cls_R, star_xpos=cls_xpos, star_ypos=cls_ypos, 
#                                    aperture_radius=15, sky_inner=25, sky_outer=30, error_array=cls_R_bgerror)
#cls_B_bgerror = bg_error_estimate(cls_B)
#cls_B_phottable = measurePhotometry(cls_B, star_xpos=cls_xpos, star_ypos=cls_ypos, 
#                                    aperture_radius=15, sky_inner=25, sky_outer=30, error_array=cls_B_bgerror)

#Now, we create the flux tables
#Standard first
#We are only including the normalized fluxes since that is what is
#columns = ['id','xcenter', 'ycenter','Vflux_1sec','Vfluxerr_1sec','Rflux_1sec','Rfluxerr_1sec','Bflux_1sec','Bfluxerr_1sec']
#std_fluxtable = pd.DataFrame(
# {'id' : std_V_phottable['id'],
# 'xcenter' : std_V_phottable['xcenter'],
# 'ycenter' : std_V_phottable['ycenter'],
# 'Vflux_1sec' : std_V_phottable['bg_subtracted_star_counts']/fits.getheader(std_V)['EXPTIME'],
# 'Vfluxerr_1sec': std_V_phottable['bg_sub_star_cts_err']/fits.getheader(std_V)['EXPTIME'],
# 'Rflux_1sec' : std_R_phottable['bg_subtracted_star_counts']/fits.getheader(std_R)['EXPTIME'],
# 'Rfluxerr_1sec': std_R_phottable['bg_sub_star_cts_err']/fits.getheader(std_R)['EXPTIME'],
# 'Bflux_1sec' : std_B_phottable['bg_subtracted_star_counts']/fits.getheader(std_B)['EXPTIME'],
# 'Bfluxerr_1sec': std_B_phottable['bg_sub_star_cts_err']/fits.getheader(std_B)['EXPTIME']}, columns=columns)

#Then cluster
#cls_fluxtable =  pd.DataFrame(
# {'id' : cls_V_phottable['id'],
# 'xcenter' : cls_V_phottable['xcenter'],
# 'ycenter' : cls_V_phottable['ycenter'],
# 'Vflux_1sec' : cls_V_phottable['bg_subtracted_star_counts']/fits.getheader(cls_V)['EXPTIME'],
# 'Vfluxerr_1sec': cls_V_phottable['bg_sub_star_cts_err']/fits.getheader(cls_V)['EXPTIME'],
# 'Rflux_1sec' : cls_R_phottable['bg_subtracted_star_counts']/fits.getheader(cls_R)['EXPTIME'],
# 'Rfluxerr_1sec': cls_R_phottable['bg_sub_star_cts_err']/fits.getheader(cls_R)['EXPTIME'],
# 'Bflux_1sec' : cls_B_phottable['bg_subtracted_star_counts']/fits.getheader(cls_B)['EXPTIME'],
# 'Bfluxerr_1sec': cls_B_phottable['bg_sub_star_cts_err']/fits.getheader(cls_B)['EXPTIME']}, columns=columns) 

#Next, calculate the instrumental magnitudes
#Drop NA rows (that is, those in which it was attempted to take a log of a negative number, meaning a negative flux)
#std_fluxtable['Vinst'] = -2.5 * np.log10(std_fluxtable['Vflux_1sec'])
#std_fluxtable['Rinst'] = -2.5 * np.log10(std_fluxtable['Rflux_1sec'])
#std_fluxtable['Binst'] = -2.5 * np.log10(std_fluxtable['Bflux_1sec'])
#std_fluxtable = std_fluxtable.dropna()

#cls_fluxtable['Vinst'] = -2.5 * np.log10(cls_fluxtable['Vflux_1sec'])
#cls_fluxtable['Rinst'] = -2.5 * np.log10(cls_fluxtable['Rflux_1sec'])
#cls_fluxtable['Binst'] = -2.5 * np.log10(cls_fluxtable['Bflux_1sec'])
#cls_fluxtable = cls_fluxtable.dropna()

#Next calculate the errors in those instrumetal magnitudes
#std_fluxtable['Vinst_err'] = 2.5 * 0.434 * std_fluxtable['Vfluxerr_1sec']/std_fluxtable['Vflux_1sec']
#std_fluxtable['Rinst_err'] = 2.5 * 0.434 * std_fluxtable['Rfluxerr_1sec']/std_fluxtable['Rflux_1sec']
#std_fluxtable['Binst_err'] = 2.5 * 0.434 * std_fluxtable['Bfluxerr_1sec']/std_fluxtable['Bflux_1sec']

#cls_fluxtable['Vinst_err'] = 2.5 * 0.434 * cls_fluxtable['Vfluxerr_1sec']/cls_fluxtable['Vflux_1sec']
#cls_fluxtable['Rinst_err'] = 2.5 * 0.434 * cls_fluxtable['Rfluxerr_1sec']/cls_fluxtable['Rflux_1sec']
#cls_fluxtable['Binst_err'] = 2.5 * 0.434 * cls_fluxtable['Bfluxerr_1sec']/cls_fluxtable['Bflux_1sec']

#The standard star's coordinates are 2064,2260
#sub_V = std_fluxtable[std_fluxtable["xcenter"] > 2030]
#sub_V[sub_V['xcenter'] < 2090]

#Retrieve the standard star magnitudes and caculate the correction factor zp for each band
#We got our V and B values from SIMBAD, and the R value we calculated ourselves 
#(less reliable and thus made a B-V CMD instead of a V-R)

#magzp_V = -1*float(std_fluxtable.loc[[99]]['Vinst']) + 5.984
#magzp_V_error = np.sqrt((float(std_fluxtable.loc[[99]]['Vinst_err']))**2 + (0.010))

#magzp_R = -1*float(std_fluxtable.loc[[99]]['Rinst']) + 5.089
#magzp_R_error = np.sqrt((float(std_fluxtable.loc[[99]]['Rinst_err']))**2 + (0.032))

#magzp_B = -1*float(std_fluxtable.loc[[99]]['Binst']) + 7.716
#magzp_B_error = np.sqrt((float(std_fluxtable.loc[[99]]['Rinst_err']))**2 + (0.016))

#Finally, calculate the apparent magnitudes of the the cluster stars!
#zpcalc(magzp_V, magzp_V_error, "V", cls_fluxtable)
#zpcalc(magzp_R, magzp_R_error, "R", cls_fluxtable)
#zpcalc(magzp_B, magzp_B_error, "B", cls_fluxtable)

#Also calculate the V-R and B-V magnitude
#cls_fluxtable['V-R'] = cls_fluxtable['Vmag'] - cls_fluxtable['Rmag']
#cls_fluxtable['V-R_err'] = np.sqrt((cls_fluxtable['Vmag_err'])**2 + (cls_fluxtable['Rmag_err'])**2)
#cls_fluxtable['B-V'] = cls_fluxtable['Bmag'] - cls_fluxtable['Vmag']
#cls_fluxtable['B-V_err'] = np.sqrt((cls_fluxtable['Bmag_err'])**2 + (cls_fluxtable['Vmag_err'])**2)

#savechoice = input("Do you want the fluxtables to be saved (True/False)?")

#if savechoice == True:
#    #Now, save and read them to CSVs (if desired)
#    csv_folder = input("CSV savefolder:")
#    std_fluxtable.to_csv(csv_folder + "/std_photometry.csv")
#    cls_fluxtable.to_csv(csv_folder + "/cls_photometry.csv")
#
#    std_fluxtable = pd.read_csv(csv_folder + "/std_photometry.csv")
#    cls_fluxtable = pd.read_csv(csv_folder + "/cls_photometry.csv")
#else:
#    std_fluxtable = std_fluxtable
#    cls_fluxtable = cls_fluxtable

#Next, upload the isochrone data for chi-by-eye fitting
#iso_folder = input("Isochrone folder: ")
#ten_billion = pd.read_csv(iso_folder + "/isochrones_marigo08_1e10yr.txt", sep='/s+', skiprows=10) 
#ten_million = pd.read_csv(iso_folder + "/isochrones_marigo08_1e7yr.txt", sep='/s+', skiprows=10)
#hundred_million = pd.read_csv(iso_folder + "/isochrones_marigo08_1e8yr.txt", sep='/s+', skiprows=10)
#one_billion = pd.read_csv(iso_folder + "/isochrones_marigo08_1e9yr.txt", sep='/s+', skiprows=10)
#thirty_million = pd.read_csv(iso_folder + "/isochrones_marigo08_3e7yr.txt", sep='/s+', skiprows=10)
#three_hundred_million = pd.read_csv(iso_folder + "/isochrones_marigo08_3e8yr.txt", sep='/s+', skiprows=10) 

##Plot the isochrones first (B-V)
#plt.plot(ten_million['B']-ten_million['V'],ten_million['V'], label = '$1*10^{7}$ yrs old')
#plt.plot(thirty_million['B']-thirty_million['V'],thirty_million['V'], label = '$3*10^{7}$ yrs old')
#plt.plot(hundred_million['B']-hundred_million['V'],hundred_million['V'], label = '$1*10^{8}$ yrs old')
#plt.plot(three_hundred_million['B']-three_hundred_million['V'],three_hundred_million['V'], label = '$3*10^{8}$ yrs old')
#plt.plot(one_billion['B']-one_billion['V'],one_billion['V'], label = '$1*10^{9}$ yrs old')
#plt.plot(ten_billion['B']-ten_billion['V'],ten_billion['V'], label = '$1*10^{10}$ yrs old')
#plt.legend()
#plt.xlabel('B-V Magnitude', fontsize = 17.5)
#plt.ylabel('V Band Magnitude', fontsize = 17.5)
#plt.legend(fontsize = 12)  
#plt.ylim(12.5,-9)
#plt.show()

#Perform a magnitude cut at 14 as that was our approximate limiting magitude for the telescope
#cls_14 = cls_fluxtable[cls_fluxtable['Vmag'] < 13]

#Shift the requisite arrays based on reddening and distance modulus
#cls_14['B-V_shift'] = cls_14['B-V'] - 0.8
#ten_million['V_shift'] = ten_million['V'] + 10
#thirty_million['V_shift'] = thirty_million['V'] + 10
#hundred_million['V_shift'] = hundred_million['V'] + 10

#Let's plot our color magnitude diagram
#plt.figure(figsize=(8,8))
#plt.errorbar(cls_14['B-V_shift'], cls_14['Vmag'], 
#             xerr = cls_14['B-V_err'], yerr = cls_14['Vmag_err'],
#             fmt='o',
#             label = 'Cluster Photometry')
#Plotting the isochrones
#plt.plot(ten_million['B']-ten_million['V'], ten_million['V_shift'], label = '$1*10^{7}$ yrs old')
#plt.plot(thirty_million['B']-thirty_million['V'], thirty_million['V_shift'], label = '$3*10^{7}$ yrs old')
#plt.plot(hundred_million['B']-hundred_million['V'], hundred_million['V_shift'], label = '$1*10^{8}$ yrs old')

#Plot details
#plt.ylim([18,0])
#plt.xlim([-1,2.5])
#plt.title('B-V Color Magnitude Diagram Scaled to HIP 112998')
#plt.xlabel('B-V Magnitude', fontsize = 17.5)
#plt.ylabel('V Band Magnitude', fontsize = 17.5)
#plt.legend(fontsize = 12)
#plt.show()

#Finally, calculate the orthogonal residuals

#Setting up the residual arrays to store in
#tmil_residuals = []
#thmil_residuals = []
#hmil_residuals = []

#Run the nested for loops
#for n in cls_14.index:
#    #Resettable ararys for the individual star residuals
#    ten_million_resi = []
#    thirty_million_resi = []
#    hundred_million_resi = []
    #Different for loops for the 3 closest isochrones
#    for g in np.arange(0,len(ten_million)):
#        ten_million_resi.append(np.sqrt(
#            (ten_million['B'][g] - ten_million['V'][g] - cls_14['B-V'][n])**2+
#            (ten_million['V_shift'][g] - cls_14['Vmag'][n])**2))
#    for g in np.arange(0,len(ten_million)):
#        thirty_million_resi.append(np.sqrt(
#            (thirty_million['B'][g] - thirty_million['V'][g] - cls_14['B-V'][n])**2+
#            (thirty_million['V_shift'][g] - cls_14['Vmag'][n])**2))
#    for g in np.arange(0,len(hundred_million)):
#        hundred_million_resi.append(np.sqrt(
#            (hundred_million['B'][g] - hundred_million['V'][g] - cls_14['B-V'][n])**2+
#            (hundred_million['V_shift'][g] - cls_14['Vmag'][n])**2))
#    tmil_residuals.append(min(ten_million_resi)**2)
#    thmil_residuals.append(min(thirty_million_resi)**2)
#    hmil_residuals.append(min(hundred_million_resi)**2)
    
#tmil_sum = sum(tmil_residuals)
#hmil_sum = sum(hmil_residuals)
#thmil_sum = sum(thmil_residuals)

#Finally, print the residuals
#print("Ten Million Year Isochrone ChiSq: " + str(tmil_sum))
#print("Thirty Million Year Isochrone ChiSq: " + str(thmil_sum))
#print("Hundred Million Year Isochrone ChiSq: " + str(hmil_sum))