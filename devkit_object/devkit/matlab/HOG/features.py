 import cv2
from matplotlib import pyplot as plt
import time
import cv
import numpy as np

from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter

def hog(image, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(3, 3), visualise=False, normalise=False):
    
    if len(image.shape) is 3:
            image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    image = np.atleast_2d(image)
    
    
    
    if normalise:
        image = sqrt(image)
    

    
    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)
    gy[:-1, :] = np.diff(image, n=1, axis=0)
    

    
    magnitude = sqrt(gx ** 2 + gy ** 2)
    orientation = arctan2(gy, (gx + 1e-15)) * (180 / pi) + 90
    
    sy, sx = image.shape
    cx, cy = pixels_per_cell
    bx, by = cells_per_block
    
    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y
    
    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    for i in range(orientations):
        #create new integral image for this orientation
        # isolate orientations in this range
        
        temp_ori = np.where(orientation < 180 / orientations * (i + 1),
                            orientation, 0)
        temp_ori = np.where(orientation >= 180 / orientations * i,
                                                temp_ori, 0)
                            # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, magnitude, 0)
                            
        orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cy, cx))[cy/2::cy, cx/2::cx]
    
    
    # now for each cell, compute the histogram
    #orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    
    radius = min(cx, cy) // 2 - 1
    


        
    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksy, n_blocksx,
                              by, bx, orientations))
    
    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[y:y + by, x:x + bx, :]
            eps = 1e-5
            normalised_blocks[y, x, :] = block / sqrt(block.sum() ** 2 + eps)
    


    return normalised_blocks.ravel()



def histogram ( img , verbose = False, mode = 'regular'):
    color = ('b','g','r')

    if verbose:
        plt.ion()
        plt.show()
        print "Extracting Histogram"


    one_channel = False
    change_flag = False
    if len(img.shape) < 3 :
        if mode is not 'regular' :
            print "Image is gray scale. Mode " + mode + " is not available. Defaulting to gray histogram"
            mode = 'gray'
            one_channel = True
            change_flag = True


    hist= []
    hist_output = []
    if mode is 'regular' :

        for i,col in enumerate(color):
            hist.append ( cv2.calcHist([img],[i],None,[256],[0,256]) )
            cv2.normalize(hist[i],hist[i],1)

            if verbose :
                plt.plot(hist[i],color = col)
                plt.xlim([0,256])
                plt.draw()
                time.sleep(0.05)
            hist_output.extend( hist[i].flatten() )



    elif mode is 'color':
        hist_output = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
        cv2.normalize(hist_output,hist_output,1)
        if verbose :
            plt.plot(hist_output,)
            plt.xlim([0,len(hist_output)])
            plt.draw()
            time.sleep(0.05)



    elif mode is 'gray':

        if one_channel is False :
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        if change_flag is False :
            hist_output = cv2.calcHist([img], [0], None,[256],[0,256])
        else :
            hist_output = cv2.calcHist([img], [0], None, [512], [0,256])

        cv2.normalize(hist_output,hist_output,1)
        if verbose :
            plt.plot(hist_output,)
            plt.xlim([0,len(hist_output)])
            plt.draw()
            time.sleep(0.05)

    if verbose :
        cv2.waitKey()

    return hist_output



def feature_detect ( detector, img , verbose = False ):

    if len(img.shape) is 3:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    forb = cv2.FeatureDetector_create(detector)
    kpts = forb.detect( img )
    if verbose:
        print detector, 'number of KeyPoint objects', len(kpts)
    return kpts



def extract_features(img, verbose = False):

    features = {}
    features ['histogram'] = histogram ( img , verbose ,'gray')

    detector_format = ["","Grid","Pyramid"]
    detector_types = ["FAST","STAR","SIFT","SURF","ORB","MSER","GFTT","HARRIS"]

    for form in detector_format:
        for detector in detector_types:
            features [form + detector] = feature_detect ( form + detector, img, verbose )
            if verbose:
                print len(features[form + detector])

    features['hog'] = hog(img , visualise = False , normalise = True)
    if verbose:
        print len(features['hog'])

    return features
