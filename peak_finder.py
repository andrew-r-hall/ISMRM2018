# Histogram Peak Detector
import sys
#import cv2
import numpy as np
import nibabel as nib
import scipy.misc as misc
import matplotlib.pyplot as plt

# Returns a list where a where nonzero values are minima
def find_maxima(smoothed_hist):
    copy = np.copy(smoothed_hist)
    minima = np.zeros_like(smoothed_hist)
    for i in range(int(np.max(smoothed_hist))):
        copy[smoothed_hist >= i] = 1
        copy[smoothed_hist <  i] = 0
        #print(copy)
        for j in range(len(copy)):
            try:
                if(copy[j] == 1 and (copy[j-1] == 0) and (copy[j+1] == 0) and (copy[j-2] == 0) and (copy[j+2] == 0)):
                    minima[j] = 1
            except IndexError:
                pass
    #minima = np.multiply(minima , smoothed_hist)
    min_locations = [0]
    for i in range(len(minima)):
        if(minima[i] != 0):
            min_locations.append(i)
    return(min_locations)


def get_maxima(hist):

    #hist = get_hist(cv)
    #sm = smooth(hist[1:] , window_len=3)[3:]
    mins = find_maxima(hist)

    return(mins)
