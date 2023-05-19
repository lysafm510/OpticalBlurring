from math import log, sqrt

def max(FWHM):
    threshold = 0.01
    sigma = (FWHM ** 2) / (8 * log(2.0))
    x = sqrt(-2*log(threshold)*sigma)
    print(x)

max(542)
