import cv2
import numpy as np


SURF_WINDOW = 20
DCT_WINDOW = 20
windowSize = 10
gridSpacing = 7

path='C:/Users/rolco/Desktop/test.jpg'

def load_image(path):
    '''
    Read in a file and separate into L*a*b* channels
    '''
        
    #read in original image
    img = cv2.imread(path) 
    
    #convert to L*a*b* space and split into channels
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    return l, a, b

l,a,b=load_image(path)

def feature_surf(img, pos):
    '''
    Gets the SURF descriptor of img at pos = (x,y).
    Assume img is a single channel image.
    '''
    octave2 = cv2.GaussianBlur(img, (0, 0), 1)
    octave3 = cv2.GaussianBlur(img, (0, 0), 2)
    kp = cv2.KeyPoint(pos[0], pos[1], SURF_WINDOW)
    surf = cv2.xfeatures2d.SURF_create()
    _, des1 = surf.compute(img, [kp])
    _, des2 = surf.compute(octave2, [kp])
    _, des3 = surf.compute(octave3, [kp])
    return np.concatenate((des1[0], des2[0], des3[0]))

def feature_dft(img, pos):
    xlim = (max(pos[0] - windowSize,0), min(pos[0] + windowSize,img.shape[1]))
    ylim = (max(pos[1] - windowSize,0), min(pos[1] + windowSize,img.shape[0]))
    patch = img[ylim[0]:ylim[1],xlim[0]:xlim[1]]
        
    l = (2*windowSize + 1)**2
    
    #return all zeros for now if we're at the edge
    if patch.shape[0]*patch.shape[1] != l:
        return np.zeros(l)
    return np.abs(np.fft(patch.flatten()))


#def feature_position(img, pos):
#    m,n = img.shape
#    x_pos = pos[0]/n
#    y_pos = pos[1]/m
    
#    return np.array([x_pos, y_pos])
            

def get_features(img, pos):
    intensity = np.array([img[pos[1], pos[0]]])
    #position = feature_position(img, pos)
    meanvar = np.array([getMean(img, pos), getVariance(img, pos)])
    feat = np.concatenate((meanvar, feature_surf(img, pos), feature_dft(img, pos)))
    return feat


def getMean(img, pos):
     ''' 
     Returns mean value over a windowed region around (x,y)
     '''
     
     xlim = (max(pos[0] - windowSize,0), min(pos[0] + windowSize,img.shape[1]))
     ylim = (max(pos[1] - windowSize,0), min(pos[1] + windowSize,img.shape[0]))
     return np.mean(img[ylim[0]:ylim[1],xlim[0]:xlim[1]])

        
def getVariance(img, pos):
    
    xlim = (max(pos[0] - windowSize,0), min(pos[0] + windowSize,img.shape[1]))
    ylim = (max(pos[1] - windowSize,0), min(pos[1] + windowSize,img.shape[0]))
    
    return np.var(img[ylim[0]:ylim[1],xlim[0]:xlim[1]])/1000


