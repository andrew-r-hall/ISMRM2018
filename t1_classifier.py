import numpy as np
import nibabel as nib
import peak_finder as pk
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneOut

### Reads in a nifti image. Returns image data, affine, and header.
def read_image(path):
    img=nib.load(path)
    data = img.get_data()
    affine = img.get_affine()
    header = img.get_header()

    return(data,affine,header)

### Returns an image where only the masked areas are nonzero
def apply_masks(main_image , mask):
    mask[mask != 0] = 1
    masked = np.multiply(main_image,mask)
    return(masked)


### Applies a z-score normalization to the input dataset
def zscore_norm(image):
    mean = np.mean(image[image != 0])
    std = np.std(image[image != 0])
    zscaled = np.round(np.divide(np.subtract(image,mean),std) , decimals=3)
    return(zscaled)

def get_hists(main_image):
    h = np.histogram(main_image[main_image!=np.min(main_image)], bins=np.unique(main_image[main_image!=np.min(main_image)]))
    h,b =h[0],h[1][:-1]
    return(h,b)

def temp_master():
    y1h,y1b,y2h,y2b,o2h,o2b,o1h,o1b = [],[],[],[],[],[],[],[]
    x , y = [], []
    subjects = np.loadtxt('subs.txt' , dtype=str)
    for i in subjects:
        img_path = './freesurfer/'+i+'/brain.nii.gz'
        codex_path = './freesurfer/'+i+'/codex.txt'
        codex = np.loadtxt(codex_path , dtype=str)
        d,a,h=read_image(img_path)
        z = zscore_norm(d)
        h,b = get_hists(z)
        if(codex[2] == 'Young'):
            if((codex[-1] == '2' and str(i)[-1] == '2') or (codex[-1] == '1' and str(i)[-1] == '1')):
                #y2h.append(h)
                #y2b.append(b)
                #plt.subplot(221)
                #plt.scatter(b,h , color='r')
                p = pk.get_maxima(h)
                x.append(np.sort(h[p])[-3:])
                y.append(1)
                #plt.scatter(p,h[p] , color='r')
                #plt.plot(h , color='r')
                print(i,' a')
            elif((codex[-1] == '2' and str(i)[-1] != '2') or (codex[-1] == '1' and str(i)[-1] != '1')):
                #y1h.append(h)
                #y1b.append(b)
                #plt.subplot(222)
                #plt.scatter(b,h , color='b')
                p = pk.get_maxima(h)
                x.append(np.sort(h[p])[-3:])
                y.append(0)
                #plt.scatter(p,h[p])
                #plt.plot(h , color='b')
                print(i,' b')
        if(codex[2] == 'Old'):
            if((codex[-1] == '2' and str(i)[-1] == '2') or (codex[-1] == '1' and str(i)[-1] == '1')):
                #y2h.append(h)
                #y2b.append(b)
                #plt.subplot(223)
                #plt.scatter(b,h , color='r')
                p = pk.get_maxima(h)
                x.append(np.sort(h[p])[-3:])
                y.append(1)
                #plt.scatter(p,h[p] , color='r')
                #plt.plot(h , color='r')
                print(i,' c')
            elif((codex[-1] == '2' and str(i)[-1] != '2') or (codex[-1] == '1' and str(i)[-1] != '1')):
                #y1h.append(h)
                #y1b.append(b)
                #plt.subplot(224)
                #plt.scatter(b,h , color='b')
                p = pk.get_maxima(h)
                x.append(np.sort(h[p])[-3:])
                y.append(0)
                #plt.scatter(p,h[p] , color='b')
                #plt.plot(h , color='b')
                print(i,' d')
    plt.show()
    return(x,y)

def kmeans(x):
    kmeans = KMeans(n_clusters=4).fit(x)
    predictions = kmeans.labels_
    return predictions

def iou(y,predictions):
    inter = np.intersect1d(y,predictions)
    union = np.union1d(y,predictions)
    iou = len(inter)/len(union)
    return iou

def svc(xtrain,ytrain,xtest,ytest):
    s = LinearSVC()
    s.fit(xtrain,ytrain)
    predict = s.predict(xtest)
    score = iou(ytest,predictions)
    return(s)

def loo(x,y):
    l = LeaveOneOut()
    l.get_n_splits(x)
    for train_index , test_index in l.split(x):
        print('train:',train_indexm,'test:',text_index)
        x_train , x_test = x[train_index] ,
