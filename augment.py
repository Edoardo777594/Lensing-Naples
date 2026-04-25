import cv2
import csv
import time
import random
import numpy as np
from skimage.util import random_noise
from PIL import Image,ImageEnhance
import astropy.io.fits as pyfits
from skimage import transform,data


#####################data reading########################
def func_csv_read(band=1):
    if band==1:
        pos_csv='./csv_file/train/train_pos_rband.csv'
        neg_csv='./csv_file/train/train_neg_rband.csv'
    else:
        pos_csv='./csv_file/train/train_pos_3band.csv'
        neg_csv='./csv_file/train/train_neg_3band.csv'
        
    x=[]
    y=[]
    with open (pos_csv) as f:
        reader=csv.DictReader(f)
        for row in reader:
            x.append(row['name'])
            y.append(1)        
    with open (neg_csv) as f:
        reader=csv.DictReader(f)
        for row in reader:
            x.append(row['name'])
            y.append(0)
    idxs = list(range(len(y)))
    #np.random.shuffle(idxs)
    X = np.array(x)[idxs]
    Y = np.array(y)[idxs]
    return X,Y
    
def func_data_read(name,band=1):
    if band==1:
       with pyfits.open(name,memmap=False) as f:
            img=f[0].data 
            
            scale_min = 0
            scale_max = img.max()
            img=img.clip(min=scale_min, max=scale_max)
            
            """indices = np.where(img< 0)
            img[indices] = 0.0
            img = np.sqrt(img)"""
            max_img = img.max()
            img = ((img/max_img)*255.)
    else:
        img=cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

#####################data augmentation########################
#------------------1.Zoom in and out---------------------
def func_rescale(img,band=1):
    if band==1:  
        h, w = img.shape
        rescale_factor=random.uniform(0.9, 1.1)
        img_new=transform.resize(img, (int(h * rescale_factor), int(w * rescale_factor)))
        h_new, w_new= img_new.shape
        if rescale_factor>=1:
            img_new = img_new[int((h_new-h)/2) : int((h_new+h)/ 2), int((w_new-h)/2) : int((w_new+h)/ 2)]
        else:
            img_new=np.pad(img_new, ((int((h-h_new)/2),int((h-h_new)/2)),(int((w-w_new)/2),int((w-w_new)/ 2))), 'constant', constant_values=0)
         
    else:
        h, w, _ = img.shape
        rescale_factor=random.uniform(0.9, 1.1)
        img_new = cv2.resize(img, (int(h * rescale_factor), int(w * rescale_factor)))
        h_new, w_new, _ = img_new.shape
        if rescale_factor>=1:
            img_new = img_new[int((h_new-h)/2) : int((h_new+h)/ 2), int((w_new-h)/2) : int((w_new+h)/ 2)]
        else:
            img_new=np.pad(img_new, ((int((h-h_new)/2),int((h-h_new)/2)),(int((w-w_new)/2),int((w-w_new)/ 2)), (0, 0)), 'constant', constant_values=0)
    return img_new
 
#-------------------2.shift and crop-------------------
def func_shift_crop(img,band=1):
    if band==1:
        h, w = img.shape
        pix1=int((h-85)/2)
        pix2=h-int((h-85)/2)
        if pix2-pix1==86:
           pix2=pix2-1
    else:
        h, w, _ = img.shape
        pix1=int((h-85)/2)
        pix2=h-int((h-85)/2)
        if pix2-pix1==86:
           pix2=pix2-1
        
    shift_left_right=random.randint(-4,4)
    shift_up_down=random.randint(-4,4)
    mat_shift = np.float32([[1,0,shift_left_right], [0,1,shift_up_down]])     
    img_new = cv2.warpAffine(img, mat_shift, (h, w))
    img_new = img[pix1:pix2, pix1:pix2] # The cropping coordinates are[y0:y1, x0:x1]
    return img_new

#-------------------3.Mirror and rotation-------------------    
def func_flip_rotate(img,band=1): 
    if random.choice([0,1])==0:   
        img = cv2.flip(img,1)   #horizontal mirroring
    if random.choice([0,1])==0:
        img = cv2.flip(img,0)   #vertical mirroring
        
    if band==1:     
        h, w = img.shape
    else:
        h, w, _ = img.shape
        
    ang=random.choice([0,90,180,270])
    if ang !=0:
        M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1)
        img = cv2.warpAffine(img, M,(w, h))
    img_new=img    
    return img_new

#-----------------4.Gaussian noise----------------------
def func_gaussian_noise(img,band=1): 
    var=10**(np.random.uniform (np.log10(1), np.log10(5)))
    if band==3:
        var=var/255.0/10
    img_new=random_noise(img, mode='gaussian',mean=0,var=var) 
    img_new = cv2.normalize(img_new, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    return img_new

#------------------5.Color change------------------------    
def func_randomColor(img,band=1):
    cj_img = Image.fromarray(img)
    if band==1:
        cj_img = cj_img.convert('L')
    
    saturation=0.5
    brightness=0.5
    contrast=0.5
    sharpness=0.5
    if random.random() < saturation:
        random_factor = random.uniform(0.8, 1.2)
        cj_img = ImageEnhance.Color( cj_img).enhance(random_factor)  #saturation
    if random.random() < brightness:
        random_factor = random.uniform(0.8, 1.2)
        cj_img = ImageEnhance.Brightness(cj_img).enhance(random_factor)  #luminance
    if random.random() < contrast:
        random_factor = random.uniform(0.8, 1.2)
        cj_img = ImageEnhance.Contrast( cj_img).enhance(random_factor)  #contrast
    if random.random() < sharpness:
        random_factor = random.uniform(0.8, 1.2)
        cj_img=ImageEnhance.Sharpness(cj_img).enhance(random_factor)  #sharpness
    img_new=np.asarray(cj_img)
    return img_new   
    
def func_argu(img,band=1):
    img_new=func_rescale(img,band=band)
    img_new=func_shift_crop(img_new,band=band)
    img_new=func_flip_rotate(img_new,band=band)
    img_new=func_gaussian_noise(img_new,band)
    img_new=func_randomColor(img_new,band=band)
    img_new = cv2.normalize(img_new, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    if band==1:
        img_new=np.expand_dims(img_new, 2)
    return img_new 
     
def func_data_arg(X,Y,batch_size,band=1):
    cnt=0
    x=[]
    y=[]
    while True:
        for i in range(len(X)):
            name=X[i]
            img= func_data_read(name,band=band)
            img_new=func_argu(img,band=band)
            if img_new is not None:
                x.append(img_new)
                y.append(Y[i])
                cnt += 1
                if cnt == batch_size:
                    cnt=0 
                    x_train = np.array(x)
                    y_train = np.array(y)
                    yield x_train,y_train
                    x,y=[],[]

#####################################
def flip_row(a):
    a = np.array(a)
    b = np.empty_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b[len(b)-1-i,j] = a[i,j]
    return b
def flip_col(a):
    a = np.array(a)
    b = np.empty_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b[i,len(b) - 1 - j] = a[i, j]
    return b
def transpose(a):
    a = np.array(a)
    b = np.empty_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b[j,i] = a[i, j]
    return b
def rotate_90(a):
    a = np.array(a)
    b = flip_row(a)
    return transpose(b)
def rotate_270(a):
    a = np.array(a)
    b = flip_col(a)
    return transpose(b)
def color_flip(a):
    img = np.empty_like(a)
    img[:,:,0] = flip_col(a[:,:,0])
    img[:,:,1] = flip_col(a[:,:,1])
    img[:,:,2] = flip_col(a[:,:,2])
    return img
def color_rot(a):
    img = np.empty_like(a)
    img[:, :, 0] = rotate_90(a[:, :, 0])
    img[:, :, 1] = rotate_90(a[:, :, 1])
    img[:, :, 2] = rotate_90(a[:, :, 2])
    return img
'''                
def func_data_arg1(X,Y,band=1):
    cnt=0
    x=[]
    y=[]
    num=len(X)
    while True:
        for i in range(len(X)):
            if i%500==0:
                print ('lens finished:', round(i*100/num,3),'%')
            name=X[i]
            img= func_data_read(name,band=band)
            img_new=func_argu(img,band=band)
            if img_new is None:
                print(name)
                
band=1                
X,Y=func_csv_read(band=band)
func_data_arg1(X,Y,band=band)
'''    