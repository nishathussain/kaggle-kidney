import logging, cv2, os,torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from torch import nn
from torch.utils.data import Dataset, DataLoader
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
##    C      O      N      F      I      G   ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
fold = 0
nfolds = 4 # 3 folds in train 1 in test
bs = 4
SEED = 2020
# TRAIN = '../input/hubmap-256x256/train/'
# MASKS = '../input/hubmap-256x256/masks/'
TRAIN = '../input/hubmap-512x512/train/'
MASKS = '../input/hubmap-512x512/masks/'
LABELS = '../input/hubmap-kidney-segmentation/train.csv'
NUM_WORKERS = 1
mean=np.array([0.65459856, 0.48386562, 0.69428385]); std=np.array([0.15167958, 0.23584107, 0.13146145])
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
##       E             N             D       ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

class HuBMAPDataset(Dataset):
    def __init__(self, fold=None, train=True, tfms=None):
        print('fold: ',fold,'train: ',train)
        ids = pd.read_csv(LABELS).id.values
        kf = KFold(n_splits=nfolds,random_state=SEED,shuffle=True)
        ids = set(ids[list(kf.split(ids))[fold][0 if train else 1]])
        self.fnames = [fname for fname in os.listdir(TRAIN) if fname.split('_')[0] in ids]
        self.train = train
        self.tfms = tfms
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(TRAIN,fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(MASKS,fname),cv2.IMREAD_GRAYSCALE)
        
        if self.tfms is not None:
            augmented = self.tfms(image=img,mask=mask)
            img,mask = augmented['image'],augmented['mask']
        return img2tensor((img/255.0 - mean)/std),img2tensor(mask)

# Data Augmentation    
def get_aug(p=1.0):
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, 
                         border_mode=cv2.BORDER_REFLECT),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        OneOf([
            HueSaturationValue(10,15,10),
            CLAHE(clip_limit=2),
            RandomBrightnessContrast(),            
        ], p=0.3),
    ], p=p)
    
def save_img(data,name,out):
    data = data.float().cpu().numpy()
    img = cv2.imencode('.png',(data*255).astype(np.uint8))[1]
    out.writestr(name, img)

