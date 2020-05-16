
import cv2
import glob
import numpy as np
import random as r

files = glob.glob("landscapes_small/mountain/*.jpg")
print(files)

count = 0

orig = []
cut = []

for f in files[:]:
 
    img = cv2.imread(f)
    
    x1 = r.randint(32,98)
    y1 = r.randint(32,98)

    crop = cv2.resize(img,(128, 128))
    orig.append(np.array(crop))
    
    cv2.rectangle(crop, (x1,y1), (x1+32, y1+32), (255,255,255), -1)
    cut.append(np.array(crop))
    count+=1
        
cut = np.array(cut)
orig = np.array(orig)

print(cut.shape)
print(orig.shape)

np.save('orig_new', orig)
np.save('cut_new', cut)
