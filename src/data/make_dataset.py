#%%
from pathlib import Path
from PIL import Image
import os, glob, math
import numpy as np
import cv2
import matplotlib.pyplot as plt

current_dir = Path(__file__)
project_dir = [p for p in current_dir.parents if p.parts[-1]=='fingerprint-authentication'][0]

dest_path_processed = f'{project_dir}\\data\\processed\\'
dest_path_final = f'{project_dir}\\data\\final\\'

#%% cutting white space in images
paths_raw = sorted(glob.glob(f'{project_dir}\\data\\raw\\*',))
file_names_raw = sorted(os.listdir(f'{project_dir}\\data\\raw\\'))

threshold = 160
for path, file_name in zip(paths_raw, file_names_raw):
    img = np.array(Image.open(path))
    left, right, up, down = 0,0,0,0
    # remove black line on the right
    if img[50,-1]<200:
       img = img[:,0:img.shape[1]-15]

    for i in range(img.shape[0]-1): 
        if min(img[i,:]) < threshold:
            left = i
            break
    for i in range(img.shape[0]-1): 
        if min(img[-i-1,:]) < threshold:
            right = img.shape[0] - i
            break
    for i in range(img.shape[1]-1): 
        if min(img[:,i]) < threshold:
            up = i
            break
    for i in range(img.shape[1]-1):         
        if min(img[:,-i]) < threshold:
            down = img.shape[1] - i
            break

    new_img = Image.fromarray(img[left:right, up:down])
    
    print(f'removing white space {file_name}')
    new_img.save(dest_path_processed+file_name)


#%%
# transformations on ds
paths_processed = sorted(glob.glob(f'{project_dir}\\data\\processed\\*',))
file_names_processed = sorted(os.listdir(f'{project_dir}\\data\\processed\\'))

for path in paths_processed:
    img = cv2.imread(path, 0)
    img_dilation = cv2.dilate(img, (5,5), iterations=1)
    img_erosion = cv2.erode(img, (5,5), iterations=2)
    img_binarized = cv2.adaptiveThreshold(img_dilation, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                               cv2.THRESH_BINARY, 11, 2)
    #final_frame = cv2.hconcat([img, img_dilation, img_erosion, img_binarized])
    cv2.imwrite(path, img_binarized)

#%% cut from up and down to achieve the same width to height ratio - 1:125

for path, file_name in zip(paths_processed, file_names_processed):
    img = np.array(Image.open(path))
    height = img.shape[0]
    width = img.shape[1]

    # width:height ratio should be 1:1,125
    if round(height/width, 3) > 1.125:
        target_height = round(width * 1.125)
        n_rows_to_del = height - target_height
        if n_rows_to_del%2==0:
            n_rows_half = int(n_rows_to_del/2)
            bottom_idx = img.shape[0]-n_rows_half
            new_img = Image.fromarray(img[n_rows_half:bottom_idx, :])
        else:
            n_rows_half = int(math.floor(n_rows_to_del/2))
            bottom_idx = img.shape[0]-n_rows_half-1
            new_img = Image.fromarray(img[n_rows_half:bottom_idx, :])   
        new_img.thumbnail((136,153))   
        print(f'cutting {path}')
        new_img.save(dest_path_final+file_name) 
    elif round(height/width, 3) < 1.125:
        target_width = round(height/1.125)
        n_cols_to_del = width - target_width
        if n_cols_to_del%2==0:
            n_cols_half = int(n_cols_to_del/2)
            right_idx = img.shape[1]-n_cols_half
            new_img = Image.fromarray(img[:, n_cols_half:right_idx])
        else:
            n_cols_half = int(math.floor(n_cols_to_del/2))
            right_idx = img.shape[1]-n_cols_half-1
            new_img = Image.fromarray(img[:, n_cols_half:right_idx])
        new_img.thumbnail((136,153))   
        print(f'cutting {path}')
        new_img.save(dest_path_final+file_name)
    else:
        new_img.thumbnail((136,153))   
        print(f'cutting {path}')
        new_img.save(dest_path_final+file_name)





#%% removing bad samples 
to_remove = ['01_1','20_6','44_1','86_1','91_1','82_1','09_2']

for el in to_remove:
    os.remove(dest_path_final+el+'.tif')
    
paths_final = sorted(glob.glob(f'{project_dir}\\data\\final\\*',))
count = 0
to_check = []
for path in paths_final:
    img = np.array(Image.open(path))
    height = img.shape[0]
    width = img.shape[1]

    if height == 153 and width == 136:
        count += 1
    else:
        to_check.append(path)


print(count, len(paths_final))
print(to_check)

#%%
# paths_processed = sorted(glob.glob(f'{project_dir}\\data\\raw\\*',))



# img = cv2.imread(paths_processed[1], 0)

# #hist = cv2.calcHist(img, [0], None, [256], [0, 256])

# plt.hist(img)
# ret, img_binarized = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

# # con = cv2.hconcat([img, img_binarized])
# cv2.imshow('prr', img)
# cv2.imshow('prr2', img_binarized)
# cv2.waitKey(0)