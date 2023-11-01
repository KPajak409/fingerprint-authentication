#%%
from pathlib import Path
from PIL import Image
import os, glob, math
import numpy as np

current_dir = Path(__file__)
project_dir = [p for p in current_dir.parents if p.parts[-1]=='fingerprint-authentication'][0]
paths_raw = sorted(glob.glob(f'{project_dir}\\data\\raw\\*',))
file_names = sorted(os.listdir(f'{project_dir}\\data\\raw\\'))
dest_path = f'{project_dir}\\data\\processed\\'

#%% cutting white space in images
threshold = 160
for path, file_name in zip(paths_raw,file_names):
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
    new_img.save(dest_path+file_name)

#%% cut from up and down to achieve the same width to height ratio - 1:125
paths_processed = sorted(glob.glob(f'{project_dir}\\data\\processed\\*',))

for path in paths_processed:
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
        new_img.save(path) 
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
        new_img.save(path)
    else:
        new_img.thumbnail((136,153))   
        print(f'cutting {path}')
        new_img.save(path)

#%% removing bad samples and checking if every file has the same dimensions
to_remove = ['01_1','20_6','44_1','86_1','91_1','82_1','09_2']

for el in to_remove:
    os.remove(dest_path+el+'.tif')

paths_processed = sorted(glob.glob(f'{project_dir}\\data\\processed\\*',))
count = 0
to_check = []
for path in paths_processed:
    img = np.array(Image.open(path))
    height = img.shape[0]
    width = img.shape[1]

    if height == 153 and width == 136:
        count += 1
    else:
        to_check.append(path)


print(count, len(paths_processed))
print(to_check)