#%%
from pathlib import Path
from PIL import Image
import os, glob, math
import numpy as np
import cv2
import matplotlib.pyplot as plt


current_dir = Path(__file__)
project_dir = [p for p in current_dir.parents if p.parts[-1]=='fingerprint-authentication'][0]

dest_path_processed = f'{project_dir}\\scans\\processed\\'
source_path_raw = f'{project_dir}\\scans\\raw\\'
    
def scan_preprocess(file_name):
    threshold = 160

    img = np.array(Image.open(source_path_raw + file_name))
    left, right, up, down = 0,0,0,0

    # cutting white space in images
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

    img_stripped = img[left:right, up:down]
    print(f'removing white space {file_name}')

    # transformations on ds
    img_dilation = cv2.dilate(img_stripped, (5,5), iterations=1)
    #img_erosion = cv2.erode(img, (5,5), iterations=2) # not particularly useful
    img_binarized = cv2.adaptiveThreshold(img_dilation, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                                cv2.THRESH_BINARY, 11, 2)

    # scale down and save
    img_final = Image.fromarray(img_binarized)
    img_final.thumbnail((136,153))
    img_final.save(dest_path_processed+file_name)
    

# -------------------------------------------------------------------------
# Written by Brian Ejike (2017)
# Distributed under the MIT License

import serial, time, uuid
from pathlib import Path


WIDTH = 256
HEIGHT = 288
READ_LEN = int(WIDTH * HEIGHT / 2)
    
DEPTH = 8
HEADER_SZ = 54

portSettings = ['COM6', 57600]

# assemble bmp header for a grayscale image
def assembleHeader(width, height, depth, cTable=False):
    header = bytearray(HEADER_SZ)
    header[0:2] = b'BM'   # bmp signature
    byte_width = int((depth*width + 31) / 32) * 4
    if cTable:
        header[2:6] = ((byte_width * height) + (2**depth)*4 + HEADER_SZ).to_bytes(4, byteorder='little')  #file size
    else:
        header[2:6] = ((byte_width * height) + HEADER_SZ).to_bytes(4, byteorder='little')  #file size
    #header[6:10] = (0).to_bytes(4, byteorder='little')
    if cTable:
        header[10:14] = ((2**depth) * 4 + HEADER_SZ).to_bytes(4, byteorder='little') #offset
    else:
        header[10:14] = (HEADER_SZ).to_bytes(4, byteorder='little') #offset

    header[14:18] = (40).to_bytes(4, byteorder='little')    #file header size
    header[18:22] = width.to_bytes(4, byteorder='little') #width
    header[22:26] = (-height).to_bytes(4, byteorder='little', signed=True) #height
    header[26:28] = (1).to_bytes(2, byteorder='little') #no of planes
    header[28:30] = depth.to_bytes(2, byteorder='little') #depth
    #header[30:34] = (0).to_bytes(4, byteorder='little')
    header[34:38] = (byte_width * height).to_bytes(4, byteorder='little') #image size
    header[38:42] = (1).to_bytes(4, byteorder='little') #resolution
    header[42:46] = (1).to_bytes(4, byteorder='little')
    #header[46:50] = (0).to_bytes(4, byteorder='little')
    #header[50:54] = (0).to_bytes(4, byteorder='little')
    return header
    
    
def getPrint():
    user_unique_id = uuid.uuid4()
    # version for testing
    # out = open(input("Enter filename/path of output file (without extension): ")+'.bmp', 'xb',)
    # path to create scan named by user id
    out = open(f'{source_path_raw}{user_unique_id}.bmp', 'xb')
    
    
    # assemble and write the BMP header to the file
    out.write(assembleHeader(WIDTH, HEIGHT, DEPTH, True))
    for i in range(256):
        # write the colour palette
        out.write(i.to_bytes(1,byteorder='little') * 4)
    try:
        # open the port; timeout is 1 sec; also resets the arduino
        ser = serial.Serial(portSettings[0], portSettings[1], timeout=1)
    except Exception as e:
        print('Invalid port settings:', e)
        print()
        out.close()
        return False
    while ser.isOpen():
        try:
            # assumes everything recved at first is printable ascii
            curr = ser.read().decode()
            # based on the image_to_pc sketch, \t indicates start of the stream
            if curr != '\t':
                # print the debug messages from arduino running the image_to_pc sketch
                print(curr, end='')
                continue
            for i in range(READ_LEN): # start recving image 
                byte = ser.read()
                # if we get nothing after the 1 sec timeout period
                if not byte:
                    print("Timeout!")
                    out.close()  # close port and file
                    ser.close()
                    return False
                    
                # Since each received byte contains info for 2 adjacent pixels,
                # assume that both pixels were originally close enough in colour
                # to now be assigned the same colour
                out.write(byte * 2)
                
            out.close()  # close file
            print('Image saved as', out.name)
            
            # read anything that's left and print
            left = ser.read(100)
            print(left.decode('ascii', errors='ignore'))
            ser.close()
            
            print()
            return out.name
        except Exception as e:
            print("Read failed: ", e)
            out.close()
            ser.close()
            os.remove(f'{source_path_raw}{user_unique_id}.bmp')
            return False
        except KeyboardInterrupt:
            print("Closing port.")
            out.close()
            ser.close()
            os.remove(f'{source_path_raw}{user_unique_id}.bmp')
            return False
    
