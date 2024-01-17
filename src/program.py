#%%
"""
This module is main program for operating on the system with
sign in and registration operations.
"""
import others.take_scan as scan
from pathlib import Path
import glob, os, torch
from torchvision.transforms import ToTensor
from models.siamese_nn import Siamese_nn
from PIL import Image
import re
import matplotlib.pyplot as plt
import torch.nn.functional as F
current_dir = Path(__file__)
project_dir = [p for p in current_dir.parents if p.parts[-1]=='fingerprint-authentication'][0]
processed_path = f'{project_dir}\\scans\\processed\\'

to_tensor = ToTensor()

model = Siamese_nn()
weights = torch.load(f'{project_dir}/models/6cv_1fc_lr0001_douczka')
model.load_state_dict(weights)
model.eval()

def isValidUsername(username):
    """
    This method is responsible for validation of username 
    given by user.
    
    Simple rules applied by regex operation:
        - username includes small and big characters,
        - username includes digits from 0 to 9,
        - username includes special characters such as: - and _ ,
        - username has to be between 3 and 20 characters long.
    """
    pattern = r'^[a-zA-Z0-9_-]{3,20}$'
    match = re.match(pattern, username)
    if match:
        return True
    else:
        return False


# at first read and save user fingerprint scan and as success return path to saved image
# it obviously won't work if scanner isn't connected, you can use path to existing file instead in order to test scan_preprocessing

# later run preprocess on user scan by sending his id as argument
# finally you can load preprocessed user fingerprint scan by loading it from the path project_dir/scans/processed/user_id.bmp


def register():
    """
    This method is responsible for registration of user to the system.
    
    User writes down their desired username in the console.
    It is checked for validation regarding semantics and whether
    or not the username is already taken.
    
    If everything is successful then operation of fingerprint retrieval
    starts. Fingerprint then will be preprocessed and saved in '../data/raw' path.
    """
    print('Please write down your desired username: ')
    while True:
        username_to_check = input("Enter a username: ")
        if isValidUsername(username_to_check):
            print(f"The username '{(username_to_check):}' is valid!")
            user_file_name = f'{project_dir}\\data\\raw\\{username_to_check}.bmp'
            if not os.path.exists(user_file_name):
                break
            else:
                print('The username is already taken')
            
        else:
            print(f"The username '{username_to_check}' is not valid. It should contain only letters, numbers, hyphens, and underscores, and be between 3 and 20 characters long.")
    
    
    path = scan.getPrint(username_to_check)
    scan.scan_preprocess(username_to_check)
    

def login():
    """
    This method is responsible for signing in to the system.
    
    User writes down their username in the console. It is checked
    whether or not such username exists.
    
    If it exists then user will be asked to scan his fingerprint. 
    After successful scanning, then user's registration fingerprint will be retrieved. 
    If verification will go successfully, then access to the system will be granted.
    """
    print('Please write down your username: ')
    while True:
        username = input('Enter a username: ')
        username = username.strip()
        user_file_name = f'{project_dir}\\scans\\processed\\{username}.bmp'
        if os.path.exists(user_file_name):
            break
        else:
            print(f"The username '{username}' does not exists.")
    
    try:
        existing_user_scan = Image.open(f'{processed_path}{username}.bmp')
        #path = scan.getPrint(f'{username}')
        scan_taken = scan.scan_preprocess(f'{username}', login=True)
        
    except Exception as e:
        print('Read file failed: ', e)
    
    existing_user_scan = to_tensor(existing_user_scan)
    scan_taken = to_tensor(scan_taken)
    
    
    exist_scan = torch.permute(existing_user_scan, (1,2,0))
    scan_taken2 = torch.permute(scan_taken, (1,2,0))
    
    existing_user_scan = torch.unsqueeze(existing_user_scan, dim = 0)
    scan_taken = torch.unsqueeze(scan_taken, dim = 0)
    
    #print(existing_user_scan.shape, scan_taken.shape)
     
    plt.subplot(1,2,1)
    plt.imshow(exist_scan)
    plt.subplot(1,2,2)
    plt.imshow(scan_taken2)
    with torch.no_grad():
        out1, out2 = model(existing_user_scan, scan_taken)
    confidence = F.pairwise_distance(out1, out2)
    
    print(confidence)
    if confidence < 0.5:
        print('Access granted')
        return
    else:
        print('Access NOT granted')
        return

#%%

def mainProgram():
    """
    This method is responsible for console menu, in which we can
    try to login or/and register to the system.
    
    The function showcases 3 options: 1-login, 2-registration and 9-exit the program.
    
    First option triggers login() function.
    Second option triggers register() function.
    Last option exits the program and shuts down the console.
    """ 
    while True:
        print("Menu:")
        print("1 - Login")
        print("2 - Registration")
        print("9 - Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            login()
        elif choice == '2':
            register()
        elif choice == '9':
            print("Exiting the menu. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a valid option.")
    pass

# %%
