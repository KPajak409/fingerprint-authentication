# Fingerprint authentication system

Project utilizes a **Siamese Neural Network** for fingerprint comparison and authentication. This system aims to verify whether two fingerprint scans belong to the same individual. 
Here are the key components and implementation details:


## Project demo
To test how model performs at different image pairs, you can find demo version at the link below.\
https://huggingface.co/spaces/Padzong/fingerprint-auth-app

## Project overview
1. The primary goal of this project is to create a robust and accurate fingerprint authentication system.
2. It leverages a Siamese Neural Network, which is well-suited for similarity matching tasks like fingerprint matching.
3. The system compares pairs of fingerprint scans and determines whether they correspond to the same person.

## Workflow
### Data Collection:
* Fingerprint scans are obtained using a physical scanner.
* The scanner communicates with an Arduino board, which acts as an interface between the scanner and the computer.
* The publicly available scans serve as the training and testing data for the neural network.
### Siamese Neural Network:
* The Siamese architecture consists of two identical convolutional neural networks (twins) that share weights.
* Each twin processes one of the input fingerprint images.
* The output embeddings from both twins are compared to compute a similarity score.
### Training:
* The Siamese network is trained using PyTorch, a popular deep learning framework.
* Training involves minimizing **contrastive loss**.

# Required software 

- Python 3.11.5,
- Miniconda (Conda 23.11.0).
- Arduino IDE 2.2.1

# Fingerprint scanner

Project is adjusted to use in real life scenario by taking physically fingerprint scan and afterwards authenticate user based on his fingerprint. To communicate with the scanner you can use various microcontrollers like arduino uno, raspberry pi or rasbperry pico. There is multiple ways to use fingerprint scanner, usage may be slightly different depending on which option you choose. You can even skip the use of an external device and provide image directly to test login feature, but it must meet the image size requirements. We decided to use arduino uno with the following fingerprint scanner:

Model: SEN0188
Manufacturer: DFRobot

Link to user's manual of scanner: https://wiki.dfrobot.com/SEN0188_Fingerprint

Link to official manufacturer's scanner: https://www.dfrobot.com/index.php?route=product/product&product_id=1343&search=sen0188&description=true#.Vl6XMb_W2Hs

Link to polish offer for scanner: https://botland.com.pl/czytniki-linii-papilarnych/5060-czytnik-linii-papilarnych-z70-czujnik-odciskow-palcow-5903351241434.html

Datasheet for scanner can be found here: https://drive.google.com/drive/folders/1yKTjJ7-Ptg1mjBRmqJ_CpcXHdhb-kwfR?usp=drive_link

Code necessary for arduino to communicate with the scanner is in the path arduino/fingersave.ino

# Documentation

Documentation is in the path: docs/html/index.html

# Installation

- Clone the project to the desired destination or download whole project and unzip it in the destination,
- Go to the directory with the project and in the terminal, compile following commands:
     -  conda env create -f environment.yml
     -  conda activate isu-proj
 
# User guide to starting the program

- Download pre-trained model and dataset from this link: https://drive.google.com/drive/folders/1yKTjJ7-Ptg1mjBRmqJ_CpcXHdhb-kwfR?usp=drive_link,
- Put the trained model in the root directory of the cloned project on your device,
- Connect scanner and arduino with the computer,
- Make sure to be on the root directory in the console,
- Use following command to trigger console user interface: python program.py.





