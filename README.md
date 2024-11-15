# Fingerprint authentication system

Project utilizes a **Siamese Neural Network** for fingerprint comparison and authentication. This system aims to verify whether two fingerprint scans belong to the same individual.
* The primary goal of this project is to create a robust and accurate fingerprint authentication system.
* It leverages a Siamese Neural Network, which is well-suited for similarity matching tasks like fingerprint matching.

# Table of contents
* [Workflow](#WorkFlow)
* [Required software](#Required-software)
* [Fingerprint scanner](#Fingerprint-scanner)
* [Documentation](#Documentation)
* [Installation](#Installation)
* [User guide](#User-guide-to-starting-the-program)
* [Results](#Results-and-threshold-finding)

## Project demo
To test how model performs at different image pairs, you can find demo version at the link below.\
https://huggingface.co/spaces/Padzong/fingerprint-auth-app

# Workflow
### Data Collection:
* Fingerprint scans are obtained using a physical scanner.
* The scanner communicates with an Arduino board, which acts as an interface between the scanner and the computer.
* The publicly available scans serve as the training and testing data for the neural network.
### Data Cleaning and Preprocessing:
* Removing samples of very poor quality.
* Cutting excessive white background of the images.
* Dilatation - fills smallers holes on images and thickens them.
* binarization - changes pixels to 0 or 255, based on a threshold calculated globally on all pixels per image.
* Adjust to a common aspect ratio for all images.
* Downscaling prepared images to 136px x 153px size, this size allows to save the most of samples and is the same ratio as scans taken by scanner.
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

# Results and threshold finding
To achieve the best model performance it's crucial to find the appropriate threshold, to decide whether fingerprint scans represent the same or different finger. It is possible with help of distributing predicted values by model on histogram.

![Value distribution](value_dist.png)

The values are calculated by measuring euclidean distance between feature vectors received from model. The values close to 0 depict pairs representing the same fingerprint and values far from 0 depict pairs of different fingerprints. Thanks to visible gap in value distribution, the threshold value was chosen to be 0.5

![Confusion matrix](confusionmatrix.png)

With threshold at 0.5 the model achieves 100% accuracy at test samples, but it can be strict. It means sometimes scans representing the same fingerprint may be annotated as different.




