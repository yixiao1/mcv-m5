# Scene Understanding for Autonomous Vehicles
Master in Computer Vision - M5 Visual recognition

## Group 06
Name: Group 06 (Tricky Team)  
Juan Felipe Montesinos(jfmontgar@gmail.com)  
Ferran Carrasquer(ferrancarrasquer@gmail.com)  
Yi Xiao(yi.xiao@e-campus.uab.cat)  

## Abstract   
In this 5-week project it will be developed as a Deep Learning based system to recognise objects, obtain their bounding box and segmentation related to the field of ADAS for designing and implementig deep neural networks for scene understanding for autonomous vehicles.

## Report & Slides
1. A detailed report about the work done can be found [here](https://www.overleaf.com/read/bcqybcqytyvj)
2. A Google Slides presentation can be found [here]()

## Week 1: summary of two papers about VGG and SqueezeNet
1. paper reading: check the file named Paper_Summary

Short explanation of the code in the repository
Results of the different experiments
Instructions for using the code
Indicate the level of completeness of the goals of this week


## Week 2: Object Recognition
1. the weights of the model can be checked [here]()
2. Code and usage
All the experiment configuration files are put into code/config foder. There is a file inside code folder named train.py. 
To run the code, you need to fix the paths for datasets in train.py for working on your machine, and then write the commond: python train.py -c config/dataset.py -e expName, where dataset.py is the configuration file for this test, and expName is the name of the directory where the results to be saved.

3. Completeness of task
For the object recognition we implement several architectures, training them from scratch as well as fine-tuning using pretrained weights. We also boost the performance of the networks using different pre-processing techniques, and performing data augmentation and hyperparameter optimization.
Below is the check list of our completeness of task
 - [x] (a) Run the provided code
            - Analysis the dataset
            - Accuracy of train and test set
            - Evaluate different techniques in the configuration file 
                i. crop vs. resize;
                ii. other pre-processings:
            - Transfer learning to Belgium traffic signs dataset 
            - Understand the code
 - [x] (b) Train the network on a different dataset(KITTI)
            - we trained from scratch (or fine-tuning)? T.B.D



