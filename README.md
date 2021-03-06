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
1. A detailed report about the work done can be found [here](https://www.overleaf.com/14201045nbngtjzxgtrc)
2. A Google Slides presentation can be found [here](https://docs.google.com/presentation/d/1o2RH6WHfbfyuQad9ZDE3kQ5-N749o_uBFhq0lSWSTsE/edit?usp=sharing)


## Week 1: summary of two papers about VGG and SqueezeNet
1. paper reading: check the file named Paper_Summary
   
   
## Week 2: Object Recognition
### wights  
the weights of the model can be checked [here](https://drive.google.com/drive/folders/1xRXmhrm1Ng86Y3ANa_N83xyltfwZU_IP?usp=sharing)  

### Code and usage  
All the experiment configuration files are put into code/config foder. There is a file inside code folder named train.py  to run the code.  
You need to fix the paths for datasets in train.py for working on your machine, and then write the commond:
```
CUDA_VISIBLE_DEVICES=0 python train.py -c config/file.py -e experimentFolder -l resultsFolder -s datasetsFolder

```
where file.py is the configuration file for this test, and expName is the name of the directory where the results to be saved. resultsFolder is the path to save experiment, and datasetsFolder is the path of datasets.  

### Completeness of tasks
Below is the check list of our completeness of task. All the results of the tasks are shown in the [report](https://www.overleaf.com/14201045nbngtjzxgtrc)  
- [x] (a) Run the provided code  
          - Analysis the dataset  
          - Accuracy of train and test set  
          - Evaluate different techniques in the configuration file such as: crop vs. resize; division std normalization; ImageNet normalization
          - Transfer learning to Belgium traffic signs dataset 
          - Understand the code  
- [x] (b) Train the VGG16 network on a different dataset(KITTI)  
          - Trained from scratch 
- [x] (c) Implete a new network     
          - (c.1) we impleted ResNet network on TT100K and KITTI datasets 
- [x] (d) Boost the performance of the network  
          - Changing learning rate and data augmentation
- [x] (e) Report
 
### Results Summary  
![](results/1.1.png)  
![](results/1.2.png)  
![](results/1.3.png)  

## Week 3/4: Object Detection
### wights  
the weights of the model can be checked [here](https://drive.google.com/drive/folders/1Aw_FuOW_3VCYB5EoUSkCgMnOJGOtmCvN)  

### Code and usage  
All the experiment configuration files are put into code/config foder. There is a file inside code folder named train.py  to run the code.  
You need to fix the paths for datasets in train.py for working on your machine, and then write the commond:
```
CUDA_VISIBLE_DEVICES=0 python train.py -c config/file.py -e experimentFolder -l resultsFolder -s datasetsFolder

```
where file.py is the configuration file for this test, and expName is the name of the directory where the results to be saved. resultsFolder is the path to save experiment, and datasetsFolder is the path of datasets.  

### Completeness of tasks
Below is the check list of our completeness of task. All the results of the tasks are shown in the [report](https://www.overleaf.com/14201045nbngtjzxgtrc)  
- [x] (a) Run the provided code  
          - Analysis the dataset  
          - Calculate the f-score and FPS on train, val and test sets.
- [x] (b) Read two papers   
          - You Only Look at Once (YOLO)  
          - Single Shot MultiBox Detector (SSD)  
- [x] (c) Implement a new network           
          - (c.2) Implement our own network: training and testing on both tt100k and udacity    
- [x] (d) Train the networks on a different dataset  
          - Udacity  
- [x] (e) Boost the performance of our network  
          - data augmentation (both tt100k and udacity dataset)  
          - learning rate (both tt100k and udacity dataset)    
- [x] (f) Report  

### Results Summary  
![](results/2.1.png)  
![](results/2.2.png)  
![](results/2.3.png)  


## Week 5/6: Image Semantic Segmentation  
### wights  
the weights of the model can be checked [here](https://drive.google.com/drive/folders/1honFLzx-pXc6eClIu2-kfTtvX_82nUcC)  

### Code and usage  
All the experiment configuration files are put into code/config foder. There is a file inside code folder named train.py  to run the code.  
You need to fix the paths for datasets in train.py for working on your machine, and then write the commond:
```
CUDA_VISIBLE_DEVICES=0 python train.py -c config/file.py -e experimentFolder -l resultsFolder -s datasetsFolder

```
where file.py is the configuration file for this test, and expName is the name of the directory where the results to be saved. resultsFolder is the path to save experiment, and datasetsFolder is the path of datasets.  

### Completeness of tasks
Below is the check list of our completeness of task. All the results of the tasks are shown in the [report](https://www.overleaf.com/14201045nbngtjzxgtrc)  
- [x] (a) Run the provided code  
          - Analysis the dataset  
          - Calculate the f-score and FPS on train, val and test sets.
- [x] (b) Read two papers   
          - Fully convolutional networks for semantic segmentation    
          - Segnet         
- [x] (c) Implement a new network      
          - Segnet(basic and VGG)             
- [x] (d) Train the networks on a different dataset  
          - KITTI   
          - Cityscapes   
          - Pascal2012     
- [x] (e) Boost the performance of our network  
          - Data Augmentation      
          - Feature Normalization         
- [x] (f) Report  

### Results Summary  
![](results/3.1.png)  

