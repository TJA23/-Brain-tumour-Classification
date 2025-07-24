# Brain-tumour-Classification 

![Brain-MRI-Image-Classification-Using-Deep-Learning-Cover-Photo](https://github.com/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning/blob/main/Project-Cover-Photo.jpg)  

## Introduction  
Brain tumors are abnormal growths of cells within the brain that can be either malignant (cancerous) or benign (non-cancerous). Early and accurate diagnosis is critical for effective treatment planning and improving patient survival rates. Traditional diagnosis involves manual inspection of MRI scans by radiologists, a process that can be time-consuming and subject to human error, especially when tumor features are subtle.

To address these challenges, we leverage deep learning models to automate the classification of brain tumors from MRI images. This project aims to build a robust system capable of detecting and classifying brain tumors into four categories:
	•	Glioma Tumor
	•	Meningioma Tumor
	•	Pituitary Tumor
	•	No Tumor

**Using a combination of custom-built Convolutional Neural Networks (CNNs) and transfer learning with pre-trained models like VGG16, MobileNet, and ResNet50, our model is trained to analyze MRI images and predict the tumor type with high accuracy.**


## About Dataset  
- https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset

Provided by a Roboflow user
License: CC BY 4.0

This project has created a labeled MRI brain tumor dataset for the detection of three tumor types: pituitary, meningioma, and glioma. The dataset contains 2443 total images, which have been split into training, validation, and test sets. The training set has 1695 images, the validation set has 502 images, and the test set has 246 images.

**Data:**
* Number of images: 2443
* Image types: MRI scans

**Classes:**
* Pituitary tumor
* Meningioma tumor
* Glioma tumor
* No Tumor

**Split:**
* Training set: 1695 images
* Validation set: 502 images
* Test set: 246 images

**Labeling:**
* The images have been labeled by medical experts using a standardized labeling protocol.
* The labels include the type of tumor and the location of the tumor.
  

## Results  
- Developed 3 Deep Neural Network models i.e. Multi-Layer Perceptron, AlexNet-CNN, and Inception-V3 in order to classify the Brain MRI Images to 4 different independent classes.  
- Inception-V3 model used is a pre-trained on the ImageNet dataset which consist of 1K classes but for this project we have tuned the later part i.e. the Fully-Connected part of the model while retaining the weights of the CNN part to satisfy the needs of this work. 
- **The pre-trained Inception-V3 model has performed significantly well with an accuracy of `87.57%` as compare to AlexNet-CNN and Multi-Layer Perceptron deep neural network model.**  

## Future Works  
- To improve the robustness and accuracy of model further we can develop a efficient Data-Augmentation pipline in order to expose the CNN model to more variants of the Brain MRI Images.  
- Training process can be migrated to TPUs (Tensor Processing Units) by representing the data in TFRecord format for significant reduction in training time.  
- Implementation of [Region Convolutional Neural Networks (R-CNN)](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e) to not only detect the tumor in a Brain MRI Image but also label, localise and highlight the tumor region.

