# Introduction:
This work is insipration from Human visual system, as we see outer world thorugh our eyes, this information is passed through brain where one third of brain process this information and recognizes objects.

# Technical workflow:
For replaciting Visual system, we have used webcam that replicated human eyes to see other world and used Pretrained models and transfer learning model for brain processing. The more information is as follows:

1. Used pretrained model
  - Pretrained Model *ResNet50* is used to get prediction of 1000 categories of objects.
2. Transfer learning
  - The Final layers of *InceptionV3* is freezed and faces data is learned on that. The data is scraped from Facebook manually

### Foot Notes:
For installation opencv on python 3.5 follow this tutorial:
https://www.solarianprogrammer.com/2016/09/17/install-opencv-3-with-python-3-on-windows/


links for transfer learning

https://groups.google.com/forum/#!topic/keras-users/JDnLxlm3sHM; 
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
