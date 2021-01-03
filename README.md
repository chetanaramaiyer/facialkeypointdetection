# facialkeypointdetection
In this project, I automatically detected the points to pick from the shape of a face using Neural Networks. By inputing an image containing a face, my model should detect key points on the face automatically.  I trained my Neural Network model using training images with certain key points that were already picked. The model performance gradually improves (mean square error decreases) when I compute more rounds on the training data. Then, I test the model using validation images with ground truth face shapes also given.  The overall process was as follows:  1. Created training data datasets and validation data datasets.  2. Used dataloader to go through all images and coresponding selected points.  3. Used CNN model, which had convolution layers, max pool layers and RELU layers.  4. Trained CNN model on training data by minimizing MSE loss.  5. See result on validation data and evaluate training and validation accuracy.

Overview:

In this project, I automatically detected the points to pick from the shape of a face using Neural Networks. By inputing an image containing a face, my model should detect key points on the face automatically.

I trained my Neural Network model using training images with certain key points that were already picked. The model performance gradually improves (mean square error decreases) when I compute more rounds on the training data. Then, I test the model using validation images with ground truth face shapes also given.

The overall process was as follows:

1. Created training data datasets and validation data datasets.

2. Used dataloader to go through all images and coresponding selected points.

3. Used CNN model, which had convolution layers, max pool layers and RELU layers.

4. Trained CNN model on training data by minimizing MSE loss.

5. See result on validation data and evaluate training and validation accuracy.


Nose Tip Detection:
The Dataset:
I used the IMM Face Database for this project, which contains 240 facial images of 40 people with 6 images from different viewpoints for everyone. For every image, 58 facial key points are already selected. To start with, we only consider the key point 53 which is the point that is referring to the nose. I normalized the inputs to -0.5 to 0.5 and resized them to a fixed resolution.

Using the Dataloader:


I first loaded 192 images (first 32 faces and the 6 different angles for each face) and their corresponding selected points into training dataloader. Then, I loaded last 48 images (the last 8 faces and the 6 different angles for each face) and their corresponding selected points into validation dataloader.

Using the CNN Model:
I created and used a CNN model with input size 80*60, output size 2 and conbination of layers in the middle: conv2 with kernal size 2*2, RELU, maxPool with window 2*2.

After, I trained the model on training data using 50 epoches. Through the graph below, I can see the MSE loss of the model on training dataset and validation dataset are decreasing overall. Thiss means our model is improved slowly as we continue to train it more with more epoches.

Model Architecture:
Net_((conv1): Conv2d(1, 12, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))

(conv2): Conv2d(12, 18, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))

(conv3): Conv2d(18, 28, kernel_size=(5, 5), stride=(1, 1))

(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

(fcl1): Linear(in_features=980, out_features=84, bias=True)

(fcl2): Linear(in_features=84, out_features=2, bias=True))


Result:
After I trained the model, I applied the model onto the testing data. Below, I show examples of how well the simple model does on predicting the nose points. Green point indicates the actually selected nose point, and the red point indicates the predicted noose point.


Full Facial Points Detection:


The last step after warping the images is to create the midway face image by simply averaging the two warped images. All this requires is taking the average of the pixels in the two warped images. Here is the result:

The Dataset:
I used the same dataset as the previous part. Now, instead of predicting just the nose point, we predict all 58 keypoints using the CNN model!

Using the Dataloader:


In this step, I resized the images in the datasets to size 240*180 instead of 80*60 to get more training information. Then, I created an augmentation class that further changed the images by rotating them. This created even more training data. Since we have relatively small dataset (only 240 images), we try to increase our dataset size by augmenting the original images to create new images.

Using the CNN:


I created a new CNN model that has input size 240*180, output size 116 (coordinates of 58 key points). Now, there are even more layers in the middle, which includess conv2 with kernal size 2*2, RELU, maxPool with window 2*2.

Below I've visualized the convolution kernels. Essentially, these are the weights of the first convolutional layer in this simple facial detection neural network.

Model Architecture:
FullFacialNet((conv1): Conv2d(1, 12, kernel_size=(5, 5), stride=(1, 1))

(conv2): Conv2d(12, 24, kernel_size=(3, 3), stride=(1, 1))

(conv3): Conv2d(24, 32, kernel_size=(3, 3), stride=(1, 1))

(conv4): Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1))

(conv5): Conv2d(48, 64, kernel_size=(1, 1), stride=(1, 1))

(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

(fcl1): Linear(in_features=134784, out_features=24, bias=True)

(fcl2): Linear(in_features=24, out_features=116, bias=True))


Lastly, I trained the model on training data with 20 epoches. As you can see below in the graph, during the training process the MSE loss of the model on training dataset and validation dataset are decreasing. This means our model is gradually improved with more training epoches.


Train With Larger Dataset:
The Dataset:
In part 3, we simply used the same model on a much larger dataset with much larger images. I used the staff-provided code to load the image filenames, bounding box coordinates, and keypoints. For each image, I cropped it, rescaled it to 224x224, changed it to grayscale and changed it to tensor objects (all of which were Transform classes). Since the images were so large, we wanted to crop them to only contain the faces. I created a RandomCrop transform class to be applied to every image so that it’s cropped by the given bounding box. I created a Dataset with all the training data and randomly split it to be 80% training data and 20% validation data. Then, I constructed training and validation Dataloaders with these training and validation datasets. I did a similar thing with the test_dataset by uploading the test xml file and parsed through to return the filenames and bounding box coordinates from that test xml file. Then, I used that information to create a Testing Dataset (one that doesn’t contain landmarks). Using the training dataset and data loader, and the validation dataset and validation data loader, I ran my model on the training and validation datasets. This took a really long time. Below is my learning curve plotting the train and validation loss for each of the 10 training epochs:

Then, I used the best_model from the previous training to select landmarks from the test_images that I previously parsed and read in. I do this by creating a TestDataLoader from my testdataset and iterating through that to get all the images and bounding boxes. I run my model on the images to get the predicted landmarks. Then, I displayed my selected points on different faces. Here are the ressults:

I saved the selected points in a csv file, which I then submitted to Kaggle. My Kaggle score was 345.33933, which was pretty low because the first few images are very off in the testing dataset. The predicted points of the first few images don't match up with the faces at all. However, the later images are much more accurate.


Testing on my own images:


This was my favorite part. I ran my model on my own personal images. I didn't create custom bounding boxes for these images, which is why it is not that accurate. However, if I was able to create custom bounding boxes, it would fit to the shape of their faces much better.

