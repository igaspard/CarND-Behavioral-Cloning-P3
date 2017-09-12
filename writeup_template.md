#**Behavioral Cloning by Gaspard Shen**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-arch.png "Model Diagram"
[image2]: ./examples/center.jpg "Center Image"
[image3]: ./examples/Recovery0.jpg "Recovery Image"
[image4]: ./examples/Recovery1.jpg "Recovery Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I use the NVIDIA Architecture  As compare to the initial model based on the LeNet, I can observe obvious improve at both the mean squared error, validation error and the simulator behavior.

The model includes RELU layers to introduce nonlinearity (code line XX), and the data is normalized in the model using a Keras lambda layer (code line XX). And then crop the image upper 75 pixels, and bottom of 25 pixels to remove the information out of the road to achieve better training results.

####2. Attempts to reduce overfitting in the model

When the mean square error and validation error can't decrease after several epochs. I will decrease the epochs.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line XX).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road and recover from the side to the center of the road.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to try the course suggestion and try and error.

My first step was to use a convolution neural network model similar to the LeNet and start with one lap training data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by using the NVIDIA Architecture.
The simulator behavior was better, i finally can pass the first corner and then driver into the water...

Then I decide to capture more data. Here is my strategy.
1. Correct 3 laps of the center lane driving.
2. 1 laps of the Counter-Clockwise lap to collect more right turn data since the original lap has much more left turns.
3. Since the previous model look like can't handle the sharp turn well and can't recover the
And after training this data, the result looks great at the simulator.

Even further, I use the data augmentation to doubling the training sample by flipping the image.

Let's check it out the simulator to see how well the car was driving around track one.
Now, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 38-50) consisted of a convolution neural network with the as below diagram that consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers.

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

Then I recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover once the vehicle left the center of the road. These images show what a recovery looks like starting from :

![alt text][image3]
![alt text][image4]

Then I drive the vehicle this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would increasing the training set to avoid overfitting.

After the collection process, I had 39884 number of data points. I then preprocessed this data by normalized and cropping the upper 75 pixels and bottom of 25 pixels.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced as below.

31907/31907 [==============================] - 74s - loss: 0.0313 - val_loss: 0.0226
Epoch 2/10
31907/31907 [==============================] - 73s - loss: 0.0290 - val_loss: 0.0236
Epoch 3/10
31907/31907 [==============================] - 73s - loss: 0.0284 - val_loss: 0.0233
Epoch 4/10
31907/31907 [==============================] - 73s - loss: 0.0278 - val_loss: 0.0256
Epoch 5/10
31907/31907 [==============================] - 73s - loss: 0.0271 - val_loss: 0.0237
Epoch 6/10
31907/31907 [==============================] - 73s - loss: 0.0264 - val_loss: 0.0246
Epoch 7/10
31907/31907 [==============================] - 73s - loss: 0.0254 - val_loss: 0.0241
Epoch 8/10
31907/31907 [==============================] - 73s - loss: 0.0244 - val_loss: 0.0272
Epoch 9/10
31907/31907 [==============================] - 73s - loss: 0.0233 - val_loss: 0.0276
Epoch 10/10
31907/31907 [==============================] - 73s - loss: 0.0219 - val_loss: 0.0267
