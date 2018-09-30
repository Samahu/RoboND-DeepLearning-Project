# Follow Me Project #

[image_0]: ./network-structure.jpg

In this project, I have trained a deep neural network to identify and track a target in simulation.

**Downloading the data**

I have downloaded the training and validation data locally and extracted them into the data folder.

**Downloaded the QuadSim binary**

I did download and use the QuadSim to generate and record new training data, however, for this submission I relied mainly on the provided datasets.

**Installing Dependencies**

In this part  I did clone the reposity [RoboND-Python-Starterkit](https://github.com/udacity/RoboND-Python-StarterKit), however, when I follow the instructions and create the RoboND environment I get a "parse error". This seem to be rather a new issue since no has reported it yet to the issues tracker. I did face this issue when running within a windows 10 VM and also within a completely new Windows 10 installation. I report the issue to the  https://github.com/udacity/RoboND-Python-StarterKit/issues/5.

After many trials and error I had a good workaround which is detailed in my comment in the same issues thread: https://github.com/udacity/RoboND-Python-StarterKit/issues/5#issuecomment-425746056

## Implementing the Segmentation Network ##

I did implement the required methods for the encoder and decoder blocks.

**Encoder Block**

```python
def encoder_block(input_layer, filters, strides):
    
    # DONE Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```

**Decoder Block**

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # DONE Upsample the small input layer using the bilinear_upsample() function.    
    upsampled_layer = bilinear_upsample(small_ip_layer)
    
    # DONE Concatenate the upsampled and large input layers using layers.concatenate
    concatenated_layer = layers.concatenate([upsampled_layer, large_ip_layer], axis=-1)
    
    # DONE Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(concatenated_layer, filters)
    
    return output_layer
```

**Network Structure**  
The chosen network structure is detailed as follows:

```python
def fcn_model(inputs, num_classes):

    # DONE Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    l1 = encoder_block(inputs, 16, 2)
    l2 = encoder_block(l1, 32, 2)

    # DONE Add 1x1 Convolution layer using conv2d_batchnorm().
    l3 = conv2d_batchnorm(l2, 64, kernel_size=1, strides=1)
    
    # DONE: Add the same number of Decoder Blocks as the number of Encoder Blocks
    l4 = decoder_block(l3, l1, 32)
    x = l5 = decoder_block(l4, inputs, 16)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```

The following image better illustrates the network structure:

![alt text][image_0]

Each of the two encoders is based of a Separable Convolution that would result in a reduced width and height of the input image consecutively but with an increased depth. The reduced width and height comes from having a stride size of 2. The number of filters specifies the new depth which double with every filter. The output of the second encoder is then passed to a 1x1 convolution to sum up the depthwise separable convolution step. The benefit of the separable convolution over fully connected convolution is a reduced number of parameters which allows a smaller network that has a similar performance as a fully connected one and can be trained faster.

The decoders decode the immediate previous layer (higher widthxdepth but lower depth) using a bilinear upsample operation. The output of the bilinear sampel is combined with an earlier layer and then passed to a Separable Convolution with reduced number of fitlers.

All layers have batch normalizations in between. These helps to train faster and also has a regularization effect.


## Training, Predicting and Scoring ##

These results are based on runnin the model on my local PC using merely the CPU which has 8 logical processors.

**Training my Model (CPU)**

Initially I started with a smaller network structure and only 5 epochs to verify my code then I double the number of filters for each layers and doubled the number of epochs. Current parameters are listed as follows:

*Used Parameters*

```python
learning_rate = 0.01
batch_size = 64
num_epochs = 10
steps_per_epoch = 64
validation_steps = 50
workers = 8
```

Using these parameters and for a relatively small-to-medium sized neurla network, the training took around 3 hours!!!

*Scoring*

The final_IoU score is: 0.41626866634390686 while final_score is 0.2825924314405811

Full run can be found here: [model_training_cpu](./model_training_cpu.html)

**Training my Model (GPU)**

Training on the CPU was very slow so I switched to spent some time to enable tensorflow to run on my local GPU.

*Used Parameters*

```python
learning_rate = 0.01
batch_size = 64
num_epochs = 10
steps_per_epoch = 64
validation_steps = 50
workers = 8
```

For the same set of parameters the training took less than 1 minute to complete.

*Scoring*

The final_IoU score is: 0.48423010090053176 while final_score is 0.32172734943778075 which is considerably better than the score previous score.

Full run can be found here: [model_training_gpu](./model_training_gpu.html)

**Tuning the model**
The model was doing farely well and since testing new parameters wasn't a big deal I started to experment with different sort of parameters:

1- increase the depth of the network:

```python
def fcn_model(inputs, num_classes):

    # DONE Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    l1 = encoder_block(inputs, 16, 2)
    l2 = encoder_block(l1, 32, 2)
    l3 = encoder_block(l2, 64, 2)
    
    # DONE Add 1x1 Convolution layer using conv2d_batchnorm().
    l4 = conv2d_batchnorm(l3, 128, kernel_size=1, strides=1)
    
    # DONE: Add the same number of Decoder Blocks as the number of Encoder Blocks
    l5 = decoder_block(l4, l2, 64)
    l6 = decoder_block(l5, l1, 32)
    l7 = decoder_block(l6, inputs, 16)  
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(l7)
```

[model_training_gpu_deeper](./model_training_gpu_deeper.html)

The final_IoU score is: 0.5009583480596416 and final_score is 0.3329120027316983

2- Increase epoch to 16 and steps per epoch to 128
```python
learning_rate = 0.01
batch_size = 64
num_epochs = 16
steps_per_epoch = 128
validation_steps = 50
workers = 8
```

[model_training_deeper_16_epochs_steps_128](./model_training_deeper_16_epochs_steps_128.html)

The final_IoU score is: 0.5328716191810497 and final_score is 0.39073219749257454 which is slightly below the score to pass the project.

3- Reduce learning rate (Not helpful, Reverted)
```python
learning_rate = 0.001
batch_size = 64
num_epochs = 32
steps_per_epoch = 128
validation_steps = 50
workers = 8

4- Increase num epochs to 20
```
 ```python
learning_rate = 0.01
batch_size = 64
num_epochs = 20
steps_per_epoch = 128
validation_steps = 50
workers = 8
```
The final_IoU score is: **0.5677908791302491** and final_score is **0.4104730240846068** where both values are above the passing score according to the ruberic.

[model_training_gpu_deeper_20_epochs_steps_128](./model_training_gpu_deeper_20_epochs_steps_128.html)

**Ideas for Improving your Score**

- Collect more data that contain the hero.
- Add more samples for the hero while it is distant
- Do hyper parameter search using either grid search or more advanced method for parameter optimizations