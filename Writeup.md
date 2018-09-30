
## Follow Me Project ##

In this project, I have trained a deep neural network to identify and track a target in simulation.

**Downloading the data**

I have downloaded the training and validation data locally and extracted them into the data folder.

**Downloaded the QuadSim binary**

I did download and use the QuadSim to generate and record new training data, however, for this submission I relied mainly on the provided datasets.

**Installing Dependencies**

In this part I many issues during the install. I did clone the reposity [RoboND-Python-Starterkit](https://github.com/udacity/RoboND-Python-StarterKit), however, when I follow the instructions and create the RoboND environment I get a "parse error". This seem to be rather a new issue since no has reported it yet to the issues tracker. I did face this issue when running within a windows 10 VM and also within a completely new Windows 10 installation. I report the issue to the  https://github.com/udacity/RoboND-Python-StarterKit/issues/5.

I had two successful workarounds, one is to copy the environment from an older working installation. The other one (after many trials and error) is to run "conda env list" 

## Implementing the Segmentation Network

I did implement the required methods for the encoder and decoder blocks.

# Encoder Block

```python
def encoder_block(input_layer, filters, strides):
    
    # DONE Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```

# Decoder Block

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

# Network Structure
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

<<< Explaining Layers: >>>


## Training, Predicting and Scoring ##

These results are based on runnin the model on my local PC using merely the CPU which has 8 logical processors.

### Training my Model ###

Initially I started with a smaller network structure and only 5 epochs to verify my code then I double the number of filters for each layers and doubled the number of epochs. Current parameters are listed as follows:

# Used Parameters:

```python
learning_rate = 0.01
batch_size = 64
num_epochs = 10
steps_per_epoch = 64
validation_steps = 50
workers = 8
```

## Scoring ##

The final_IoU score is: 0.41626866634390686 while final_score is 0.2825924314405811

**Ideas for Improving your Score**

- Collect more data that contain the hero.
- Using GPU to allow me increase network size and increase epochs and steps_per_epochs to get better results.