# Engineering-Notes
2020 Engineering Notes


# Theory


4. Let x be the K x 1 vector output of the last layer of a xNN and e = crossEntropy(p*, softMax(x)) be the error where p* is a K x 1 vector with a 1 in position k* representing the correct class and 0s elsewhere. Derive ∂e/∂x. Large portions of this are shown in the slides, however, the purpose of this question is for you to derive all of the parts yourself to gain more confidence with error gradients. Here’s a cookbook of steps and hints:

4.1. Derive the gradient of the cross entropy for a 1 hot label at position k*. Use the derivative rule for log (assume base e) and note that only 1 element of the gradient is non zero.

$$e = crossEntropy(p^*,softMax(x))$$

$$crossEntropy(p^*,p_x)=-\sum p^*log(p_x)$$
$$=-log(p_{x_*})$$
$$=-log(softMax(x_*))$$

$$\frac{\delta e}{\delta softMax} = [0,0,...,\frac{-1}{softMax(x_*)},...,0,0]^T$$

4.2. Derive the Jacobian of the soft max. Use the derivative quotient rule and note 2 cases: i != j and i == j (where i and j refer to the Jacobian row and col). Apply a common trick for functions with exponentials and re write the derivatives in terms of original function.

$$softMax(x) = \frac{e^{x_j}}{\sum e^{x_i}}$$

$$\frac{\delta softmax(i=j)}{\delta x} = \frac{(\sum e^{x_i}*e^{x_j}) - (e^{x_j} *e^{x_i})}{(\sum e^{x_i})^2}$$


$$=\frac{e^{x_j}}{\sum e^{x_i}}*\frac{\sum e^{x_i}-e^{x_i}}{\sum e^{x_i}}$$


$$=S_j(1-S_i)$$


<br>

$$\frac{\delta softmax(i\neq j)}{\delta x} = \frac{0 - (e^{x_j} *e^{x_i})}{(\sum e^{x_i})^2}$$

$$=-S_i*S_j$$

For any i,j in the Jacobian matrix, 
- if $i=j$ : $\frac{\delta Softmax}{\delta x} = $ Softmax(i)(1-Softmax(j)
- if $i \neq j$: $\frac{\delta Softmax}{\delta x} = $ -Softmax(i)Softmax(j)

4.3. Apply the chain rule to derive the gradient of e = crossEntropy(p*, softMax(x)) as the Jacobian matrix times the gradient vector. Take advantage of only 1 element of the gradient vector being non zero effectively selecting the corresponding col of the Jacobian matrix.


$$\frac{\delta e}{\delta x} = \frac{\delta e}{\delta softMax}\frac{\delta Softmax}{\delta x}$$

$$ = [p_0,...,p_{k^*}-1,...,p_N]$$

Through matrix-vector multiplication where we will multiply the gradient (\frac{\delta e}{\delta softMax}) across each column of the Jacobian. The result will be a vector formed from the $k^*$th row of each Jacobian column multiplied by $\frac{-1}{P_{k*}}$

4.4. Note the beautiful and numerically stable result


4.5. Remind yourself in the future when implementing classification networks in
software, use a single call to the high level library’s built in combined soft max cross entropy function if it’s available instead of making 2 calls to separate soft max and cross entropy functions. But realize that some libraries combine separate functions as an optimization step behind the scenes for you so if it’s not available then it’s probably still ok.



5. Consider a simple residual block of the form y = x + f(H x + v) where x is a K x 1 input feature vector, H is K x K linear transformation matrix, v is a K x 1 bias vector, f is a ReLU pointwise nonlinearity and y is a K x 1 output feature vector. Assume that ∂e/∂y is given. Write out a single expression using the chain rule for ∂e/∂x in terms of ∂e/∂y and the Jacobians of the other operations. For the ReLU, define the Jacobian as a K x K diagonal matrix I{0, 1}. Note the clean flow of the gradient from the output to the input, this is a key for training deep networks.


$$\frac{\delta y}{\delta x} = Identity(k,k) + I(0,1)H$$
$$\frac{\delta e}{\delta x}  = \frac{\delta e}{\delta y} \cdot (Identity(k,k) + I(0,1)H)$$


6. Write out the gradient descent update for H and v in the above example. Define intermediate feature maps as necessary. Note the need to save feature maps from the forward pass which has memory implications for xNN training.

$$\frac{\delta e}{\delta v} = \frac{\delta e}{\delta y} \cdot I(0,1)$$
$$\frac{\delta e}{\delta H} = \frac{\delta e}{\delta y}I(0,1)H$$

$$v_{t+1}=v_{t}-\alpha \frac{\delta e}{\delta v}$$
$$H_{t+1}=H_{t}-\alpha \frac{\delta e}{\delta H}$$

# Practice
import tensorflow as tf
import tensorflow_hub as hub
import os
import datetime
import pathlib
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import time
import math
from tqdm.notebook import tqdm
#tf.debugging.set_log_device_placement(True)

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
OG_IMAGENET_SIZE = (224,224)
IMAGE_SIZE = (64,64)

tf.__version__ 

## Data Extraction w MobileNet_V2

I Used the TinyImageNet dataset found [here](https://tiny-imagenet.herokuapp.com/). To extract the data into a standard format I had to restructure the train/val folders with pythonic folder restructuring. The test data was without labels, so I used MovileNet V2 to label these.

__My main reason for using Tiny Imagenet, is that the assigned dataset is currently in the process of decompressing with 32 hours remaining...__ (I had to switch the processed data to a different hard-drive because my external HDD drive couldnt handle the IO)

"""
Preprocessing for the Tiny ImageNet dataset to add labels to directories
"""
with open(os.path.join(dir_path,'words.txt'), 'r') as f:
    ids = f.readlines()
ids = list(map(lambda x: x.rstrip().split('\t'), ids))
ids = {row[0]:row[1].split(',')[0] for row in ids}

for folder in os.listdir(os.path.join(dir_path,"train")):
    path = os.path.join(dir_path,'train',folder)
    if folder in ids:
        os.rename(path, os.path.join(dir_path,'train',ids[folder]))
    else:
        print("{} not found.".format(folder))
        
for folder in os.listdir(data_root):
    if len(os.listdir(data_root/folder))==2:
        for item in os.listdir(data_root/folder):
            if item.endswith(".txt"): #remove the bounding box file
                os.remove(data_root/folder/item)
                
            if os.path.isdir(data_root/folder/item): # move the images up a directory
                for file in os.listdir(data_root/folder/item):
                    os.rename(data_root/folder/item/file, data_root/folder/file)
                os.rmdir(data_root/folder/item)
                
                
                def decode_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, OG_IMAGENET_SIZE)
    return img

filename_dataset = tf.data.Dataset.from_tensor_slices(file_names)
for x in filename_dataset.take(1):
    print(x)
    
    inference_data = filename_dataset.map(decode_image, num_parallel_calls = AUTOTUNE)
inference_data = tf.data.Dataset.zip((filename_dataset, inference_data))
inference_data = inference_data.batch(BATCH_SIZE)
inference_data = inference_data.prefetch(AUTOTUNE)


labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
def get_classes(predictions):
    return imagenet_labels[np.argmax(predictions, axis=-1)] 
    
    
    classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" 
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape = OG_IMAGENET_SIZE+(3,))
])

num_files = len(file_names)//BATCH_SIZE
for files, x in tqdm(inference_data, total=num_files):
    with tf.device('/GPU:0'):
        pred = classifier.predict(x)
    classes = get_classes(pred)
    for i, file in enumerate(files):
        f = pathlib.Path(file.numpy().decode('ascii'))
        if not os.path.exists(f.parent/classes[i]):
            os.mkdir(f.parent / classes[i])
        else:    
            pass
        os.rename(f, f.parent / classes[i] / f.parts[-1])
        
        
        ## Training Data Pipeline
        
        
        # data
DATA_NUM_CLASSES        = 200
DATA_CHANNELS           = 3
DATA_ROWS               = 64
DATA_COLS               = 64
#DATA_CROP_ROWS          = 28
#DATA_CROP_COLS          = 28

# model
MODEL_LEVEL_0_REPEATS   = 3
MODEL_LEVEL_1_REPEATS   = 3
MODEL_LEVEL_2_REPEATS   = 3

# training
TRAINING_BATCH_SIZE      = 32
TRAINING_SHUFFLE_BUFFER  = 5000
TRAINING_LR_MAX          = 0.001
# TRAINING_LR_SCALE        = 0.1
# TRAINING_LR_EPOCHS       = 2
TRAINING_LR_INIT_SCALE   = 0.01
TRAINING_LR_INIT_EPOCHS  = 5
TRAINING_LR_FINAL_SCALE  = 0.01
TRAINING_LR_FINAL_EPOCHS = 25

# training (derived)
TRAINING_NUM_EPOCHS = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
TRAINING_LR_INIT    = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL   = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE

DATA_PATH = 'F://Data/ImageNet/tiny-imagenet-200/'
SAVE_MODEL_PATH = pathlib.Path('F://Models/ImageNet_64/1/')
#!mkdir -p "$SAVE_MODEL_PATH"

AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASSES

def decode_image(img):
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.random_flip_left_right(img)
    #img = tf.image.random_crop(img, size=[DATA_CROP_ROWS, DATA_CROP_COLS, 3])
    return img

def process_path(path):
    """
    Input: file_path of a sample image
    Output: image in 3x64x64 float32 Tensor and one hot tensor
    """
    label = get_label(path)
    image = tf.io.read_file(path)
    image = decode_image(image)
    return image, label


def prepare_dataset(data_path, cache=False, shuffle_buffer_size=TRAINING_SHUFFLE_BUFFER):
    list_files = tf.data.Dataset.list_files(str(data_path/'*/*'))
    #map the above function to file_name dataset
    ds = list_files.map(process_path, num_parallel_calls=AUTOTUNE) 
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(TRAINING_BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
    
    data_root = pathlib.Path(DATA_PATH)
CLASSES = os.listdir(data_root/'train')
train_ds_cachefile = prepare_dataset(data_root/'train', 
                                          cache=str(SAVE_MODEL_PATH.parent/"cache.tfcache"), 
                                          shuffle_buffer_size=TRAINING_SHUFFLE_BUFFER)

valid_ds_cachefile = prepare_dataset(data_root/'val', 
                                          cache=str(SAVE_MODEL_PATH.parent/"valid_cache.tfcache"), 
                                          shuffle_buffer_size=TRAINING_SHUFFLE_BUFFER)

test_ds_nocache = prepare_dataset(data_root/'test', 
                                          cache=False, 
                                          shuffle_buffer_size=TRAINING_SHUFFLE_BUFFER)
                                          
                                          
