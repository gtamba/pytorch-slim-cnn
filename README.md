This repository has a PyTorch implementation of SlimNet as described in the paper:

> Slim-CNN: A Light-Weight CNN for Face Attribute Prediction
> Ankit Sharma, Hassan Foroosh
> [Paper](https://arxiv.org/abs/1907.02157)

- Training
  - [Requirements](#requirements)
  - [Dataset](#dataset)
  - [Script Parameters](#input-and-output-options)
 
- [Examples](#examples)
  - [Train](#setup)
  - [Sample Inference Code](#sample-out-of-the-box-inference-code)
- [Benchmarks](#benchmarks)
  - [Model Footprint/Runtime](#benchmarks)
  - [Notebooks](#notebooks)


## Requirements
Python 3.6.x
```
torch==1.2.0
torchvision==0.4.0
```
## Dataset

The CelebA Facial Recognition Dataset is available [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), the default train arguments
look for the dataset in the following format in the current working directory:   

```
data/list_attr_celeba.csv
data/list_eval_partition.csv
data/img_align_celeba/00001.jpg
data/img_align_celeba/00002.jpg
...
```

Alternatively, the argument you pass as --data_dir in the train script should be the path to (relative or absolute) the directory containing 2 CSV files and folder of cropped images.

## Input and output options
```
  --data_dir   STR    Training data folder.      Default is data`.
  --save_dir   STR    Model checkpoints folder.  Default is `checkpoints`.
```
## Model options
```
  --save_every                  INT     Save frequency                                    Default is 5
  --num_epochs                  INT     Number of epochs.                                 Default is 20.
  --batch_size                  INT     Number of images per batch.                       Default is 64.
  --conv_filters                INT     Number of initial conv filters                    Default is 20.
  --conv_filter_size            INT     Initial conv filter size.                         Default is 7.
  --conv_filter_stride          INT     Initial conv filter stride.                       Default is 2.  
  --filter_counts               INT     List of Filter counts for the Slim modules        Default is 16 32 48 64.
  --depth_multiplier            INT     Depth width for separable depthwise convolution   Default is 1.
  --num_classes                 INT     Number of class labels        .                   Default is 40.  
  --num_workers                 INT     Number of threads for dataloading                 Default is 2.
  --weight_decay                FLOAT   Weight decay of Adam.                             Default is 0.0001.
  --learning_rate               FLOAT   Adam learning rate.                               Default is 0.0001.
  --decay_lr_every              FLOAT   Frequency to decay learning rate                  Default is 0
  --lr_decay                    FLOAT   Factor to decay learning rate by                  Default is 0.1.

```

## Examples

#### Setup
```
git clone https://github.com/gtamba/pytorch-slim-cnn & cd pytorch-slim-cnn
pip install -r requirements.txt
mkdir checkpoints
```

#### Train a model with the CelebA dataset in the data/ folder
```
python train.py --num_epochs 6 --save_every 2 --batch_size 256
```

#### Sample out of the box inference code
```python
import torch
from slimnet import SlimNet
from torchvision import transforms
from PIL import Image
import numpy as np

PATH_TO_IMAGE = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/000001.jpg'
labels = np.array(['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
       'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
       'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
       'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
       'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
       'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
       'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
       'Wearing_Necklace', 'Wearing_Necktie', 'Young'])

# GPU isn't necessary but could definitly speed up, swap the comments to use best hardware available
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'

transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

# Make tensor and normalize, add pseudo batch dimension and move to configured device
with open(PATH_TO_IMAGE, 'rb') as f:
    x = transform(Image.open(f)).unsqueeze(0).to(device)

model = SlimNet.load_pretrained('models/celeba_20.pth').to(device)

with torch.no_grad():
            model.eval()
            logits = model(x)
            sigmoid_logits = torch.sigmoid(logits)
            predictions = (sigmoid_logits > 0.5).squeeze().numpy()

print(labels[predictions.astype(bool)])
```

## Benchmarks 

##### Model Footprint
#
Model Size ~ 7 mb
Number of parameters ~ 0.6 M 


##### Simple `timeit` benchmarks (10000 loops)

- CPU : 0.1598 seconds per image ~ 6.25 frames per second
- GPU : 0.0753 seconds per image ~ 13.3 frames per second


## Training Metrics

todo, see notebook for now for training metric trends

## Notebooks

[Notebook to evaluate model on test set as well as plot training metrics](https://github.com/gtamba/pytorch-slim-cnn/blob/master/notebooks/evaluate.ipynb)


### Notes

-  No data augmentation was used although it would definitely help in performance/robustness
-  It is unclear if weight_decay on the Optimizer translates well as L2 regularization to the network weights as opposed to manually adding them to the loss
-  Number of epochs and batch size were not specified in the paper, tried 20 epochs with batch size 256 which is conservative at best, but the loss trend shows an overfitting inflection around the 20th epoch mark as can be seen in the notebook
-  torchvision now provides an out of the box Dataset API for the CelebA dataset so you don't need a custom one like mine

