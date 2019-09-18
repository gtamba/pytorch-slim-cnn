This repository has a PyTorch implementation of SlimNet as described in the paper:

> Slim-CNN: A Light-Weight CNN for Face Attribute Prediction
> Ankit Sharma, Hassan Foroosh
> [[Paper]](https://arxiv.org/abs/1907.02157)


### Requirements
Python 3.6
```
torch==1.2.0
torchvision==0.4.0a0+6b959ee
```
### Dataset

The CelebA Facial Recognition Dataset is available [[here]](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), the default train arguments
look for the dataset in the following format in the current working directory:   

```
data/list_attr_celeba.csv
data/list_eval_partition.csv
data/img_align_celeba/00001.jpg
data/img_align_celeba/00002.jpg
...
```

Alternatively, the argument you pass as --data_dir in the train script should be the path to (relative or absolute) the directory containing 2 CSV files and folder of cropped images.

#### Input and output options
```
  --data_dir   STR    Training data folder.      Default is `data`.
  --save_dir   STR    Model checkpoints folder.  Default is `checkpoints`.
```
#### Model options
```
  --save_every                  INT     Save frequency                                    Default is 5
  --num_epochs                  INT     Number of epochs.                                 Default is 20.
  --batch_size                  INT     Number of images per batch.                       Default is 64.
  --conv_filters                INT     Number of initial conv filters                    Default is 20.
  --conv_filter_size            INT     Initial conv filter size.                         Default is 7.
  --conv_filter_stride          INT     Initial conv filter stride.                       Default is 2.  
  --filter_counts               INT     List of Filter counts for the Slim modules        Default is 16 32 48 64.
  --depth_multiplier            INT     Depth level for separable depthwise convolution   Default is 1.
  --num_classes                 INT     Number of class labels        .                   Default is 40.  
  --num_workers                 INT     Number of threads for dataloading                 Default is 2.
  --weight_decay                FLOAT   Weight decay of Adam.                             Default is 0.0001.
  --learning_rate               FLOAT   Adam learning rate.                               Default is 0.0001.
  --decay_lr_every              FLOAT   Frequency to decay learning rate                  Default is 0
  --lr_decay                    FLOAT   Factor to decay learning rate by                  Default is 0.1.

```

### Examples
```
git clone https://github.com/gtamba/pytorch-slim-cnn & cd pytorch-slim-cnn
```

(Optional/Recommended: Setup your desired virtual environment using virtualenv or conda)
```
pip install -r requirements.txt
mkdir checkpoints
```
```
python train.py --num_epochs 6 --save_every 2 --batch_size 256
```

### TODO

##### Inference / Evaluation

##### Pretrained model

##### Notebooks

