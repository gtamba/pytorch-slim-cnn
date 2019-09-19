import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd


class CelebADataset(Dataset):
    def __init__(self, image_folder='/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba', labels='/kaggle/input/celeba-dataset/list_attr_celeba.csv', validation_index='/kaggle/input/celeba-dataset/list_eval_partition.csv', transform=None, target_transform=None, split='train'):
        super(CelebADataset, self).__init__()

        validation = pd.read_csv(validation_index)
        labels = pd.read_csv(labels)
        labels = pd.concat([validation, labels], axis=1)
        labels = labels.loc[:, ~labels.columns.duplicated()]
        self.image_folder = image_folder
        self.transform = transform
        self.target_transform = target_transform

        if split == 'train':
            self.labels = labels[labels.partition == 0].drop(
                ['partition'], axis=1)
        elif split == 'validation':
            self.labels = labels[labels.partition == 1].drop(
                ['partition'], axis=1)
        elif split == 'test':
            self.labels = labels[labels.partition == 2].drop(
                ['partition'], axis=1)

    def __load_image__(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        file_name = self.labels.iloc[index, :]['image_id']
        x = self.__load_image__(Path(self.image_folder) / file_name)
        y = torch.FloatTensor(list(map(lambda x: max(x, 0), list(
            self.labels.iloc[index, :].drop(['image_id'])))))

        if self.transform is not None:
            x = self.transform(x)
        
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return self.labels.shape[0]
