from models import SlimNet
from datasets import CelebADataset
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import transforms
import argparse
import pickle
from pathlib import Path
import sys

parser = argparse.ArgumentParser(
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data and model checkpoints directories
parser.add_argument('--data_dir', type=str, default='data',
	help='Data directory containing a folder img_align_celeba with cropped images, and the csv files list_attr_celeba.csv, list_eval_partition.csv')
parser.add_argument('--save_dir', type=str, default='checkpoints',
	help='Directory to store models and optimizer states')
parser.add_argument('--save_every', type=int, default=10,
	help='Save frequency. Number of passes between checkpoints of the model.')

# Model Specifications
parser.add_argument('--conv_filters', type=int, default=96,
	help='Output filter count of the Conv2D operation in the first layer')
parser.add_argument('--conv_filter_size', type=int, default=7,
	help='Filter size of the Conv2D operation in the first layer')
parser.add_argument('--conv_stride', type=int, default=2,
	help='Stride of the Conv2D operation in the first layer')
parser.add_argument('--filter_counts', type=int, nargs='+', default=[16,32,48,64],
	help='list of constants that determine filter counts in the conv operation for each slim module')
parser.add_argument('--depth_multiplier', type=int, default=1,
	help='multiplier for depth of separable depthwise convolutions')
parser.add_argument('--num_classes', type=int, default=40,
	help='Number of classification label')

# Optimizer / Training Specifications
parser.add_argument('--batch_size', type=int, default=64,
	help='Minibatch size')
parser.add_argument('--num_epochs', type=int, default=20,
	help='Number of full passes through the training examples.')
parser.add_argument('--learning_rate', type=float, default=0.0001,
	help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
	help='L2 Regularization constant for optimizer')
parser.add_argument('--lr_decay', type=float, default=0.1,
	help='Decay constant to scale the learning rate')
parser.add_argument('--decay_lr_every', type=int, default=0,
	help="""frequency of epochs to decay learning rate by a factor of lr_decay, 
					default set to 0 for no lr scheduling as the paper does not use LR decay""")
parser.add_argument('--num_workers', type=int, default=2,
	help='number of threads to parallelize dataloading')
args = parser.parse_args()


if not Path(args.save_dir).exists():
	sys.exit(f"Checkpoint directory {args.save_dir} does not exist")

	
transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
data_dir = Path(args.data_dir)
image_folder = data_dir / "img_align_celeba"
label_csv = data_dir / 'list_attr_celeba.csv'
data_split_csv = data_dir / 'list_eval_partition.csv'

train_dataset = CelebADataset(image_folder=image_folder, labels=label_csv,
	validation_index=data_split_csv, split='train', transform=transform)

validation_dataset = CelebADataset(image_folder=image_folder, labels=label_csv, 
	validation_index=data_split_csv, split='validation', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

initial_conv = [args.conv_filters, args.conv_filter_size, args.conv_stride]

model = SlimNet(filter_count_values=args.filter_counts, initial_conv=initial_conv, num_classes=args.num_classes, depth_multiplier=args.depth_multiplier)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

scheduler = None if args.decay_lr_every == 0 else optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_lr_every, gamma=args.lr_decay)

# Push to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

epochs = args.num_epochs
training_losses = []
validation_losses = []
training_accuracies = []
validation_accuracies = []

for e in range(epochs):
    print(f'Epoch #{e+1}')
    running_loss = 0
    running_val_loss = 0
    running_train_accuracy = 0
    running_val_accuracy = 0
    total_train = 0
    total_validation = 0
    correct_train = 0
    correct_validation = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        with torch.set_grad_enabled(True):
            model.train()
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
        
        running_loss += loss.item()
        
        sigmoid_logits = torch.sigmoid(logits)
        predictions = sigmoid_logits > 0.5
        total_train += labels.size(0) * labels.size(1)
        correct_train += (labels.type(predictions.type()) == predictions).sum().item()

    else:
        if scheduler is not None:
        	scheduler.step()

        with torch.no_grad():
            model.eval()
            for batch_idx, (images,labels) in enumerate(validation_loader):
                images, labels = images.to(device), labels.to(device)
                logits = model.forward(images)
                val_loss = criterion(logits, labels)
                running_val_loss += val_loss.item()
                sigmoid_logits = torch.sigmoid(logits)
                predictions = sigmoid_logits > 0.5
                total_validation += labels.size(0) * labels.size(1)
                correct_validation += (labels.type(predictions.type()) == predictions).sum().item()
   
    training_accuracies.append(100 * correct_train / total_train)
    validation_accuracies.append(100 * correct_validation / total_validation)
    training_losses.append(running_loss/len(train_loader))
    validation_losses.append(running_val_loss/len(validation_loader))
   
    print(f"Training Loss : {running_loss/len(train_loader)}")
    print(f"Validation Loss : {running_val_loss/len(validation_loader)}")
    print(f"Training Accuracy : {100 * correct_train / total_train}")
    print(f"Validation Accuracy : {100 * correct_validation / total_validation}")
    
    if (e + 1) % args.save_every == 0:
        model.save(f"model_{e+1}.pth", optimizer, scheduler)

# Save loss trends

with open(Path(args.save_dir) / 'metrics.pkl', 'wb') as handle:
    pickle.dump((training_losses, validation_losses, training_accuracies, validation_accuracies), handle, protocol=pickle.HIGHEST_PROTOCOL)