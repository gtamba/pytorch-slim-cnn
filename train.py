from models import SlimCNN
from datasets import CelebADataset
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import transforms

transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
#Uses default paths of Dataset in a Kaggle Kernel
train_dataset = CelebADataset(image_folder="/Users/tamba/Desktop/celb/img_align_celeba", labels='/Users/tamba/Desktop/celb/list_attr_celeba.csv', validation_index='/Users/tamba/Desktop/celb/list_eval_partition.csv', split='train', transform=transform)
validation_dataset = CelebADataset(image_folder="/Users/tamba/Desktop/celb/img_align_celeba", labels='/Users/tamba/Desktop/celb/list_attr_celeba.csv', validation_index='/Users/tamba/Desktop/celb/list_eval_partition.csv',split='validation', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=2)


model = SlimCNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Push to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

save_every = 1
epochs = 1
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
        #print(f'Training Batch no {batch_idx}')
        with torch.set_grad_enabled(True):
            model.train()
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        #print(logits.shape)
        #_, predicted = logits.topk(1, dim=1)
        
        sigmoid_logits = torch.sigmoid(logits)
        predictions = sigmoid_logits > 0.5
        total_train += labels.size(0) * labels.size(1)
        correct_train += (labels.type(predictions.type()) == predictions).sum().item()



    else:
        #scheduler.step()

        with torch.no_grad():
            model.eval()
            for batch_idx, (images,labels) in enumerate(validation_loader):
                images, labels = images.to(device), labels.to(device)
                logits = model.forward(images)
                val_loss = criterion(logits, labels)
                running_val_loss += val_loss.item()
                #_, predicted = logits.topk(1, dim=1)
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
    
    if (e + 1) % save_every == 0:
        model.save(f"model_{e+1}.pth", optimizer, scheduler)

# Save loss trends

import pickle
with open('losses.pickle', 'wb') as handle:
    pickle.dump((training_losses, validation_losses, training_accuracies, validation_accuracies), handle, protocol=pickle.HIGHEST_PROTOCOL)