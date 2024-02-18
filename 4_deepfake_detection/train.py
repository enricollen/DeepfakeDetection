import collections
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import clip
from torch.utils.data import DataLoader
from albumentations import Compose, PadIfNeeded
from tqdm import tqdm
from images_dataset import ImagesDataset
from utils import center_crop
from images_dataset import ImagesDataset
from transforms.albu import IsotropicResize

IMAGE_SIZE = 224
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0000001
BATCH_SIZE = 32
PATIENCE = 10
TRAIN_DATA_PATH = 'train_small/'
VAL_DATA_PATH = 'val_small/'
SAVE_MODEL=False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x    


def read_images(data_path, transform):
    dataset = []
    for image_name in os.listdir(data_path):
        if not image_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        label = 1 if image_name.startswith("SD") else 0

        image_path = os.path.join(data_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            continue

        image = center_crop(image)
        image = transform(image=image)['image']

        row = (image, label)
        dataset.append(row)
    return dataset

def create_pre_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT)
    ])


transform = create_pre_transform(IMAGE_SIZE)
train_dataset = read_images(TRAIN_DATA_PATH, transform)
validation_dataset = read_images(VAL_DATA_PATH, transform)

train_labels = [float(row[1]) for row in train_dataset]
train_dataset = [row[0] for row in train_dataset]
validation_labels = [float(row[1]) for row in validation_dataset]
validation_dataset = [row[0] for row in validation_dataset]

# train and validation datasets and DataLoaders
train_dataset = ImagesDataset(np.array(train_dataset), np.array(train_labels), IMAGE_SIZE, mode='train')
val_dataset = ImagesDataset(np.array(validation_dataset), np.array(validation_labels), IMAGE_SIZE, mode='val')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# MLP
input_dim = clip_model.visual.output_dim  # Dimension of image features from CLIP
hidden_dim = 512
output_dim = 1  # Binary classification
classifier = MLP(input_dim, hidden_dim, output_dim).to(device)

# Print some useful statistics
print("Train images:", len(train_dataset), "Validation images:", len(validation_dataset))
print("__TRAINING STATS__")
train_counters = collections.Counter(train_labels)
print(train_counters)

class_weights = train_counters[0] / train_counters[1]
print("Weights", class_weights)

print("__VALIDATION STATS__")
val_counters = collections.Counter(validation_labels)
print(val_counters)
print("___________________")

# loss, and optimizer
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights])).to(device)
optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Training loop
best_val_loss = float('inf')
prev_val_loss = float('inf')
patience_counter = 0
for epoch in range(NUM_EPOCHS):

    # Training
    classifier.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    classified_as_label_0_train = 0
    classified_as_label_1_train = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for index, (images, labels) in progress_bar: #captions
        images = np.transpose(images, (0, 3, 1, 2))
        images = images.to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(images).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            labels = labels.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = classifier(image_features)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # training accuracy
        predicted_train = torch.round(torch.sigmoid(outputs))
        correct_train += (predicted_train == labels).sum().item()
        total_train += labels.size(0)

        train_loss += loss.item()
        progress_bar.set_postfix(train_loss=train_loss / (index + 1), train_accuracy=correct_train / total_train)

        # samples classified as label 0 and label 1
        classified_as_label_0_train += torch.sum(predicted_train == 0).item()
        classified_as_label_1_train += torch.sum(predicted_train == 1).item()

    train_loss /= len(train_loader)
    train_accuracy = correct_train / total_train
    
    print("label 0 (Train):", classified_as_label_0_train)
    print("label 1 (Train):", classified_as_label_1_train)
    
    # Validation
    classifier.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    classified_as_label_0_val = 0
    classified_as_label_1_val = 0
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Validation)")

    for index, (images, labels) in progress_bar:
        images = np.transpose(images, (0, 3, 1, 2))
        images = images.to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(images).float()
            labels = labels.float().unsqueeze(1).to(device)
            
            outputs = classifier(image_features)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            
            # Compute validation accuracy
            predicted_val = torch.round(torch.sigmoid(outputs))
            correct_val += (predicted_val == labels).sum().item()
            total_val += labels.size(0)
            
            # Count samples classified as label 0 and label 1
            classified_as_label_0_val += torch.sum(predicted_val == 0).item()
            classified_as_label_1_val += torch.sum(predicted_val == 1).item()
        
        progress_bar.set_postfix(val_loss=val_loss / (index + 1), val_accuracy=correct_val / total_val)

    val_loss /= len(val_loader)
    val_accuracy = correct_val / total_val

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    print("label 0 (Val):", classified_as_label_0_val)
    print("label 1 (Val):", classified_as_label_1_val)
    
    # Check if validation loss increased with respect to the previous validation loss
    if val_loss > prev_val_loss:
        patience_counter += 1
        print("Validation loss increased [" + str(patience_counter) + "/" + str(PATIENCE) + "]")
    else:
        patience_counter = 0  # Reset patience counter if validation loss improved
    
    prev_val_loss = val_loss 

    # Early stopping
    if patience_counter >= PATIENCE:
        print("Early stopping.")
        break

# Save the model if needed
if SAVE_MODEL:
    torch.save(classifier.state_dict(), 'classifier.pth')