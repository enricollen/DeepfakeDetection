import collections
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm
from LazyImagesDataset import LazyImagesDataset

IMAGE_SIZE = 224
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0000001
BATCH_SIZE = 50
PATIENCE = 10
DATA_PATH = 'C:/Users/nello/Desktop/TESI/dataset_after_merging_WITH_DUPLICATES'
TRAIN_CSV_PATH = 'csv/train.csv'
VAL_CSV_PATH = 'csv/validation.csv'
SAVE_PATH = 'models_saved'
SAVE_MODEL=True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

if __name__ == '__main__':

    # Lazy loading for training dataset
    train_dataset = LazyImagesDataset(DATA_PATH, TRAIN_CSV_PATH, IMAGE_SIZE, mode="train") 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    len_train_dataset = len(train_dataset)
    #del train_dataset

    # Lazy loading for validation dataset
    val_dataset = LazyImagesDataset(DATA_PATH, VAL_CSV_PATH, IMAGE_SIZE, mode="validation") 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    len_validation_dataset = len(val_dataset)
    #del val_dataset


    # CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # MLP
    input_dim = clip_model.visual.output_dim  # Dimension of image features from CLIP
    hidden_dims = [256, 128] #hidden_dim = 512
    output_dim = 1  # Binary classification
    classifier = MLP(input_dim, hidden_dims, output_dim).to(device)

    # Print statistics
    #print("CLIP Architecture:")
    #print(clip_model)
    print("\nMLP Architecture:")
    print(classifier)
    print("\nTrain images:", len_train_dataset, "Validation images:", len_validation_dataset)
    print("__TRAINING STATS__")
    train_counters = collections.Counter(train_dataset.labels)
    print(train_counters)

    class_weights = train_counters[0] / train_counters[1]
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(val_dataset.labels)
    print(val_counters)
    print("___________________\n\n")


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

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if SAVE_MODEL:
                if not os.path.exists(SAVE_PATH):
                    os.makedirs(SAVE_PATH)
                model_filename = os.path.join(SAVE_PATH, 'best_classifier.pth')
                torch.save(classifier.state_dict(), model_filename)

        # Early stopping
        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break