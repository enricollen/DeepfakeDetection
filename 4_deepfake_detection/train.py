import collections
import os
import shutil
import warnings
import numpy as np
import torch
import torch.optim as optim
import clip
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
from images_dataset import ImagesDataset
from MLP import MLP
from LOGGER.logger import Logger

load_dotenv()

# Suppress warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
os.environ["OPENCV_LOG_LEVEL"]="SILENT"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = int(os.getenv('IMAGE_SIZE'))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
PATIENCE = int(os.getenv('PATIENCE'))
NUM_WORKERS = int(os.getenv('NUM_WORKERS'))
HIDDEN_DIMS = eval(os.getenv('HIDDEN_DIMS'))
DATA_PATH = os.getenv('DATA_PATH')
TRAIN_CSV_PATH = os.getenv('TRAIN_CSV_PATH')
VAL_CSV_PATH = os.getenv('VAL_CSV_PATH')
SAVE_PATH = os.getenv('SAVE_PATH')
SAVE_MODEL = bool(os.getenv('SAVE_MODEL'))
MODAL_MODE = int(os.getenv('MODAL_MODE'))

if MODAL_MODE == 0:
    EXPORTED_MODEL_NAME = 'best_unimodal_classifier.pth'
    LOG_DIR = 'LOGGER/Unimodal/'
elif MODAL_MODE == 1:
    EXPORTED_MODEL_NAME = 'best_multimodal_classifier.pth'
    LOG_DIR = 'LOGGER/Multimodal/'  

def empty_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"The folder {folder_path} does not exist.")

def print_statistics(classifier, train_dataset, val_dataset, train_counters, class_weights):
    #print("CLIP Architecture:")
    #print(clip_model)
    print("\nMLP Architecture:")
    print(classifier)

    len_train_dataset = len(train_dataset)
    len_validation_dataset = len(val_dataset)
    print("\nTrain images:", len_train_dataset, "Validation images:", len_validation_dataset)
    print("__TRAINING STATS__")
    
    print(train_counters)
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(val_dataset.labels)
    print(val_counters)
    print("___________________\n\n")
    

if __name__ == '__main__':

    # Call the function to empty the folder
    empty_folder("C:/Users/nello/Documents/vscode_projects/Thesis/4_deepfake_detection/LOGGER/Unimodal")
    empty_folder("C:/Users/nello/Documents/vscode_projects/Thesis/4_deepfake_detection/LOGGER/Multimodal")

    # Lazy loading for training dataset
    train_dataset = ImagesDataset(DATA_PATH, TRAIN_CSV_PATH, IMAGE_SIZE, set="train", modal_mode=MODAL_MODE) 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    #del train_dataset

    # Lazy loading for validation dataset
    val_dataset = ImagesDataset(DATA_PATH, VAL_CSV_PATH, IMAGE_SIZE, set="validation", modal_mode=MODAL_MODE) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    #del val_dataset


    # CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # MLP
    if MODAL_MODE == 0:
        input_dim = clip_model.visual.output_dim  # dimension of image features from CLIP (512)
    else:
        input_dim = clip_model.visual.output_dim*2  # dimension of image features + text features from CLIP (1024)
    hidden_dims = HIDDEN_DIMS 
    output_dim = 1  # Binary classification
    classifier = MLP(input_dim, hidden_dims, output_dim).to(device)


    # loss and optimizer
    train_counters = collections.Counter(train_dataset.labels)
    class_weights = train_counters[0] / train_counters[1]
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights])).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Print statistics
    print_statistics(classifier, train_dataset, val_dataset, train_counters, class_weights)

    # Training loop
    starting_msg = "unimodal (image-only)" if MODAL_MODE == 0 else "multimodal (image+text)" if MODAL_MODE == 1 else "unknown"
    print("Starting train phase in " + starting_msg + " mode...")

    best_val_loss = float('inf')
    prev_val_loss = float('inf')
    patience_counter = 0

    LOGGER = Logger(LOG_DIR)

    for epoch in range(NUM_EPOCHS):

        # Training
        classifier.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        classified_as_label_0_train = 0
        classified_as_label_1_train = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for batch_index, (images, captions, labels) in progress_bar:
            images = np.transpose(images, (0, 3, 1, 2))
            images = images.to(device)
            if MODAL_MODE == 1: 
                captions = torch.squeeze(captions, dim=1)  # Remove the singleton dimension
                captions = captions.to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(images).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)
                labels = labels.float().unsqueeze(1).to(device)
                if MODAL_MODE == 0:
                    features = image_features
                elif MODAL_MODE == 1:                    
                    text_features = clip_model.encode_text(captions)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    features = torch.cat((image_features, text_features), dim=1)
                else:
                    print("ERROR: unknown modal mode")
                    exit()

            optimizer.zero_grad()
            outputs = classifier(features.float())
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # training accuracy
            predicted_train = torch.round(torch.sigmoid(outputs))
            correct_train += (predicted_train == labels).sum().item()
            total_train += labels.size(0)

            train_loss += loss.item()
            progress_bar.set_postfix(train_loss=train_loss / (batch_index + 1), train_accuracy=correct_train / total_train)

            # samples classified as label 0 and label 1
            classified_as_label_0_train += torch.sum(predicted_train == 0).item()
            classified_as_label_1_train += torch.sum(predicted_train == 1).item()

            ### LOGGER ### 
            #log into Tensorboard for generating learning graphs, display batch images and show FC weights distribution
            if batch_index == 0:  # Only visualize 8 images from first batch 
                LOGGER.log_image('Batch Images', images) 

            if batch_index % 10 == 0:
                train_batch_accuracy = correct_train / total_train
                LOGGER.log_scalar('Train/Accuracy', train_batch_accuracy, epoch * len(train_loader) + batch_index)
                LOGGER.log_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + batch_index)
                LOGGER.log_histogram('fc', classifier)                        
            
        embeddings=features.cpu().numpy()
        metadata = labels.cpu().numpy()
        LOGGER.log_embeddings(embeddings=embeddings, metadata=metadata, step=epoch, tag='Train Embeddings')
        ### END LOGGER ###
                
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

        for batch_index, (images, captions, labels) in progress_bar:
            images = np.transpose(images, (0, 3, 1, 2))
            images = images.to(device)
            if MODAL_MODE == 1: 
                captions = torch.squeeze(captions, dim=1) 
                captions = captions.to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(images).float()
                labels = labels.float().unsqueeze(1).to(device)
                if MODAL_MODE == 0:
                    features = image_features
                elif MODAL_MODE == 1:                    
                    text_features = clip_model.encode_text(captions)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    features = torch.cat((image_features, text_features), dim=1)
                else:
                    print("ERROR: unknown modal mode")
                    exit()
                
                features = torch.nn.functional.normalize(features)
                outputs = classifier(features.float())
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                
                # Compute validation accuracy
                predicted_val = torch.round(torch.sigmoid(outputs))
                correct_val += (predicted_val == labels).sum().item()
                total_val += labels.size(0)
                
                # Count samples classified as label 0 and label 1
                classified_as_label_0_val += torch.sum(predicted_val == 0).item()
                classified_as_label_1_val += torch.sum(predicted_val == 1).item()

                ### LOGGER ### 

                if batch_index % 10 == 0:
                    val_batch_accuracy = correct_val / total_val
                    LOGGER.log_scalar('Validation/Accuracy', val_batch_accuracy, epoch * len(val_loader) + batch_index)
                    LOGGER.log_scalar('Validation/Loss', loss.item(), epoch * len(val_loader) + batch_index)

                ### END LOGGER ###
            
            progress_bar.set_postfix(val_loss=val_loss / (batch_index + 1), val_accuracy=correct_val / total_val)

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
                model_filename = os.path.join(SAVE_PATH, EXPORTED_MODEL_NAME)
                torch.save(classifier.state_dict(), model_filename)

        # Early stopping
        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

    LOGGER.close()