import collections
import os
import random
import shutil
import warnings
import numpy as np
import torch
import torch.optim as optim
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
from images_dataset_R50_BERT import ImagesDataset
from MLP import MLP
import torchvision.models as models
from transformers import BertModel, BertTokenizer
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
TRAIN_DATA_PATH = os.getenv('TRAIN_DATA_PATH')
VALIDATION_DATA_PATH = os.getenv('VALIDATION_DATA_PATH')
TRAIN_CSV_PATH = os.getenv('TRAIN_CSV_PATH')
VAL_CSV_PATH = os.getenv('VAL_CSV_PATH')
SAVE_PATH = os.getenv('SAVE_PATH')
SAVE_MODEL = bool(os.getenv('SAVE_MODEL'))
MODAL_MODE = int(os.getenv('MODAL_MODE'))
HYPERPARAMETERS_DUMP_PATH = os.getenv('LOG_DIR')
LOG_DIR = os.getenv('LOG_DIR')

if MODAL_MODE == 0:
    EXPORTED_MODEL_NAME = 'best_unimodal_classifier.pth'
elif MODAL_MODE == 1:
    EXPORTED_MODEL_NAME = 'best_multimodal_classifier.pth'

def hyperparameters_dump(len_train, len_validation):
    if not os.path.exists(HYPERPARAMETERS_DUMP_PATH):
        os.makedirs(HYPERPARAMETERS_DUMP_PATH)
        
    with open(HYPERPARAMETERS_DUMP_PATH+"/params.txt", "w") as f:
        f.write("NUM_EPOCHS=" + str(NUM_EPOCHS) + "\n")
        f.write("LEARNING_RATE=" + str(LEARNING_RATE) + "\n")
        f.write("WEIGHT_DECAY=" + str(WEIGHT_DECAY) + "\n")
        f.write("BATCH_SIZE=" + str(BATCH_SIZE) + "\n")
        f.write("PATIENCE=" + str(PATIENCE) + "\n")
        f.write("NUM_WORKERS=" + str(NUM_WORKERS) + "\n")
        f.write("HIDDEN_DIMS=" + str(HIDDEN_DIMS) + "\n")
        f.write("----------------------------------\n")
        f.write("Train samples = " + len_train + "\n")
        f.write("Validation samples = " + len_validation + "\n")

def print_statistics(classifier, train_dataset, val_dataset, train_counters):
    print("\nMLP Architecture:")
    print(classifier)

    len_train_dataset = len(train_dataset)
    len_validation_dataset = len(val_dataset)
    print("\nTrain images:", len_train_dataset, "Validation images:", len_validation_dataset)
    print("__TRAINING STATS__")
    
    print(train_counters)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(val_dataset.labels)
    print(val_counters)
    print("___________________\n\n")

if __name__ == '__main__':

    torch.backends.cudnn.deterministic = True
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    resnet50 = models.resnet50(pretrained=True)
    # Remove the final layer to get the features instead of predictions
    resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
    resnet50 = resnet50.to(device)

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = bert_model.to(device)

    train_dataset = ImagesDataset(TRAIN_DATA_PATH, TRAIN_CSV_PATH, IMAGE_SIZE, bert_tokenizer, set="train", modal_mode=MODAL_MODE) 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS) 

    val_dataset = ImagesDataset(VALIDATION_DATA_PATH, VAL_CSV_PATH, IMAGE_SIZE, bert_tokenizer, set="validation", modal_mode=MODAL_MODE) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Hyperparameters dump
    hyperparameters_dump(str(len(train_dataset)), str(len(val_dataset)))

    # MLP
    if MODAL_MODE == 0:
        input_dim = 2048  # Output dimension from ResNet50
    else:
        input_dim = 2048 + 768  # Sum of output dimensions from ResNet50 and BERT
    hidden_dims = HIDDEN_DIMS 
    output_dim = 1  # Binary classification
    classifier = MLP(input_dim, hidden_dims, output_dim).to(device)


    # loss and optimizer
    train_counters = collections.Counter(train_dataset.labels)
    loss_fn = torch.nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Print statistics
    print_statistics(classifier, train_dataset, val_dataset, train_counters)

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

        for batch_index, (images, input_ids, attention_masks, labels) in progress_bar:
            
            images = images.to(device)
            labels = labels.float().unsqueeze(1)

            with torch.no_grad():
                if MODAL_MODE == 1 and input_ids is not None and attention_masks is not None:
                    input_ids = input_ids.to(device).squeeze(1)
                    attention_masks = attention_masks.to(device).squeeze(1)
                    
                    text_outputs = bert_model(input_ids=input_ids, attention_mask=attention_masks)
                    input_ids = input_ids.cpu()
                    attention_masks = attention_masks.cpu()
                    text_features = text_outputs.pooler_output
                    text_features = text_features.cpu()
                    #text_features /= text_features.norm(dim=-1, keepdim=True)
                        
                image_features = resnet50(images)
                image_features = image_features.cpu()
                #image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = torch.squeeze(image_features)

                if MODAL_MODE == 0:
                    features = image_features
                elif MODAL_MODE == 1:
                    features = torch.cat((image_features, text_features), dim=1)
                    features = features.float()
                else:
                    print("ERROR: unknown modal mode")
                    exit()

            features = features.to(device)
            outputs = classifier(features.float())
            outputs = outputs.cpu()
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
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
            if batch_index % 50 == 0:
                train_batch_accuracy = correct_train / total_train
                LOGGER.log_scalar('Train/Accuracy', train_batch_accuracy, epoch * len(train_loader) + batch_index)
                LOGGER.log_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + batch_index)
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

        for batch_index, (images, input_ids, attention_masks, labels) in progress_bar:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            with torch.no_grad():
                if MODAL_MODE == 1 and input_ids is not None and attention_masks is not None:
                    input_ids = input_ids.to(device).squeeze(1)
                    attention_masks = attention_masks.to(device).squeeze(1)
                    
                    text_outputs = bert_model(input_ids=input_ids, attention_mask=attention_masks)
                    input_ids = input_ids.cpu()
                    attention_masks = attention_masks.cpu()
                    text_features = text_outputs.pooler_output
                    text_features = text_features.cpu()
                    #text_features /= text_features.norm(dim=-1, keepdim=True)

                
                image_features = resnet50(images)
                image_features = image_features.cpu()
                #image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = torch.squeeze(image_features)

                if MODAL_MODE == 0:
                    features = image_features
                elif MODAL_MODE == 1:
                    features = torch.cat((image_features, text_features), dim=1)
                    features = features.float()
                else:
                    print("ERROR: unknown modal mode")
                    exit()
                
                features = features.float().to(device)
                outputs = classifier(features)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                
                # validation accuracy
                predicted_val = torch.round(torch.sigmoid(outputs))
                correct_val += (predicted_val == labels).sum().item()
                total_val += labels.size(0)

                val_loss += loss.item()
                progress_bar.set_postfix(val_loss=val_loss / (batch_index + 1), val_accuracy=correct_val / total_val)
                
                # samples classified as label 0 and label 1
                classified_as_label_0_val += torch.sum(predicted_val == 0).item()
                classified_as_label_1_val += torch.sum(predicted_val == 1).item()

                ### LOGGER ### 
                if batch_index % 50 == 0:
                    val_batch_accuracy = correct_val / total_val
                    LOGGER.log_scalar('Validation/Accuracy', val_batch_accuracy, epoch * len(val_loader) + batch_index)
                    LOGGER.log_scalar('Validation/Loss', loss.item(), epoch * len(val_loader) + batch_index)
                ### END LOGGER ###

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
                print("New best model saved")

        # Early stopping
        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break
    
    LOGGER.close()