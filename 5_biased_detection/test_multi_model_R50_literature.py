from collections import Counter
import os
import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import torch
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotenv import load_dotenv
from MLP import MLP  
from images_dataset_R50_BERT import ImagesDataset 
import torchvision.models as models
from transformers import BertModel, BertTokenizer
from sklearn.metrics import auc, accuracy_score, roc_curve
import pandas as pd
import timm

load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = int(os.getenv('IMAGE_SIZE'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
NUM_WORKERS = int(os.getenv('NUM_WORKERS'))
HIDDEN_DIMS = eval(os.getenv('HIDDEN_DIMS'))
TEST_DATA_PATH = os.getenv('TEST_DATA_PATH')
TEST_CSV_PATH = os.getenv('TEST_CSV_PATH')
MODAL_MODE = int(os.getenv('MODAL_MODE'))
BEST_MODEL_PATH = os.getenv('BEST_MODEL_PATH')
REPORT_OUTPUT_PATH = os.getenv('REPORT_OUTPUT_PATH')  # Path to save the report CSV

def load_model(best_model_path, modal_mode):
    print("Loading model...")

    if modal_mode == 0:  
        resnet50 = models.resnet50(pretrained=False)
        resnet50.fc = torch.nn.Linear(2048, 1)  
        resnet50.load_state_dict(torch.load(best_model_path))
        resnet50 = resnet50.to(device)
        resnet50.eval()  
        print("Done")
        return resnet50  
    
def load_swin(best_model_path, modal_mode):
    print("Loading model...")

    if modal_mode == 0:  
        model = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True)
        num_features = model.head.in_features  
        model.head.fc = torch.nn.Linear(num_features, 1)
        #model.load_state_dict(torch.load(best_model_path))
        model = model.to(device)
        model.eval()  
        print("Done")
        return model

def print_results(classified_images):
    df_test = pd.read_csv(TEST_CSV_PATH)  

    # counters for TP, TN, FP, FN for each category
    counts = {
        "Pristine Image + Truthful Text": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
        "Pristine Image + Fake News Text": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
        "Generated Image + Truthful Text": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
        "Generated Image + Fake News Text": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
    }

    # Define the mapping from image name to attributes
    attributes_map = {row['id']: (row['pristine_image'], row['real_text'], row['fakenews_text']) for _, row in df_test.iterrows()}

    # Process each classified image
    for image_name, true_label, predicted_label, status in classified_images:
        pristine, real_text, fake_news_text = attributes_map[image_name]
        category = None

        # Determine the category
        if pristine == 1 and real_text == 1:
            category = "Pristine Image + Truthful Text"
        elif pristine == 1 and fake_news_text == 1:
            category = "Pristine Image + Fake News Text"
        elif pristine == 0 and real_text == 1:
            category = "Generated Image + Truthful Text"
        elif pristine == 0 and fake_news_text == 1:
            category = "Generated Image + Fake News Text"

        if category:
            if true_label == predicted_label == 1:
                counts[category]["TP"] += 1
            elif true_label == predicted_label == 0:
                counts[category]["TN"] += 1
            elif true_label == 0 and predicted_label == 1:
                counts[category]["FP"] += 1
            elif true_label == 1 and predicted_label == 0:
                counts[category]["FN"] += 1

    # FPR and FNR for each category
    report_data = []
    for category, metrics in counts.items():
        TP, TN, FP, FN = metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"]
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FNR = FN / (TP + FN) if (TP + FN) > 0 else 0
        total = TP + TN + FP + FN
        misclassified = FP + FN

        report_data.append([category, total, misclassified, FPR*100, FNR*100])

    # Create DataFrame for report
    report_df = pd.DataFrame(report_data, columns=['Combination', 'Total', 'Misclassified', 'False Positive %', 'False Negative %'])
    return report_df

def save_all_roc_curves(correct_labels_list, preds_list, model_names, eers):
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')  #diagonal dashed line

    #colors = ['black', 'green', 'red', 'black', 'green', 'red']
    colors = ['red', 'green']

    # Iterate through each model's data
    for i, (correct_labels, preds, model_name, eer, color) in enumerate(zip(correct_labels_list, preds_list, model_names, eers, colors)):
        fpr, tpr, thresholds = roc_curve(correct_labels, preds)
        model_auc = auc(fpr, tpr)
        
        # line style
        if i < 3:  # base line model => dashed lines
            linestyle = '--'
        else:      # fine tuned model => solid lines
            linestyle = '-'

        # Plotting the ROC curve for the model
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {model_auc:.3f}) (EER = {eer:.4f})", color=color, linestyle=linestyle)
        
        # Finding the threshold closest to 0.5 and marking it on the plot
        closest_threshold_index = np.argmin(np.abs(thresholds - 0.5))
        #plt.plot(fpr[closest_threshold_index], tpr[closest_threshold_index], marker='o', color=color, label=f'Threshold 0.5 for {model_name}')
    
    # zoom into the interval
    #plt.xlim(0.0, 0.3)
    #plt.ylim(0.0, 1.0)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Swin Trasformer trained on generated images by stable diffusion on Competition Dataset')
    plt.legend(loc='best', fontsize='small')
    plt.savefig("combined_roc_with_threshold_colored.jpg")
    plt.clf()

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # Find the closest point to the ideal EER point (where fnr = fpr)
    distances = abs(fpr - fnr)
    min_distance_idx = np.argmin(distances)
    eer = (fpr[min_distance_idx] + fnr[min_distance_idx]) / 2
    return eer, thresholds[min_distance_idx]

def evaluate_model(model_path, modal_mode, test_csv_path):
    print(f"Evaluating model: {model_path} with modal_mode {modal_mode}")
    
    if modal_mode == 0:
        # Load only the ResNet model for unimodal 
        resnet50 = load_swin(model_path, modal_mode)
        bert_model = None
        bert_tokenizer = None
    else:
        # Load the full multimodal model (ResNet + BERT + MLP)
        resnet50, bert_model, bert_tokenizer, classifier = load_model(model_path, modal_mode)

    # Dataset and DataLoader for evaluation
    test_dataset = ImagesDataset(TEST_DATA_PATH, test_csv_path, IMAGE_SIZE, bert_tokenizer, set="test", modal_mode=modal_mode)  
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    predictions = []
    predictions_not_rounded = []
    ground_truths = []
    classified_images = []

    for batch_index, (image_names, images, input_ids, attention_masks, labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Inference"):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        with torch.no_grad():
            if modal_mode == 1 and input_ids is not None and attention_masks is not None:
                input_ids = input_ids.to(device).squeeze(1)
                attention_masks = attention_masks.to(device).squeeze(1)
                
                text_outputs = bert_model(input_ids=input_ids, attention_mask=attention_masks)
                text_features = text_outputs.pooler_output
                image_features = resnet50(images)
                features = torch.cat((image_features, text_features), dim=1)
            elif modal_mode == 0:
                # For unimodal, just use image features
                outputs = resnet50(images)
                features = outputs
            else:
                print("ERROR: unknown modal mode")
                exit()
            
            features = features.to(device)
            if modal_mode == 1:
                # Apply classifier in multimodal scenario
                outputs = classifier(features)
            
            predicted_test = torch.round(torch.sigmoid(outputs)).cpu().numpy()
            labels = labels.cpu().numpy()

            predictions.extend(predicted_test)
            predictions_not_rounded.extend(torch.sigmoid(outputs).cpu().numpy())
            ground_truths.extend(labels)

            for image_name, true_label, pred in zip(image_names, labels, predicted_test.flatten()):
                classification_status = 'Correctly Classified' if pred == true_label else 'Misclassified'
                classified_images.append((image_name if isinstance(image_name, str) else image_name.item(), true_label.item(), pred.item(), classification_status))

    # Performance Evaluation
    accuracy = accuracy_score(ground_truths, predictions)
    print(f"\nAccuracy: {accuracy:.4f}")

    report_df = print_results(classified_images)
    print(report_df)

    model_type = "unimodal" if modal_mode == 0 else "multimodal"
    test_csv_path = test_csv_path.split('/')[-1]
    if test_csv_path == "test.csv":
        test_type = "overall"
    elif test_csv_path == "real_text_rows.csv":
        test_type = "truthful_news"
    else:
        test_type = "fake_news"
    return ground_truths, predictions_not_rounded, f"resnet50_onlystablediffusion_"+model_type+"_"+test_type


if __name__ == "__main__":

    model_configs = [
    #("/home/enriconello/DeepFakeDetection/resnet50_image_only_stablediffusion_coco_88", 0, '/home/enriconello/DeepFakeDetection/test_balanced/csv/test.csv'),
    ("/home/enriconello/DeepFakeDetection/resnet50_image_only_stablediffusion_coco_88", 0, '/home/enriconello/DeepFakeDetection/test_balanced/csv/fakenews_text_rows.csv'),
    ("/home/enriconello/DeepFakeDetection/resnet50_image_only_stablediffusion_coco_88", 0, '/home/enriconello/DeepFakeDetection/test_balanced/csv/real_text_rows.csv')
    ]

    correct_labels_list = []
    preds_list = []
    model_names = []

    for model_path, modal_mode, test_csv_path in model_configs:
        ground_truths, predictions_not_rounded, model_name = evaluate_model(model_path, modal_mode, test_csv_path)
        correct_labels_list.append(ground_truths)
        preds_list.append(predictions_not_rounded)
        model_names.append(model_name)

    eers = []

    for ground_truths, predictions_not_rounded in zip(correct_labels_list, preds_list):
        eer, threshold = calculate_eer(ground_truths, predictions_not_rounded)
        eers.append(eer)

    save_all_roc_curves(correct_labels_list, preds_list, model_names, eers)