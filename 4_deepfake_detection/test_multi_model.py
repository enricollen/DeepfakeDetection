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
from images_dataset import ImagesDataset 
from sklearn.metrics import auc, accuracy_score
from sklearn.metrics import roc_curve
import pandas as pd

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

    # CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    if modal_mode == 0:
        input_dim = clip_model.visual.output_dim  # dimension of image features from CLIP (512)
    else:
        input_dim = clip_model.visual.output_dim * 2  # dimension of image features + text features from CLIP (1024)

    # MLP
    hidden_dims = HIDDEN_DIMS 
    output_dim = 1  # Binary classification
    classifier = MLP(input_dim, hidden_dims, output_dim).to(device)
    classifier.load_state_dict(torch.load(best_model_path))
    classifier.eval() 

    print("Done")

    return clip_model, classifier

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
    plt.plot([0, 1], [0, 1], 'k--')

    # Define a list of colors for the threshold points
    colors = ['pink', 'purple', 'red', "brown"]

    for (correct_labels, preds, model_name, eer, color) in zip(correct_labels_list, preds_list, model_names, eers, colors):
        fpr, tpr, thresholds = roc_curve(correct_labels, preds)
        model_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {model_auc:.3f}) (EER = {eer:.4f})")
        
        #find the threshold closest to 0.5
        closest_threshold_index = np.argmin(np.abs(thresholds - 0.5))
        plt.plot(fpr[closest_threshold_index], tpr[closest_threshold_index], marker='o', color=color, label=f'Threshold 0.5 for {model_name}')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
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


def evaluate_model(model_path, modal_mode):
    print(f"Evaluating model: {model_path} with modal_mode {modal_mode}")
    clip_model, classifier = load_model(model_path, modal_mode)

    # dataset and dataloader for evaluation
    test_dataset = ImagesDataset(TEST_DATA_PATH, TEST_CSV_PATH, IMAGE_SIZE, set="test", modal_mode=modal_mode)  
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    predictions = []
    predictions_not_rounded = []
    ground_truths = []
    classified_images = []

    for batch_index, (image_names, images, captions, labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Inference"):
        images = np.transpose(images, (0, 3, 1, 2))
        images = images.to(device)
        if modal_mode == 1: 
            captions = torch.squeeze(captions, dim=1) 
            captions = captions.to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(images).float()
            labels = labels.float().unsqueeze(1).to(device)
            features = image_features if modal_mode == 0 else torch.cat((image_features, clip_model.encode_text(captions) / clip_model.encode_text(captions).norm(dim=-1, keepdim=True)), dim=1)
            
            features = torch.nn.functional.normalize(features)
            outputs = classifier(features.float())
            predicted_test = torch.round(torch.sigmoid(outputs)).cpu().numpy()
            labels = labels.cpu().numpy()

            predictions.extend(predicted_test)
            predictions_not_rounded.extend(torch.sigmoid(outputs).cpu().numpy())
            ground_truths.extend(labels)

            for image_name, true_label, pred in zip(image_names, labels, predicted_test.flatten()):
                classification_status = 'Correctly Classified' if pred == true_label else 'Misclassified'
                classified_images.append((image_name if isinstance(image_name, str) else image_name.item(), true_label.item(), pred.item(), classification_status))

    accuracy = accuracy_score(ground_truths, predictions)
    print(f"\nAccuracy: {accuracy:.4f}")
    report_df = print_results(classified_images)
    print(report_df)

    return ground_truths, predictions_not_rounded, f"{model_path.split('/')[-3]}"


if __name__ == "__main__":

    torch.backends.cudnn.deterministic = True
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    
    model_configs = [
    ("/home/enriconello/DeepFakeDetection/Thesis/5_biased_detection/comparison/train_1_unimodal/run3/best_unimodal_classifier.pth", 0),
    ("/home/enriconello/DeepFakeDetection/Thesis/5_biased_detection/comparison/train_1_multimodal/run_1/best_multimodal_classifier.pth", 1),
    ("/home/enriconello/DeepFakeDetection/Thesis/5_biased_detection/comparison/train_2_unimodal/run3/best_unimodal_classifier.pth", 0),
    ("/home/enriconello/DeepFakeDetection/Thesis/5_biased_detection/comparison/train_2_multimodal/run7/best_multimodal_classifier.pth", 1),
    ]

correct_labels_list = []
preds_list = []
model_names = []

for model_path, modal_mode in model_configs:
    ground_truths, predictions_not_rounded, model_name = evaluate_model(model_path, modal_mode)
    correct_labels_list.append(ground_truths)
    preds_list.append(predictions_not_rounded)
    model_names.append(model_name)

eers = []

for ground_truths, predictions_not_rounded in zip(correct_labels_list, preds_list):
    eer, threshold = calculate_eer(ground_truths, predictions_not_rounded)
    eers.append(eer)

save_all_roc_curves(correct_labels_list, preds_list, model_names, eers)