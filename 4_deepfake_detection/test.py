import os
import numpy as np
import torch
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotenv import load_dotenv
from MLP import MLP  
from images_dataset import ImagesDataset 
from sklearn.metrics import f1_score
import pandas as pd

load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = int(os.getenv('IMAGE_SIZE'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
NUM_WORKERS = int(os.getenv('NUM_WORKERS'))
HIDDEN_DIMS = eval(os.getenv('HIDDEN_DIMS'))
DATA_PATH = os.getenv('DATA_PATH')
TEST_CSV_PATH = os.getenv('TEST_CSV_PATH')
MODAL_MODE = int(os.getenv('MODAL_MODE'))
BEST_MODEL_PATH = os.getenv('BEST_MODEL_PATH')
REPORT_OUTPUT_PATH = os.getenv('REPORT_OUTPUT_PATH')  # Path to save the report CSV

def load_model(best_model_path):
    print("Loading model...")

    # CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    if MODAL_MODE == 0:
        input_dim = clip_model.visual.output_dim  # dimension of image features from CLIP (512)
    else:
        input_dim = clip_model.visual.output_dim * 2  # dimension of image features + text features from CLIP (1024)

    # MLP
    hidden_dims = HIDDEN_DIMS 
    output_dim = 1  # Binary classification
    classifier = MLP(input_dim, hidden_dims, output_dim).to(device)
    best_model_path = os.path.join(best_model_path, "best_unimodal_classifier.pth" if MODAL_MODE == 0 else "best_multimodal_classifier.pth")
    classifier.load_state_dict(torch.load(best_model_path))
    classifier.eval() 

    print("Done")

    return clip_model, classifier

def print_results():
    df_test = pd.read_csv(TEST_CSV_PATH)  

    # total count for each combination of attributes in the test set
    total_real_image_real_text = df_test[(df_test['real_image'] == 1) & (df_test['real_text'] == 1)].shape[0]
    total_real_image_fake_news_text = df_test[(df_test['real_image'] == 1) & (df_test['fakenews_text'] == 1)].shape[0]
    total_fake_image_real_text = df_test[(df_test['fake_image'] == 1) & (df_test['real_text'] == 1)].shape[0]
    total_fake_image_fake_news_text = df_test[(df_test['fake_image'] == 1) & (df_test['fakenews_text'] == 1)].shape[0]

    count_real_image_real_text_misclassified = 0
    count_real_image_fake_news_text_misclassified = 0
    count_fake_image_real_text_misclassified = 0
    count_fake_image_fake_news_text_misclassified = 0

    # for every misclassified image IDs
    for image_name, _ in misclassified_images:
        # find the row corresponding to the image ID
        row = df_test[df_test['id'] == image_name].iloc[0]

        # determine the combination of attributes
        if row['real_image'] == 1:
            if row['real_text'] == 1:
                count_real_image_real_text_misclassified += 1
            else:
                count_real_image_fake_news_text_misclassified += 1
        else:
            if row['real_text'] == 1:
                count_fake_image_real_text_misclassified += 1
            else:
                count_fake_image_fake_news_text_misclassified += 1

    total_counts = {
        "Real Image + Real Text": total_real_image_real_text,
        "Real Image + Fake news text": total_real_image_fake_news_text,
        "Fake Image + Real Text": total_fake_image_real_text,
        "Fake Image + Fake news text": total_fake_image_fake_news_text
    }

    misclassified_counts = {
        "Real Image + Real Text": count_real_image_real_text_misclassified,
        "Real Image + Fake news text": count_real_image_fake_news_text_misclassified,
        "Fake Image + Real Text": count_fake_image_real_text_misclassified,
        "Fake Image + Fake news text": count_fake_image_fake_news_text_misclassified
    }

    report_df = pd.DataFrame({
    'Combination': ['Real Image + Real Text', 'Real Image + Fake news text', 'Fake Image + Real Text', 'Fake Image + Fake news text'],
    'Missclassified': [misclassified_counts["Real Image + Real Text"], misclassified_counts["Real Image + Fake news text"], misclassified_counts["Fake Image + Real Text"], misclassified_counts["Fake Image + Fake news text"]]
    })

    # get total count for each combination
    total_list = [total_counts[combination] for combination in report_df['Combination']]
    report_df['Total'] = total_list
    report_df = report_df[['Combination', 'Total', 'Missclassified']]

    return report_df



if __name__ == "__main__":
    # Load model
    clip_model, classifier = load_model(BEST_MODEL_PATH)

    # dataset and dataloader for evaluation
    test_dataset = ImagesDataset(DATA_PATH, TEST_CSV_PATH, IMAGE_SIZE, set="test", modal_mode=MODAL_MODE)  
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    predictions = []
    ground_truths = []
    misclassified_images = []

    for batch_index, (image_names, images, captions, labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Inference"):
        images = np.transpose(images, (0, 3, 1, 2))
        images = images.to(device)
        if MODAL_MODE == 1: 
            captions = torch.squeeze(captions, dim=1) 
            captions = captions.to(device)

        # Perform inference
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
            predicted_test = torch.round(torch.sigmoid(outputs)).cpu().numpy()
            labels = labels.cpu().numpy()

            predictions.extend(predicted_test)
            ground_truths.extend(labels)
            misclassified_images.extend([(image_name, label) for image_name, label, pred in zip(image_names, labels, predicted_test) if pred != label])

    # Performance Eval
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    accuracy = np.mean(predictions == ground_truths)
    f1 = f1_score(ground_truths, predictions, average='binary')
    print(f"\nAccuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")

    report_df=print_results()
    print(report_df)
    #report_df.to_csv("model_performance.csv")