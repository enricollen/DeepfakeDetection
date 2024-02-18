import torch
import random
import os
import cv2
import numpy as np
import math
from multiprocessing import Manager
from multiprocessing.pool import Pool
from progress.bar import Bar
from tqdm import tqdm
from functools import partial
from sklearn.utils import shuffle
from pytorch_lightning import seed_everything
import clip
import timm
from timm.scheduler.cosine_lr import CosineLRScheduler
from albumentations import Compose, PadIfNeeded
import collections
from progress.bar import ChargingBar
from torch.utils.tensorboard import SummaryWriter
from utils import check_correct, center_crop
from images_dataset import ImagesDataset
from transforms.albu import IsotropicResize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

IMAGE_SIZE = 224
NUM_WORKERS = 1
NUM_EPOCHS = 1
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.0000001
BATCH_SIZE = 30
PATIENCE = 3
TRAIN_DATA_PATH = 'train_small/'
VAL_DATA_PATH = 'val_small/'
MODEL_PATH = 'models_saved'
MODEL_NAME = 'CLIP'

def read_images(data_path, transform):
    dataset = []
    for image_name in os.listdir(data_path):
        if not image_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        label = 1 if image_name.startswith("SD") else 0

        image_path = os.path.join(data_path, image_name)
        image = cv2.imread(image_path)
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

# Main body
if __name__ == "__main__":
    seed_everything(42)
    random.seed(42)

    # Model Loading
    clip_model, preprocess = clip.load("ViT-B/32", device=torch.device('cuda'))
    dim = 0.5
    clip_model = clip_model.float()
    clip_model.to("cuda")
    clip_model.eval()
    model = torch.nn.Linear(int(1024*dim), 1)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model parameters:", params)

    # Read dataset
    transform = create_pre_transform(IMAGE_SIZE)
    train_dataset = read_images(TRAIN_DATA_PATH, transform)
    validation_dataset = read_images(VAL_DATA_PATH, transform)

    """mgr = Manager()
    train_dataset = mgr.list()
    training_captions = os.listdir(TRAIN_DATA_PATH)

    with Pool(processes=NUM_WORKERS) as p:
        with tqdm(total=len(training_captions)) as pbar:
            for v in p.imap_unordered(partial(read_images, dataset=train_dataset, data_path = TRAIN_DATA_PATH, transform=transform), training_captions):
                pbar.update()
    
    validation_captions = os.listdir(VAL_DATA_PATH)
    validation_dataset = mgr.list()

    with Pool(processes=NUM_WORKERS) as p:
        with tqdm(total=len(validation_captions)) as pbar:
            for v in p.imap_unordered(partial(read_images, dataset=validation_dataset, data_path = VAL_DATA_PATH, transform=transform), validation_captions):
                pbar.update()"""

    # Extract labels and images from datasets
    train_labels = [float(row[1]) for row in train_dataset]
    train_dataset = [row[0] for row in train_dataset]
    validation_labels = [float(row[1]) for row in validation_dataset]
    validation_dataset = [row[0] for row in validation_dataset]

    # Calculate number of samples
    train_samples = len(train_dataset)
    validation_samples = len(validation_dataset)

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

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))

    train_dataset = ImagesDataset(np.asarray(train_dataset), np.asarray(train_labels), IMAGE_SIZE, mode='train') #train_captions
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, sampler=None,
                                batch_sampler=None, num_workers=NUM_WORKERS, collate_fn=None,
                                pin_memory=False, drop_last=False, timeout=0,
                                worker_init_fn=None, prefetch_factor=2,
                                persistent_workers=False)
    #del train_dataset

    validation_dataset = ImagesDataset(np.asarray(validation_dataset), np.asarray(validation_labels), IMAGE_SIZE, mode='validation') #validation_captions
    val_dataset = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=NUM_WORKERS, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)
    #del validation_dataset


    # TRAINING
    tb_logger = SummaryWriter(log_dir="LOGGER", comment='')
    experiment_path = tb_logger.get_logdir()

    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    num_steps = int(NUM_EPOCHS * len(dl))
    lr_scheduler = CosineLRScheduler(
                    optimizer,
                    t_initial=num_steps,
                    lr_min=LEARNING_RATE * 1e-3,
                    cycle_limit=9,
                    t_in_epochs=False,
        )
    starting_epoch = 0
    model = model.to("cuda")
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf

    for t in range(starting_epoch, NUM_EPOCHS + 1):
        save_model = False
        if not_improved_loss == PATIENCE:
            break
        counter = 0

        total_loss = 0
        total_val_loss = 0

        bar = ChargingBar('EPOCH #' + str(t), max=(len(dl)*BATCH_SIZE)+len(val_dataset))
        train_correct = 0
        positive = 0
        negative = 0

        train_batches = len(dl)
        val_batches = len(val_dataset)
        total_batches = train_batches + val_batches

        for index, (images, labels) in enumerate(dl): #captions
            images = np.transpose(images, (0, 3, 1, 2))
            labels = labels.unsqueeze(1)
            images = images.to("cuda")
            #captions = captions.to(opt.gpu_id)

            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                """if opt.mode == 1:
                    text_features = clip_model.encode_text(captions)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    features = torch.cat((image_features, text_features), dim = 1)
                else:"""
                features = image_features

                features = features.float()

                #features = torch.nn.functional.normalize(features)
                y_pred = model(features)
            y_pred = y_pred.cpu()
            y_pred.requires_grad_(True)
            labels.requires_grad_(True)

            loss = loss_fn(y_pred, labels)

            corrects, positive_class, negative_class = check_correct(y_pred, labels)
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            lr_scheduler.step_update((t * (train_batches) + index))
            counter += 1
            total_loss += round(loss.item(), 2)
            for i in range(BATCH_SIZE):
                bar.next()

            if index%100 == 0:
                print("\nLoss: ", total_loss/counter, "Accuracy: ", train_correct/(counter*BATCH_SIZE), "Train 0s: ", negative, "Train 1s:", positive)

        
        
        
        val_counter = 0
        val_correct = 0
        val_positive = 0
        val_negative = 0

        train_correct /= train_samples
        total_loss /= counter
        for index, (val_images, val_labels) in enumerate(val_dataset): #val_captions

            val_images = np.transpose(val_images, (0, 3, 1, 2))
            val_images = val_images.to("cuda")
            #val_captions = val_captions.to("cuda")

            val_labels = val_labels.unsqueeze(1)
            with torch.no_grad():
                image_features = clip_model.encode_image(val_images)
                """if opt.mode == 1:
                    text_features = clip_model.encode_text(val_captions)
                    features = torch.cat((image_features, text_features), dim=1)
                    features = torch.cat((image_features, text_features), dim = 1)
                else:"""
                features = image_features
                features = features.float()
                #features = torch.nn.functional.normalize(features)
                val_pred = model(features)
                val_pred = val_pred.cpu()
                val_loss = loss_fn(val_pred, val_labels)
                total_val_loss += round(val_loss.item(), 2)
                corrects, positive_class, negative_class = check_correct(val_pred, val_labels)
                val_correct += corrects
                val_positive += positive_class
                val_negative += negative_class
                val_counter += 1
                bar.next()

        #scheduler.step()
        bar.finish()


        total_val_loss /= val_counter
        val_correct /= validation_samples
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            save_model = True
            not_improved_loss = 0

        tb_logger.add_scalar("Training/Accuracy", train_correct, t)
        tb_logger.add_scalar("Training/Loss", total_loss, t)
        tb_logger.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], t)
        tb_logger.add_scalar("Validation/Loss", total_loss, t)
        tb_logger.add_scalar("Validation/Accuracy", val_correct, t)

        previous_loss = total_val_loss
        print("#" + str(t) + "/" + str(NUM_EPOCHS) + " loss:" +
            str(total_loss) + " accuracy:" + str(train_correct) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_correct) + " val_0s:" + str(val_negative) + "/" + str(val_counters[0]) + " val_1s:" + str(val_positive) + "/" + str(val_counters[1]))

        # Export model state
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        #if save_model and t > NUM_EPOCHS-20:
            #torch.save(model.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME + "_" + str(t)))