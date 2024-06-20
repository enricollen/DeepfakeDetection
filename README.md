# 🎭 DeepFakeNews Dataset: A Comprehensive Resource for Misinformation Detection

The DeepFakeNews dataset is a novel and comprehensive dataset designed for the detection of both deepfakes and fake news. This dataset is an extension and enhancement of the existing [Fakeddit](https://fakeddit.netlify.app/) fake news dataset (i strongly suggest reading the related paper [HERE](https://arxiv.org/abs/1911.03854) from the authors to better understand this dataset, with significant modifications to cater specifically to the complexities of modern misinformation).

## 🚀 Enhancements
Derived from the Fakeddit fake news dataset, the DeepFakeNews dataset comprehends a total of 509,916 images and has been enriched with 254,958 deepfake images generated using three different generative models:
* Stable Diffusion 2
* Dreamlike
* GLIDE

### ⚖️ Balance and Composition
- **Balanced Dataset**: Contains an equal number of pristine (authentic) and generated (deepfake) images.
- **Removal of Hand-Modified Content**: The original "manipulated content" category from Fakeddit, which consisted of images altered or modified by hand, has been removed. These have been replaced with deepfakes to provide a more relevant and challenging set of synthetic images.
- **Cleaning and Quality Control**: The Fakeddit dataset was thoroughly cleaned, removing any images that were not found, contained only logos, or were otherwise unsuitable for deepfake detection. This cleaning process ensures a higher quality and more reliable dataset for training and evaluation.

## 🛠️ Application
The DeepFakeNews dataset is suitable for both deepfake detection and fake news detection. Its diverse and balanced nature makes it an excellent benchmark for evaluating multimodal detection systems that analyze both visual and textual content.

## 📁 Dataset Structure
The dataset is publicly available on Zenodo [HERE](https://zenodo.org/records/11186584) and comes with three CSV files for training, testing, and validation sets, along with corresponding zip files containing the split images for each set. The deepfake images are named in both the CSV files and the image filenames following a specific format based on the generative model used: "SD_fake_imageid" for Stable Diffusion, "GL_fake_imageid" for GLIDE, and "DL_fake_imageid" for Dreamlike.

### 🔄 Deepfake Generation Pipeline
The Deepfake Generation Pipeline involves a two-step approach:
1. **Caption Generation**: First generating a caption for a pristine image using a captioning model.
2. **Image Generation**: Feeding this caption into a generative model to create a new synthetic image.

By incorporating images from multiple generative technologies, the dataset is designed to prevent any bias towards a single generation method in the training process of detection models. This choice aims to enhance the generalization capabilities of models trained on this dataset, enabling them to effectively recognize and flag deepfake content produced by a variety of different methods, not just the ones they have been exposed to during training. The other half consists of pristine, unaltered images to ensure a balanced dataset, crucial for unbiased training and evaluation of detection models.

## 🔙 Retrocompatibility with Fakeddit
The dataset has been structured to maintain retrocompatibility with the original Fakeddit dataset. All samples have retained their original Fakeddit class labels (6_way_label), allowing for fine-grained fake news detection across the five original categories: True, Satire/Parody, False Connection, Imposter Content, and Misleading Content. This feature ensures that the DeepFakeNews dataset can be used not only for multimodal and unimodal deepfake detection but also for traditional fake news detection tasks. It offers a versatile resource for a wide range of research scenarios, enhancing its utility in the field of digital misinformation detection.

For full info and details about dataset creation, cleaning pipeline, composition, and generation process, please refer to my [Master Thesis](https://etd.adm.unipi.it/t/etd-05082024-174758/).
