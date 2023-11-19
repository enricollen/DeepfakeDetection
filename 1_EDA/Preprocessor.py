import os
import hashlib
import json
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
from IPython.display import display


load_dotenv()  # Load variables from .env file

class Preprocessor:
    def __init__(self):
        self.data_dir = os.getenv('DATA_DIR')
        self.output_dir = os.getenv('OUTPUT_DIR')
        self.batch_size = int(os.getenv('BATCH_SIZE'))
        self.multimodal_train_tsv_dir = os.getenv('MULTIMODAL_TRAIN_TSV')
        self.multimodal_test_tsv_dir = os.getenv('MULTIMODAL_TEST_TSV')
        self.multimodal_val_tsv_dir = os.getenv('MULTIMODAL_VAL_TSV')
        self.all_train_tsv_dir = os.getenv('ALL_TRAIN_TSV')
        self.all_test_tsv_dir = os.getenv('ALL_TEST_TSV')
        self.all_val_tsv_dir = os.getenv('ALL_VAL_TSV')
        self.all_duplicates = set()

    def file_hash(self, filepath, block_size=65536):
        file_hash = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                file_hash.update(block)
        return file_hash.hexdigest()

    def duplicates_batch_detection(self):
        """
        Detects batches of duplicate files and saves them to a json file
        """
        os.chdir(self.data_dir)
        hash_keys = {}
        all_duplicates = set()
        total_files = len([filename for filename in os.listdir() if os.path.isfile(filename)])
        batch_counter = 1

        for filename in tqdm(os.listdir(), total=total_files, desc="Processing files"):
            if os.path.isfile(filename):
                filename_without_extension = os.path.splitext(filename)[0]
                filehash = self.file_hash(filename)

                if filehash in hash_keys:
                    original_filename = hash_keys[filehash]
                    all_duplicates.add(original_filename)
                    all_duplicates.add(filename_without_extension)
                else:
                    hash_keys[filehash] = filename_without_extension

                if len(all_duplicates) >= self.batch_size:
                    output_file_path = os.path.join(self.output_dir, f'duplicates_batch_{batch_counter}.json')
                    current_batch = list(all_duplicates)

                    with open(output_file_path, 'w') as file:
                        json.dump(current_batch, file)

                    batch_counter += 1
                    all_duplicates = set()

        if all_duplicates:
            output_file_path = os.path.join(self.output_dir, f'duplicates_batch_{batch_counter}.json')
            remaining_batch = list(all_duplicates)

            with open(output_file_path, 'w') as file:
                json.dump(remaining_batch, file)

    def merge_duplicates(self):
        batch_counter = 1

        while True:
            batch_file_path = os.path.join(self.output_dir, f'duplicates_batch_{batch_counter}.json')

            if os.path.exists(batch_file_path):
                with open(batch_file_path, 'r') as file:
                    batch_duplicates = json.load(file)
                    self.all_duplicates.update(batch_duplicates)
                batch_counter += 1
            else:
                break

        output_file_path = os.path.join(self.output_dir, 'all_duplicates.json')
        with open(output_file_path, 'w') as file:
            json.dump(list(self.all_duplicates), file)

    def restore_duplicates_from_json(self):
        duplicates_file_path = os.path.join(self.output_dir, 'all_duplicates.json')
        with open(duplicates_file_path, 'r') as file:
            self.all_duplicates = json.load(file)


if __name__ == "__main__":

    preprocessor = Preprocessor()

    #preprocessor.duplicates_batch_detection()
    #print("Done partial duplicates detection")

    #preprocessor.merge_duplicates()
    #print("Done merging partial duplicates files")

    preprocessor.restore_duplicates_from_json()
    print("Done restoring duplicates from json file")
    #print(preprocessor.all_duplicates[:10])

    #df = pd.read_csv(preprocessor.multimodal_tsv_dir, sep='\t')
    df_train = pd.read_csv(preprocessor.all_train_tsv_dir, sep='\t')
    df_test = pd.read_csv(preprocessor.all_test_tsv_dir, sep='\t')
    df_val = pd.read_csv(preprocessor.all_val_tsv_dir, sep='\t')
    result_train_df = df_train[df_train['id'].isin(preprocessor.all_duplicates[:10])].loc[:, "id"]
    result_test_df = df_test[df_test['id'].isin(preprocessor.all_duplicates[:10])].loc[:, "id"]
    result_val_df = df_val[df_val['id'].isin(preprocessor.all_duplicates[:10])].loc[:, "id"]
    print("Matching duplicates in training set: " + str(len(result_train_df)) + "\nMatching duplicates in test set: " + str(len(result_test_df)) +
           "\nMatching duplicates in validation set: " + str(len(result_val_df)) + "\n=> Total Matches: " + str(len(result_train_df)+len(result_test_df)+len(result_val_df)))

    
    #display(result_df)