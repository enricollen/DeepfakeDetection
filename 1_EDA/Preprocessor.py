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
        """
        Calculates the SHA256 hash of a file
        Args:
            filepath: path to the file
            block_size: size of the block to read at a time
        Returns:
            The SHA256 hash of the file
        """
        file_hash = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                file_hash.update(block)
        return file_hash.hexdigest()

    def duplicates_batch_detection_to_json(self):
        """
        Detects batches of duplicate files and saves them to a json file
        """
        print("Starting partial duplicates detection...") 
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

        print("Done")

    def merge_duplicates(self):
        """
        Merges all the partial duplicates json files into one single json file
        """
        print("Starting merging phase of partial duplicates...") 
        batch_counter = 1

        while True:
            batch_file_path = os.path.join(self.output_dir, f'duplicates_batch_{batch_counter}.json')

            if os.path.exists(batch_file_path):
                with open(batch_file_path, 'r') as file:
                    batch_duplicates = json.load(file)
                    self.all_duplicates.update(batch_duplicates) # method to append a list to another list in set()
                batch_counter += 1
            else:
                break

        output_file_path = os.path.join(self.output_dir, 'all_duplicates.json')
        with open(output_file_path, 'w') as file:
            json.dump(list(self.all_duplicates), file)
        
        print("Done")

    def restore_duplicates_from_json(self):
        """
        Restores the list of duplicates from the json file
        """
        print("Restoring duplicates from json file...") 
        duplicates_file_path = os.path.join(self.output_dir, 'all_duplicates.json')
        with open(duplicates_file_path, 'r') as file:
            self.all_duplicates = json.load(file)
        print("Done")

    def check_matching_duplicates_in_df(df_train, df_test, df_val):
        """
        Checks how many matching duplicates are in the training, test and validation sets dataframes
        Args:
            df_train: training set dataframe
            df_test: test set dataframe
            df_val: validation set dataframe
        """
        result_train_df = df_train[df_train['id'].isin(preprocessor.all_duplicates[:10000])].loc[:, "id"]
        result_test_df = df_test[df_test['id'].isin(preprocessor.all_duplicates[:10000])].loc[:, "id"]
        result_val_df = df_val[df_val['id'].isin(preprocessor.all_duplicates[:10000])].loc[:, "id"]
        print("Matching duplicates in training set: " + str(len(result_train_df)) + "\nMatching duplicates in test set: " + str(len(result_test_df)) +
           "\nMatching duplicates in validation set: " + str(len(result_val_df)) + "\n=> Total Matches: " + str(len(result_train_df)+len(result_test_df)+len(result_val_df)))

    def df_column_drop(df_train, df_test, df_val, column_name="image_url"):
        """
        Drops the column image_url from the dataframes
        """
        df_train.drop(columns=[column_name], inplace=True)
        df_test.drop(columns=[column_name], inplace=True)
        df_val.drop(columns=[column_name], inplace=True)
        return df_train, df_test, df_val

    def remove_duplicates_from_df(df_train, df_test, df_val):
        """
        Removes the duplicates contained in "all_duplicates" class field from the dataframes
        """
        print("Removing duplicates from dataframes...")
        print("Training dataframe initial rows: " + str(len(df_train)))
        df_train = df_train[~df_train['id'].isin(preprocessor.all_duplicates)]
        print("Training dataframe final rows: " + str(len(df_train)))
        print("Test dataframe initial rows: " + str(len(df_test)))
        df_test = df_test[~df_test['id'].isin(preprocessor.all_duplicates)]
        print("Test dataframe final rows: " + str(len(df_test)))
        print("Validation dataframe initial rows: " + str(len(df_val)))
        df_val = df_val[~df_val['id'].isin(preprocessor.all_duplicates)]
        print("Validation dataframe final rows: " + str(len(df_val)))
        print("Done")
        return df_train, df_test, df_val

    def save_df_to_csv(df_train, df_test, df_val):
        print("Saving dataframe to csv files...")
        df_train.to_csv('C:/Users/nello/Desktop/df_train.tsv', sep='\t', index=False)
        df_test.to_csv('C:/Users/nello/Desktop/df_test.tsv', sep='\t', index=False)
        df_val.to_csv('C:/Users/nello/Desktop/df_val.tsv', sep='\t', index=False)
        print("Done")


if __name__ == "__main__":

    preprocessor = Preprocessor()
    #preprocessor.duplicates_batch_detection_to_json()
    #preprocessor.merge_duplicates()
    preprocessor.restore_duplicates_from_json()

    if(os.getenv('DATA_DIR')=="TRUE"):
        df_train = pd.read_csv(preprocessor.multimodal_train_tsv_dir, sep='\t')
        df_test = pd.read_csv(preprocessor.multimodal_test_tsv_dir, sep='\t')
        df_val = pd.read_csv(preprocessor.multimodal_val_tsv_dir, sep='\t')
    else:
        df_train = pd.read_csv(preprocessor.all_train_tsv_dir, sep='\t')
        df_test = pd.read_csv(preprocessor.all_test_tsv_dir, sep='\t')
        df_val = pd.read_csv(preprocessor.all_val_tsv_dir, sep='\t')

    # drops the column image_url from the dataframes
    df_train, df_test, df_val = Preprocessor.df_column_drop(df_train, df_test, df_val, column_name="image_url")

    # drops the duplicates from the dataframes
    df_train, df_test, df_val = Preprocessor.remove_duplicates_from_df(df_train, df_test, df_val)

    # save the cleaned dataframes in csv format
    #Preprocessor.save_df_to_csv(df_train, df_test, df_val)

    #preprocessor.check_matching_duplicates_in_df(df_train, df_test, df_val)

    #display(result_df)