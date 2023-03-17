import os
import torch
import tokenizers
import pandas as pd
from tqdm import tqdm

class GoodreadsDataset:
    def __init__(self, path, config):
        self.path = path

        # Disable warning about tokenizer parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Load tokenizer
        self.tokenizer = tokenizers.Tokenizer.from_file(config["data"]["tokenizer"])
        self.tokenizer.enable_truncation(max_length=config["model"]["max_length"])
        self.tokenizer.enable_padding(length=config["model"]["max_length"])

        # Create new `pandas` methods which use `tqdm` progress
        tqdm.pandas()

        # Load data
        df = pd.read_csv(path)

        # Process data
        self.df = self.process_data_partition(df)

        # Create tensors (y should be float)
        self.x = torch.tensor(self.df["review_text"].tolist())
        self.y = torch.tensor(self.df["rating"].tolist(), dtype=torch.float) if "rating" in self.df.columns else None

    def process_data_partition(self, df):
        # Drop columns that are not needed
        df = df.drop(columns=[col for col in df.columns if col not in ["review_id", "review_text", "rating"]])

        # Drop rows with missing values
        df = df.dropna()

        # Drop rows with empty review text
        df = df[df["review_text"] != ""]

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Tokenize review text
        df["review_text"] = df["review_text"].progress_apply(lambda x: self.tokenizer.encode(x).ids)

        return df

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx]) if self.y is not None else self.x[idx]
        