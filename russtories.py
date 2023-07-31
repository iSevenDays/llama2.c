"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import numpy as np
import requests
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer
import os

# Size of 1KB in bytes
KB = 1024
DATA_CACHE_DIR = "data"

DATA_DIR = "/Volumes/AI_RW/machine_learning/corpus/"
TOKENIZED_DATA_DIR = os.path.join(DATA_CACHE_DIR, "TinyRusStories_all_data")


def read_txt_files():
    """Reads all .txt files in the dataset directory."""
    txt_filenames = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True))
    print(f"Number of txt files: {len(txt_filenames)}")
    return txt_filenames

def pretokenize():
    enc = Tokenizer()

    def process_file(filename):
        with open(filename, "r") as f:
            data = f.read()
        all_tokens = []
        text = data.strip()  # get rid of leading/trailing whitespace
        tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
        all_tokens.extend(tokens)
        # convert to uint16 nparray
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        # prepare the directory structure
        relative_path = os.path.relpath(filename, DATA_DIR)
        tokenized_filename = os.path.join(TOKENIZED_DATA_DIR, relative_path.replace(".txt", ".bin"))
        os.makedirs(os.path.dirname(tokenized_filename), exist_ok=True)
        # write to disk
        with open(tokenized_filename, "wb") as f:
            f.write(all_tokens.tobytes())
        print(f"Saved {tokenized_filename}")

    # get list of txt files and tokenize all of them one by one
    txt_filenames = read_txt_files()

    # process all the files in a threadpool
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_file, txt_filenames)

    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, test_split=0.2):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.test_split = test_split

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        bin_filenames = sorted(glob.glob(os.path.join(TOKENIZED_DATA_DIR, "**", "*.bin"), recursive=True))
        train_files = []
        test_files = []
        for filename in bin_filenames:
            # Use a hash of the filename to deterministically assign it to train or test set
            hash_val = hashlib.md5(filename.encode()).hexdigest()
            # Convert the hash to a number between 0 and 1
            hash_num = int(hash_val, 16) / float(1 << 128)
            if hash_num < self.test_split:
                test_files.append(filename)
            else:
                train_files.append(filename)
        print(f'Total {len(train_files)} train_files, {len(test_files)} test_files')
        if self.split == "train":
            bin_filenames = train_files
        else:
            bin_filenames = test_files
        while True:
            rng.shuffle(bin_filenames)
            for bin_file in bin_filenames:
                # open the dataset for reading but keep it on disk with memmap
                if not os.path.exists(bin_file) or os.path.getsize(bin_file) == 0:
                    print(f"File {bin_file} does not exist or is empty.")
                    continue  # skip this file and move to the next one
                # Calculate the expected size in bytes
                expected_size = np.dtype(np.uint16).itemsize * self.max_seq_len

                # Check if the expected size is larger than 1KB
                if expected_size < KB:
                    print(f"Expected size of file {bin_file} is less than 1KB. Skipping this file.")
                    continue

                m = np.memmap(bin_file, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                if num_batches <= 0:
                    continue
                assert num_batches > 0, "this file is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y

from datasets import load_dataset, interleave_datasets, IterableDataset
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset

class ContentExtractionDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, idx):
        return self.ds[idx]["content"]

    def __len__(self):
        return len(self.ds)
    

cache_dir = '/Volumes/AI_RW/cache'
class Task:

    @staticmethod
    def iter_batches(split, batch_size, max_seq_len, device, num_workers=0):
        ds = PretokDataset(split, max_seq_len)

        # example usage: for sample in iter(ds_swift): print(sample["content"])
        #ds_swift = ContentExtractionDataset(load_dataset("bigcode/the-stack-dedup", data_dir="data/Swift", split="train", cache_dir=cache_dir))
        ds_c = load_dataset("bigcode/the-stack-dedup", data_dir="data/c", split="train", streaming=False, cache_dir=cache_dir)
        #ds_cpp = ContentExtractionDataset(load_dataset("bigcode/the-stack-dedup", data_dir="data/C++", split="train", cache_dir=cache_dir))
        #ds_html = ContentExtractionDataset(load_dataset("bigcode/the-stack-dedup", data_dir="data/HTML", split="train", cache_dir=cache_dir))

        assert not isinstance(ds, IterableDataset), "ConcatDataset for ds does not support IterableDataset"
        assert not isinstance(ds_c, IterableDataset), "ConcatDataset for ds_c does not support IterableDataset"
        dataset_programming = ds_c #interleave_datasets([ds_c], probabilities=[1], seed=42)
        

        combined_ds = ds#ConcatDataset([ds, dataset_programming])

        dl = torch.utils.data.DataLoader(
            combined_ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "train_tokenizer", "pretokenize"])
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    fun = {
        "pretokenize": pretokenize,
    }
    fun[args.stage]()