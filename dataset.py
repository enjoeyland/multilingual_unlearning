import os
import json
import torch
import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset, ConcatDataset, DataLoader

XNLI_LANGUAGES = [
    "ar",
    "bg",
    "de",
    "el",
    "en",
]  # , "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]

class XNLIDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, lang="en", add_prefix=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_length
        self.lang = lang
        self.add_prefix = add_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        lang_idx = self.data[idx]["hypothesis"]["language"].index(self.lang)
        if self.add_prefix:
            text = (
                "xnli: premise: "
                + item["premise"][self.lang]
                + " hypothesis: "
                + item["hypothesis"]["translation"][lang_idx]
            )
        else:
            text = (
                item["premise"][self.lang]
                + " "
                + item["hypothesis"]["translation"][lang_idx]
            )
        inputs = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": torch.tensor(item["label"]),
        }

class XNLIEnDataset(Dataset):
    def __init__(self, tokenizer, data, max_length):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.num_classes = len(set(self.data['label']))

    def __getitem__(self, idx):
        item = self.data[idx]
        self.encodings = self.tokenizer(
            item['premise'], item['hypothesis'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            "input_ids": self.encodings['input_ids'].squeeze(),
            "attention_mask": self.encodings['attention_mask'].squeeze(),
            "labels": torch.tensor(item['label'])
        }

    def __len__(self):
        return len(self.data)

def shard_data(output_dir, data_length, shards):
    splitfile = os.path.join(f'{output_dir}', f'shard{shards}-splitfile.jsonl')
    if os.path.exists(splitfile):
        return splitfile

    indices = np.arange(data_length)
    np.random.shuffle(indices)
    partitions = np.split(
        indices,
        [t * (data_length // shards) for t in range(1, shards)],
    )

    with open(splitfile, 'w') as f:
        for i, partition in enumerate(partitions):
            f.write(json.dumps(partition.tolist()) + '\n')
    return splitfile


def get_shard(splitfile, shard):
    if os.path.exists(splitfile):
        with open(splitfile) as f:
            shards = f.readlines()
        shards = [json.loads(shard) for shard in shards]
        return np.array(shards[shard])
    else:
        raise FileNotFoundError(f"Splitfile {splitfile} not found.")

def sizeOfShard(splitfile, shard):
    '''
    Returns the size (in number of points) of the shard before any unlearning request.
    '''
    return get_shard(splitfile, shard).shape[0]

class ShardDataset(Dataset):
    def __init__(self, splitfile, shard, mixed_dataset, split, offset=0, until=None):
        self.shard = get_shard(splitfile, shard)
        self.mixed_dataset = mixed_dataset
        self.split = split
        self.offset = offset
        self.until = until if until is not None else len(self.shard)

        self.slice = self.shard[self.offset:self.until]
        if self.split == "retain":
            self.slice = self.slice[self.slice >= self.mixed_dataset.retain_start_idx]
        self.slice = self.slice.tolist()

    def __len__(self):
        return len(self.slice)

    def __getitem__(self, idx):
        if idx > len(self.slice):
            raise IndexError("Index out of the bounds of the data segment.")
        item, is_forget = self.mixed_dataset[self.slice[idx]]
        assert not is_forget
        return item
        
class MixedDataset(Dataset):
    def __init__(self, retain_data, forget_data):
        self.combined_data = ConcatDataset([forget_data, retain_data])
        self.retain_start_idx = len(forget_data)
        self.total_len = len(self.combined_data)
        self.forget_idx = list(range(self.retain_start_idx))

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data = self.combined_data[idx]
        is_forget = idx in self.forget_idx
        return data, is_forget


def load_dataloader(args, tokenizer, split):
    data = load_dataset(
        "json", data_files=os.path.join(args.data_dir, f"_{args.task}/{split}.jsonl")
    )["train"]

    datasets = []
    if args.task == "xnli":
        for lang in XNLI_LANGUAGES:
            dataset = XNLIDataset(
                data, tokenizer, args.max_length, lang=lang, add_prefix=True
            )
            datasets.append(dataset)
        if split == "valid":
            forget_data = load_dataset(
                "json",
                data_files=os.path.join(
                    args.data_dir, f"_{args.task}/forget-{args.forget_ratio}.jsonl"
                ),
            )["train"]
            for lang in XNLI_LANGUAGES:
                dataset = XNLIDataset(
                    forget_data, tokenizer, args.max_length, lang=lang, add_prefix=True
                )
                datasets.append(dataset)
    else:
        raise NotImplementedError

    dataloaders = []
    for dataset in datasets:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        dataloaders.append(dataloader)

    return dataloaders