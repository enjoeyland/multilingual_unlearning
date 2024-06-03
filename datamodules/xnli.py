import torch
import lightning as L

from pathlib import Path
from datasets import load_dataset
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

from .sisa_dataset import ShardDataset, MixedDataset, shard_data, sizeOfShard

class XNLIDataModule(L.LightningDataModule):
    SUPPORTED_LANGUAGES = [
        "ar",
        "bg",
        "de",
        "el",
        "en",
    ]  # , "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]

    def __init__(self, args, tokenizer):
        super().__init__()
        self.num_classes = 3

        self.args = args
        self.tokenizer = tokenizer

        self.method = args.method
        self.task = args.task
        
        self.max_length = args.max_length
        self.forget_ratio = args.forget_ratio
        self.data_dir = args.data_dir
        self.num_workers = args.num_workers

        self.batch_size = args.per_device_batch_size

    def setup(self, stage):
        # Load data
        self.data_name = {}
        self.data = {}
        
        for split in ["train", "valid", "test"]:
            self.data_name[split] = f"{split if split != 'valid' else 'val'}"
            self.data[split] = load_dataset(
                "json", data_files=str((Path(__file__).parent / self.data_dir / f"{self.task}/{split}.jsonl").resolve()),
            )["train"]

        for split in ["forget", "retain"]:
            self.data_name[split] = split
            self.data[split] = load_dataset(
                "json", data_files=str((Path(__file__).parent / self.data_dir / f"{self.task}/{split}-{self.forget_ratio}.jsonl").resolve()),
            )["train"]

        # Prepare datasets
        self.datasets = defaultdict(list)
        self.dataset_names = defaultdict(list)
    
        if stage == "fit":
            dataset_mapping = {
                "train": ["train"],
                "valid": ["valid", "forget"],
                "test": ["test"],
            }
            if self.method in ["original"]:
                dataset_mapping["valid"] = ["valid"]

            for split, data_splits in dataset_mapping.items():
                for s in data_splits:
                    self.datasets[split].append(XNLIDataset(self.data[s], self.tokenizer, self.max_length, lang="en", add_prefix=True))
                    self.dataset_names[split].append(self.data_name[s])

            if self.method in ["sisa", "sisa-retain"]:
                retain_data = self.data["retain"]
                forget_data = self.data["forget"]
                dataset = MixedDataset(retain_data, forget_data)
                splitfile = shard_data(self.args.output_dir, len(dataset), self.args.shards)

                shard_size = sizeOfShard(splitfile, self.args.shard)
                slice_size = shard_size // self.args.slices
                dataset = ShardDataset(
                    splitfile, 
                    self.args.shard, 
                    dataset, 
                    "retain" if self.method == "sisa-retain" else "train", 
                    until=(self.args.sl + 1) * slice_size if self.args.sl < self.args.slices - 1 else None
                )

                self.datasets["train"] = [DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)]
                self.dataset_names["train"] = ["train"]

        elif stage == "validate":
            dataset_mapping = {
                "valid": ["valid", "forget"],
            }

            for split in dataset_mapping["valid"]:
                for lang in self.SUPPORTED_LANGUAGES:
                    dataset = XNLIDataset(
                        self.data[split], self.tokenizer, self.max_length, lang=lang, add_prefix=True
                    )
                    self.datasets["valid"].append(dataset)
                    self.dataset_names["valid"].append(f"{lang}/{self.data_name[split]}")

        else:
            raise NotImplementedError(f"Stage {stage} not implemented.")

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"][0],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        dataloaders = []
        for dataset in self.datasets["valid"]:
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            dataloaders.append(dataloader)
        return dataloaders
    
    def test_dataloader(self):
        dataloaders = []
        for dataset in self.datasets["test"]:
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            dataloaders.append(dataloader)
        return dataloaders
    
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
            text = f"xnli: premise: {item['premise'][self.lang]} hypothesis: {item['hypothesis']['translation'][lang_idx]}"
        else:
            text = f"{item['premise'][self.lang]} {item['hypothesis']['translation'][lang_idx]}"

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