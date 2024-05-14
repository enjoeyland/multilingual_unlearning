from collections import defaultdict
import lightning as L

from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader

from dataset import XNLIDataset, MixedDataset, ShardDataset, shard_data, sizeOfShard

XNLI_LANGUAGES = [
    "ar",
    "bg",
    "de",
    "el",
    "en",
]  # , "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]

class XNLIDataModule(L.LightningDataModule):
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

            for split, data_splits in dataset_mapping.items():
                for s in data_splits:
                    self.datasets[split].append(XNLIDataset(self.data[s], self.tokenizer, self.max_length))
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
                for lang in XNLI_LANGUAGES:
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