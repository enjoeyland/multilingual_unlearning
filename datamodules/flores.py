
import random
import lightning as L

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class FLORESDataModule(L.LightningDataModule):
    SUPPORTED_LANGUAGES = [
        "en", "fr", "es", "zh", "ar", "vi",
        "eu", "ur", "te", "sw",
        # "ne", "mr", "ml", "yo", "xh", "zu",
    ]

    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

        self.method = args.method
        self.task = args.task
        
        self.lang = {
            "forget": args.forget_lang,
            "retain": args.retain_lang,
        }
        self.forget_num = args.forget_num
        self.retain_multiplier = args.retain_multiplier

        self.max_length = args.max_length
        self.num_workers = args.num_workers

        self.data_dir = args.data_dir
        self.cache_dir = args.cache_dir

        self.batch_size = args.per_device_batch_size

    def setup(self, stage=None):
        # Load data
        self.data_name = {}
        self.data = {}
        self.data_lang = {}
        
        for split in ["valid", "test"]:
            self.data_name[split] = f"{split if split != 'valid' else 'val'}"
            self.data[split] = load_dataset(
                "json", 
                data_files=str((Path(__file__).parent / self.data_dir / f"{self.task}/{split}.jsonl").resolve()),
                cache_dir=self.cache_dir,
            )["train"]

        for split in ["forget"]:
            self.data_name[split] = split
            self.data[split] = load_dataset(
                "json", 
                data_files=str((Path(__file__).parent / self.data_dir / f"{self.task}/{split}-{self.forget_num}.jsonl").resolve()),
                cache_dir=self.cache_dir,
            )["train"]
            self.data_lang[split] = self.lang[split]

        for split in ["retain"]:
            self.data_name[split] = split
            self.data[split] = load_dataset(
                "json", 
                data_files=str((Path(__file__).parent / self.data_dir / f"{self.task}/{split}-{self.forget_num}-x{self.args.retain_multiplier}.jsonl").resolve()),
                cache_dir=self.cache_dir,
            )["train"]
            self.data_lang[split] = self.lang[split]

        if stage == "fit":
            dataset_mapping = {
                "train": ["forget", "retain"],
                "valid": ["valid", "forget"],
            }

            if self.method in ["negtaskvector"]:
                dataset_mapping["train"] = ["forget"]

            # Randomly sample languages
            for split in dataset_mapping["train"]:
                self.datasets["train"].append(FLORESDataset(self.data[split], self.tokenizer, self.max_length, lang=self.data_lang[split]))
                self.dataset_names["train"].append(self.data_name[split])

            for lang in self.SUPPORTED_LANGUAGES:
                for split in dataset_mapping["valid"]:
                    self.datasets["valid"].append(FLORESDataset(self.data[split], self.tokenizer, self.max_length, lang=lang))
                    self.dataset_names["valid"].append(f"{self.data_name[split]}/{lang}")

        elif stage == "validate":
            dataset_mapping = {
                "valid": ["valid", "forget"],
            }
             
            for lang in self.SUPPORTED_LANGUAGES:
                for split in dataset_mapping["valid"]:
                    self.datasets["valid"].append(FLORESDataset(self.data[split], self.tokenizer, self.max_length, lang=lang))
                    self.dataset_names["valid"].append(f"{self.data_name[split]}/{lang}")

        elif stage == "test":
            dataset_mapping = {
                "test": ["test", "forget"],
            }

            for lang in self.SUPPORTED_LANGUAGES:
                for split in dataset_mapping["valid"]:
                    self.datasets["valid"].append(FLORESDataset(self.data[split], self.tokenizer, self.max_length, lang=lang))
                    self.dataset_names["valid"].append(f"{self.data_name[split]}/{lang}")
        
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(    
            self.datasets["train"][self.trainer.current_epoch % len(self.datasets["train"])],
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


class FLORESDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=256, lang=["en"]):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.lang = lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(self.lang, list):
            if len(self.lang) > 1:
                lang = random.choice(self.lang)
            else:
                lang = self.lang[0]
        else:
            lang = self.lang

        item = self.data[idx][lang]
        inputs = self.tokenizer(
            item,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    data_dir = "../../../research/multilingual-unlearning/data"    
    data = load_dataset("json", data_files=str((Path(__file__).parent / data_dir / "flores/valid.jsonl").resolve()), cache_dir="../../.cache")["train"]
    tokenizer = AutoTokenizer.from_pretrained(
                    "bigscience/bloom-560M",
                    cache_dir="../../.cache",
                    local_files_only=True,
                )
    SUPPORTED_LANGUAGES = [
        "en", "fr", "es", "zh", "ar", "vi",
        "eu", "ur", "ne", "mr", "te", "ml",
        "sw", "yo", "xh", "zu",
    ]
    for lang in SUPPORTED_LANGUAGES:
        lengths = []
        for item in data[lang]:
            lengths.append(len(tokenizer.encode(item)))
        
        print(f"Language: {lang}")
        print(f"Max length: {max(lengths)}")
        print(f"Min length: {min(lengths)}")
        print(f"Mean length: {sum(lengths) / len(lengths)}")
