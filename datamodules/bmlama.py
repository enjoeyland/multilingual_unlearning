import torch
import random
import lightning as L

from pathlib import Path
from datasets import load_dataset
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


class BMLAMADataModule(L.LightningDataModule):
    SUPPORTED_LANGUAGES_17 = ["en", "fr", "es", "ar", "zh", "vi", "ca"]
    SUPPORTED_LANGUAGES_53 = ["en", "fr", "es", "ar", "pt", "vi",
                              "ca", "hi", "bn", "id", "eu", "ur"]

    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

        self.method = args.method
        self.task = args.task

        self.SUPPORTED_LANGUAGES = self.SUPPORTED_LANGUAGES_53
        if self.task == "bmlama17":
            self.SUPPORTED_LANGUAGES = self.SUPPORTED_LANGUAGES_17

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
                data_files=str((Path(__file__).parent.parent / self.data_dir / f"{self.task}/{split}.jsonl").resolve()),
                cache_dir=self.cache_dir,
            )["train"]

        for split in ["forget"]:
            self.data_name[split] = split
            self.data[split] = load_dataset(
                "json", 
                data_files=str((Path(__file__).parent.parent  / self.data_dir / f"{self.task}/{split}-{self.forget_num}.jsonl").resolve()),
                cache_dir=self.cache_dir,
            )["train"]
            self.data_lang[split] = self.lang[split]

        for split in ["retain"]:
            self.data_name[split] = split
            self.data[split] = load_dataset(
                "json", 
                data_files=str((Path(__file__).parent.parent / self.data_dir / f"{self.task}/{split}-{self.forget_num}-x{self.args.retain_multiplier}.jsonl").resolve()),
                cache_dir=self.cache_dir,
            )["train"]
            self.data_lang[split] = self.lang[split]


        # Prepare datasets
        self.datasets = defaultdict(list)
        self.dataset_names = defaultdict(list)

        if stage == "fit":
            dataset_mapping = {
                "train": ["forget", "retain"],
                "valid": ["valid", "forget"],
            }

            if self.method == "finetune":
                if self.args.fit_target == "forget":
                    dataset_mapping["train"] = ["forget"]
                elif self.args.fit_target == "retain":
                    dataset_mapping["train"] = ["retain"]
                    dataset_mapping["valid"] = ["valid", "retain"]
            elif "sisa" in self.method:
                raise NotImplementedError("SISA method not implemented for FLORES dataset.")

            # Randomly sample languages
            for split in dataset_mapping["train"]:
                self.datasets["train"].append(BMLAMADataset(self.data[split], self.tokenizer, self.max_length, lang=self.data_lang[split]))
                self.dataset_names["train"].append(f"train/{self.data_name[split]}")

            # Evaluate all training languages
            for lang in self.lang["retain"]:
                for split in dataset_mapping["valid"]:
                    self.datasets["valid"].append(BMLAMADataset(self.data[split], self.tokenizer, self.max_length, lang=lang))
                    self.dataset_names["valid"].append(f"val/{self.data_name[split]}_{lang}")

        elif stage == "validate":
            dataset_mapping = {
                "valid": ["valid", "forget"],
            }

            # Evaluate all training languages
            for lang in self.lang["retain"]:
                for split in dataset_mapping["valid"]:
                    self.datasets["valid"].append(BMLAMADataset(self.data[split], self.tokenizer, self.max_length, lang=lang))
                    self.dataset_names["valid"].append(f"val/{self.data_name[split]}_{lang}")

        elif stage == "test":
            dataset_mapping = {
                "test": ["test", "forget"],
            }

            # Test different languages
            langs = self.lang["retain"] if self.args.test_src_lang_only else self.SUPPORTED_LANGUAGES
            for lang in langs:
                for split in dataset_mapping["test"]:
                    self.datasets["test"].append(BMLAMADataset(self.data[split], self.tokenizer, self.max_length, lang=lang))
                    self.dataset_names["test"].append(f"test/{self.data_name[split]}_{lang}")
            
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


class BMLAMADataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=32, lang="en"):
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

        item = self.data[idx]
        prompt_str = item["prompt"][lang].replace("\u200b", "")
        answers = item["answers"][lang]
        candidates = item["candidates"][lang]
        prompt = prompt_str.replace("<mask>", answers[0])

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
        )
        labels = inputs["input_ids"].clone()
        mask = torch.ones_like(labels)

        # Mask all tokens except the answer token s.t. loss is only computed on the answer token
        if "xglm" in self.tokenizer.name_or_path:
            ans_token_id = self.tokenizer.encode(" "+answers[0])[1:]    # Add space for exact match, remove cls token
        elif "bloom" in self.tokenizer.name_or_path:
            ans_token_id = self.tokenizer.encode(" "+answers[0])        # Add space for exact match
        else:
            raise ValueError(f"Unsupported model: {self.tokenizer.name_or_path}")

        # Ensure answer token is in labels
        for _id in ans_token_id:
            assert _id in labels
            assert _id not in self.tokenizer.all_special_ids
            mask[labels == _id] = 0

        # Mask all other tokens
        labels[mask == 1] = -100

        # Pad candidates to length 10
        candidates += [""] * (10 - len(candidates))

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
            "prompt": prompt_str,
            "candidates": candidates,
            "answers": answers[0],
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    data_dir = "../../research/multilingual-unlearning/data"    
    data = load_dataset("json", data_files=str((Path(__file__).parent.parent  / data_dir / "bmlama53/valid.jsonl").resolve()), cache_dir="../../.cache")["train"]
    tokenizer = AutoTokenizer.from_pretrained(
                    "bigscience/bloom-560M",
                    cache_dir="../../.cache",
                    local_files_only=True,
                )
    dataset = BMLAMADataset(data, tokenizer, 32, "en")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    SUPPORTED_LANGUAGES_17 = ["en", "fr", "es", "ar", "zh", "vi", "ca"]
    SUPPORTED_LANGUAGES_53 = ["en", "fr", "es", "ar", "pt", "vi",
                              "ca", "hi", "bn", "id", "eu", "ur"]
    for lang in SUPPORTED_LANGUAGES_53:
        lengths = []
        for item in data:
            prompt = item["prompt"][lang].replace("\u200b", "")
            answers = item["answers"][lang]
            prompt = prompt.replace("<mask>", answers[0])
            lengths.append(len(tokenizer(prompt)["input_ids"]))
        
        print(f"Language: {lang}")
        print(f"Max length: {max(lengths)}")
        print(f"Min length: {min(lengths)}")
        print(f"Mean length: {sum(lengths) / len(lengths)}")
