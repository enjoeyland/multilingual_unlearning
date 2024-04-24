import torch
import numpy as np
import lightning as L

from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from torch.optim import AdamW, Adam, SGD
from torch.utils.data import DataLoader

from dataset import XNLIDataset, ShardDataset, sizeOfShard, MixedDataset, shard_data


class MultilingualModel(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.num_classes = 3
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.model, cache_dir=hparams.cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(hparams.model, num_labels=3, cache_dir=hparams.cache_dir)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        accuracy = (outputs.logits.argmax(dim=-1) == batch["labels"]).float().mean()
        self.log_dict({"train_loss": loss, "train_accuracy": accuracy}, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        accuracy = (outputs.logits.argmax(dim=-1) == batch["labels"]).float().mean()
        self.log_dict({"val_loss": loss, "val_accuracy": accuracy}, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        accuracy = (outputs.logits.argmax(dim=-1) == batch["labels"]).float().mean()
        self.log_dict({"test_loss": loss, "test_accuracy": accuracy}, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "sgd":
            optimizer = SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        return {"optimizer": optimizer}


    def _load_dataloader(self, split):
        if split in ["train", "valid", "test"]:
            data = load_dataset(
                "json",
                data_files=str((Path(__file__).parent / self.hparams.data_dir / f"_{self.hparams.task}/{split}.jsonl").resolve()),
            )["train"]

        dataset = XNLIDataset(data, self.tokenizer, self.hparams.max_length)
        if split in ["train"]:
            return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)    
        if split in ["valid", "test"]:
            return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)
    
    def _load_shard_dataloader(self, split):
        retain_data = load_dataset(
            "json",
            data_files=str((Path(__file__).parent / self.hparams.data_dir / f"_{self.hparams.task}/retain-{self.hparams.forget_ratio}.jsonl").resolve()),
        )["train"]
        retain_data = XNLIDataset(retain_data, self.tokenizer, self.hparams.max_length)

        forget_data = load_dataset(
            "json",
            data_files=str((Path(__file__).parent / self.hparams.data_dir / f"_{self.hparams.task}/forget-{self.hparams.forget_ratio}.jsonl").resolve()),
        )["train"]
        forget_data = XNLIDataset(forget_data, self.tokenizer, self.hparams.max_length)

        dataset = MixedDataset(retain_data, forget_data)   
        splitfile = shard_data(self.hparams.output_dir, len(dataset), self.hparams.shards)

        shard_size = sizeOfShard(splitfile, self.hparams.shard)
        slice_size = shard_size // self.hparams.slices
        dataset = ShardDataset(
            splitfile, 
            self.hparams.shard, 
            dataset, 
            split, 
            until=(self.hparams.sl + 1) * slice_size if self.hparams.sl < self.hparams.slices - 1 else None)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def train_dataloader(self):
        if self.hparams.method == "original":
            return self._load_dataloader("train")
        elif self.hparams.method == "sisa":
            return self._load_shard_dataloader("train")
        elif self.hparams.method == "sisa-retain":
            return self._load_shard_dataloader("retain")

    def val_dataloader(self):
        return self._load_dataloader("valid")

    def test_dataloader(self):
        return self._load_dataloader("test")
    
class ShardEnsembleModel(L.LightningModule):
    def __init__(self, models, hparams):
        super().__init__()
        self.models = models
        self.save_hyperparameters(hparams)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.model, cache_dir=hparams.cache_dir)


    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = [model(input_ids, attention_mask=attention_mask, labels=labels) for model in self.models]
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self(*batch)
        losses = [output.loss for output in outputs]
        loss = sum(losses) / len(losses)
        accuracies = [(output.logits.argmax(dim=-1) == batch["labels"]).float().mean() for output in outputs]
        accuracy = sum(accuracies) / len(accuracies)
        self.log_dict({"train_loss": loss, "train_accuracy": accuracy}, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        outputs = self(*batch)
        losses = [output.loss for output in outputs]
        loss = sum(losses) / len(losses)
        # Compute the accuracy by most voted class.
        votes = torch.stack([output.logits.argmax(dim=-1) for output in outputs])
        votes = votes.mode(dim=0).values
        accuracy = (votes == batch["labels"]).float().mean()
        self.log_dict({"test_loss": loss, "test_accuracy": accuracy}, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "sgd":
            optimizer = SGD(self.parameters(), lr=self.hparams.learning_rate)
        return {"optimizer": optimizer}
    
    