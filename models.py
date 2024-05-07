import torch
import lightning as L

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW, Adam, SGD

from datamodules import XNLIDataModule

class MultilingualModel(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.model, cache_dir=hparams.cache_dir)
        self.datamodule = XNLIDataModule(hparams, self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(hparams.model, num_labels=self.datamodule.num_classes, cache_dir=hparams.cache_dir)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        accuracy = (outputs.logits.argmax(dim=-1) == batch["labels"]).float().mean()
        self.log_dict({"train_loss": loss, "train_accuracy": accuracy}, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss
        accuracy = (outputs.logits.argmax(dim=-1) == batch["labels"]).float().mean()

        name = self.datamodule.dataset_names["valid"][dataloader_idx]
        self.log_dict({
            f"{name}_loss": loss,
            f"{name}_accuracy": accuracy
        }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss
        accuracy = (outputs.logits.argmax(dim=-1) == batch["labels"]).float().mean()

        name = self.datamodule.dataset_names["test"][dataloader_idx]
        self.log_dict({
            f"{name}_loss": loss,
            f"{name}_accuracy": accuracy
        }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "sgd":
            optimizer = SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        return {"optimizer": optimizer}


    
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
        outputs = self(**batch)
        losses = [output.loss for output in outputs]
        loss = sum(losses) / len(losses)
        accuracies = [(output.logits.argmax(dim=-1) == batch["labels"]).float().mean() for output in outputs]
        accuracy = sum(accuracies) / len(accuracies)
        self.log_dict({"train_loss": loss, "train_accuracy": accuracy}, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        losses = [output.loss for output in outputs]
        loss = sum(losses) / len(losses)
        # Compute the accuracy by most voted class.
        votes = torch.stack([output.logits.argmax(dim=-1) for output in outputs])
        votes = votes.mode(dim=0).values
        accuracy = (votes == batch["labels"]).float().mean()

        name = self.valid_dataset_names[dataloader_idx]
        self.log_dict({
            f"{name}_loss": loss,
            f"{name}_accuracy": accuracy
        }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        losses = [output.loss for output in outputs]
        loss = sum(losses) / len(losses)
        # Compute the accuracy by most voted class.
        votes = torch.stack([output.logits.argmax(dim=-1) for output in outputs])
        votes = votes.mode(dim=0).values
        accuracy = (votes == batch["labels"]).float().mean()

        name = self.test_dataset_names[dataloader_idx]
        self.log_dict({
            f"{name}_loss": loss,
            f"{name}_accuracy": accuracy
        }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "sgd":
            optimizer = SGD(self.parameters(), lr=self.hparams.learning_rate)
        return {"optimizer": optimizer}
    
    