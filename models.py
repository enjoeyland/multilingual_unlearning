import torch
import lightning as L

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from torch.optim import AdamW, Adam
from deepspeed.ops.adam import FusedAdam
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from datamodules import XNLIDataModule


class MultilingualModel(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.model, cache_dir=hparams.cache_dir)
        if hparams.task == "xnli":
            self.datamodule = XNLIDataModule(hparams, self.tokenizer)
        else:
            raise NotImplementedError(f"Task {hparams.task} not implemented.")
        self.model = None
        # self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.model, num_labels=self.datamodule.num_classes,cache_dir=self.hparams.cache_dir)


    def configure_model(self):
        if self.model is not None:
            return
        
        if "mt5 "in self.hparams.model:
            self.model = load_model(self.hparams, AutoModelForSequenceClassification, "SEQ_CLS")
        elif "bloom" in self.hparams.model:
            # self.model = load_model(self.hparams, AutoModelForSequenceClassification, "CAUSAL_LM")
            raise NotImplementedError(f"Model {self.hparams.model} not implemented.")
        else:
            raise NotImplementedError(f"Model {self.hparams.model} not implemented.")

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
        if "deepspeed" in self.hparams.dp_strategy:
            optimizer = FusedAdam(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01, 
                                  adam_w_mode=(self.hparams.optimizer == "adamw")) 
        elif self.hparams.optimizer == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        else:
            raise NotImplementedError(f"Optimizer {self.hparams.optimizer} not implemented.")
        
        if self.hparams.lr_scheduler_type == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(
                    self.hparams.warmup_ratio * self.trainer.estimated_stepping_batches
                ),
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.hparams.lr_scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(
                    self.hparams.warmup_ratio * self.trainer.estimated_stepping_batches
                ),
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        else:
            raise NotImplementedError(f"LR scheduler {self.hparams.lr_scheduler_type} not implemented.")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }


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
        return {"optimizer": optimizer}

def load_model(args, model_cls, task_type):
    bnb_config = None

    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=args.dtype,
            bnb_4bit_use_double_quant=True,
        )


    model = model_cls.from_pretrained(
        args.model,
        num_labels=args.num_classes,
        cache_dir=args.cache_dir,
        resume_download=True,
        load_in_8bit=args.load_in_8bit,
        quantization_config=bnb_config,
        ignore_mismatched_sizes=True,
    )

    peft_config = None
    if args.peft_lora:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type=task_type,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        if args.do_train:
            if args.load_in_4bit or args.load_in_8bit:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) # lightning에서 안된다 그랬음
            model = get_peft_model(model, peft_config)

    # # print model summary
    # from lightning.pytorch.utilities.model_summary import ModelSummary
    # print(ModelSummary(model, max_depth=5))
    # print(type(model.transformer.encoder))
    # quit()

    return model