import os
import torch
import deepspeed
import lightning as L

from torchmetrics import Accuracy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from transformers.utils import logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pytorch_lightning.core.saving import save_hparams_to_yaml

from datamodules import XNLIDataModule, FLORESDataModule, BMLAMADataModule

logging.get_logger("transformers").setLevel(logging.ERROR)

class MultilingualModel(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        save_hparams_to_yaml(os.path.join(hparams.output_dir, "hparams.yaml"), hparams)
        

        self.tokenizer = AutoTokenizer.from_pretrained(hparams.model, cache_dir=hparams.cache_dir, resume_download=True)
        if hparams.task == "xnli":
            self.datamodule = XNLIDataModule(hparams, self.tokenizer)
        elif hparams.task == "flores":
            self.datamodule = FLORESDataModule(hparams, self.tokenizer)
        elif "bmlama" in hparams.task:
            self.datamodule = BMLAMADataModule(hparams, self.tokenizer)
        else:
            raise NotImplementedError(f"Task {hparams.task} not implemented.")
        self.model = None

        self.accuracy = Accuracy(task="multiclass", num_classes=self.tokenizer.vocab_size, ignore_index=-100)


    def configure_model(self):
        if self.model is not None:
            return

        if "mt5" in self.hparams.model_name:
            target_modules = ["k_proj", "q_proj", "v_proj", "o_proj"]
        elif "bloom" in self.hparams.model_name:
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif "xglm" in self.hparams.model_name:
            target_modules = ["k_proj", "q_proj", "v_proj", "out_proj"]
        else:
            raise ValueError(f"Model {self.hparams.model} not supported.")
        
        if self.hparams.task == "xnli":
            self.model = self._load_model("SEQ_CLS", target_modules)
        elif self.hparams.task == "flores" or "bmlama" in self.hparams.task:
            self.model = self._load_model("CAUSAL_LM", target_modules)
        else:
            # self.model = self._load_model("CAUSAL_LM", target_modules)
            raise ValueError(f"Task {self.hparams.task} not supported.")

    def forward(self, input_ids, attention_mask=None, labels=None, **inputs):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        dataset_name = self.datamodule.dataset_names["train"][self.current_epoch % len(self.datamodule.dataset_names["train"])]
        self._log_metrics("train", batch, outputs, loss, dataset_name)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss
        dataset_name = self.datamodule.dataset_names["valid"][dataloader_idx]
        self._log_metrics("valid", batch, outputs, loss, dataset_name)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss
        dataset_name = self.datamodule.dataset_names["test"][dataloader_idx]
        self._log_metrics("test", batch, outputs, loss, dataset_name)
        return loss

    def _log_metrics(self, split, batch, outputs, loss, dataset_name):
        metrics = { f"{dataset_name}_loss": loss }
        if self.hparams.task in ["xnli"]:
            preds = outputs.logits.argmax(dim=-1)
            accuracy = (preds == batch["labels"]).float().mean()
            metrics[f"{dataset_name}_accuracy"] = accuracy
        elif self.hparams.task in ["flores"]:
            ppl = torch.exp(loss)
            metrics[f"{dataset_name}_ppl"] = ppl

            if self.hparams.method != "finetune" and self.hparams.fit_target in dataset_name or split == "test":
                ma = self._validation_ma(batch)
                metrics[f"{dataset_name}_ma"] = ma

        elif "bmlama" in self.hparams.task:
            if self.hparams.method != "finetune" and self.hparams.fit_target in dataset_name or split == "test":
                pa, sent_loss = self._validation_pa(batch, dataset_name)
                sent_ppl = torch.exp(sent_loss)
                metrics[f"{dataset_name}_pa"] = pa
                metrics[f"{dataset_name}_sent_ppl"] = sent_ppl
            else:
                sent_loss = self._validation_sent_loss_only(batch, dataset_name)
                sent_ppl = torch.exp(sent_loss)
                metrics[f"{dataset_name}_sent_ppl"] = sent_ppl


        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
    
    def _validation_sent_loss_only(self, batch, dataset_name):
        lang = dataset_name.split("/")[1].split("_")[0]
        batch_size = batch["input_ids"].size(0)
        losses = []

        for i in range(batch_size):
            prompt = batch["prompt"][i]
            prompt_new = prompt.replace("<mask>", batch["answers"][i])
            model_input = self.tokenizer(prompt_new, return_tensors='pt').to(self.device)
            output = self.model(**model_input)
            if lang == "zh":
                logits = output['logits'][0, :-1] 
                token_ids = model_input['input_ids'][0, 1:]
            else:
                logits = output['logits'][0, :-2] 
                token_ids = model_input['input_ids'][0, 1:-1]
            loss = torch.nn.CrossEntropyLoss(reduction='mean')(logits, token_ids)
            losses.append(loss)

        loss = torch.stack(losses).mean()
        return loss

    def _validation_pa(self, batch, dataset_name):
        lang = dataset_name.split("/")[-1].split("_")[-1]
        batch_size = batch["input_ids"].size(0)
        corr, tot = 0, 0
        losses = []

        for i in range(batch_size):
            prompt = batch["prompt"][i]
            answer_pred_probs = dict()
            for j in range(len(batch["candidates"])):
                cand = batch["candidates"][j][i]
                if cand == "":
                    continue
                prompt_new = prompt.replace("<mask>", cand)
                model_input = self.tokenizer(prompt_new, return_tensors='pt').to(self.device)
                output = self.model(**model_input)
                
                if lang == "zh":
                    logits = output['logits'][0, :-1] 
                    token_ids = model_input['input_ids'][0, 1:]
                else:
                    logits = output['logits'][0, :-2] 
                    token_ids = model_input['input_ids'][0, 1:-1]
                
                answer_pred_probs[cand] = torch.nn.CrossEntropyLoss(reduction='mean')(logits, token_ids)

            # Precision@k (k=1)
            top1 = sorted(answer_pred_probs.items(), key=lambda x: x[1], reverse=False)[0][0]
            if top1 == batch["answers"][i]:
                corr += 1
            tot += 1

            losses.append(answer_pred_probs[batch["answers"][i]])

        acc = corr / tot
        loss = torch.stack(losses).mean()
        return acc, loss
    
    def _validation_ma(self, batch):
        labels, preds = [], []
        # Change the sliding direction based on the padding side
        if self.tokenizer.padding_side == "left":
            start, end, step = self.hparams.max_length-1, 0, -1
        else:
            start, end, step = 1, self.hparams.max_length, 1

        for i in range(start, end, step):
            label = batch["labels"][..., i]
            prompt = batch["input_ids"][..., :i]
            att_mask = batch["attention_mask"][..., :i]
            # break if only padding tokens are left for all seqs
            if all(label == -100): 
                break

            try:
                pred = self.model.generate(input_ids=prompt,
                                           attention_mask=att_mask,
                                           max_length=i+1)[..., -1]
            except IndexError:  # if batch == 1
                pred = self.model.generate(input_ids=torch.squeeze(prompt),
                                            attention_mask=torch.squeeze(att_mask),
                                            max_length=i+1).squeeze()[-1]
                
            labels.append(torch.squeeze(label))
            preds.append(torch.squeeze(pred))

        preds = torch.stack(preds, dim=1)
        labels = torch.stack(labels, dim=1)
        return self.accuracy(preds, labels)

    def configure_optimizers(self):
        if "deepspeed" in self.hparams.dp_strategy and "offload" not in self.hparams.dp_strategy:
            optimizer = deepspeed.ops.adam.FusedAdam(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01, 
                                adam_w_mode=(self.hparams.optimizer == "adamw")) 
        elif "deepspeed" in self.hparams.dp_strategy and "offload" in self.hparams.dp_strategy:
            optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01, 
                                adamw_mode=(self.hparams.optimizer == "adamw"))
        elif self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
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
    
    def _load_model(self, task_type, target_modules=None):
        model_kwargs = {}

        if "deepspeed" in self.hparams.dp_strategy:
            model_kwargs["ignore_mismatched_sizes"] = True
        
        bnb_config = None
        if self.hparams.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if self.hparams.bf16 else "float",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
        if self.hparams.load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        if task_type == "SEQ_CLS":
            LM = AutoModelForSequenceClassification
            model_kwargs["num_labels"] = self.datamodule.num_classes
        elif task_type == "CAUSAL_LM":
            LM = AutoModelForCausalLM
        else:
            raise ValueError(f"Task type {task_type} not supported.")

        model = LM.from_pretrained(
            self.hparams.model,
            cache_dir=self.hparams.cache_dir,
            resume_download=True,
            **model_kwargs,
        )

        lora_config = None
        if self.hparams.use_lora or self.hparams.use_qlora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                task_type=task_type,
                target_modules="all-linear" if self.hparams.use_qlora else target_modules,
                inference_mode=not self.hparams.do_train
            )
            
            if self.hparams.load_in_4bit or self.hparams.load_in_8bit:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) # lightning에서 안된다 그랬음
            model = get_peft_model(model, lora_config)

        if self.hparams.do_train:
            model.train()

        # # print model summary
        # print(model)
        # from lightning.pytorch.utilities.model_summary import ModelSummary
        # print(ModelSummary(model, max_depth=5))
        # quit()

        return model


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
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return {"optimizer": optimizer}

