import torch
import pandas as pd

from lightning.pytorch.callbacks import Callback, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class Callbacks:
    def __init__(self, args):
        self.output_dir = args.output_dir

        self.max_tolerance = args.max_tolerance


        if args.task == "flores":
            self.monitor = "val/forget_xma"
            self.mode = "min"
            self.filename = "fxma={val/forget_xma:.4f}-fxppl={val/forget_xppl:.2f}-vxppl={val/val_xppl:.2f}"
        elif args.task == "xnli":
            self.monitor = "val_accuracy"
            self.mode = "max"
            self.filename = "best"
        
        if args.method in ["sisa", "sisa-retain"]:
            self.filename = f"shard{args.shard}-slice{args.sl}"
        elif args.method in ["negtaskvector"]:
            self.mode = "max"

    def get_checkpoint_callback(self):
        return ModelCheckpoint(
            monitor=self.monitor,
            mode=self.mode,
            save_top_k=1,
            save_weights_only=True,
            save_last=False,
            # save_on_train_epoch_end=True,
            dirpath=self.output_dir,
            filename=self.filename,
            verbose=True,
            auto_insert_metric_name=False,
        )
    
    def get_early_stopping(self):
        return EarlyStopping(
            monitor=self.monitor,
            mode=self.mode,
            patience=self.max_tolerance,
            verbose=True,
        )

class CustomMetricTracker(Callback):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        if pl_module.hparams.task == "flores":
            forget_ma = {k: v for k, v in trainer.logged_metrics.items() if "ma" in k and "forget" in k and "x" not in k}
            forget_xma = torch.stack([forget_ma[k] for k in forget_ma.keys()]).mean().item()
        
            forget_ppl = {k: v for k, v in trainer.logged_metrics.items() if "ppl" in k and "forget" in k and "x" not in k}
            forget_xppl = torch.stack([forget_ppl[k] for k in forget_ppl.keys()]).mean().item()
            
            val_ppl = {k: v for k, v in trainer.logged_metrics.items() if "ppl" in k and "forget" not in k and "x" not in k}
            val_xppl = torch.stack([val_ppl[k] for k in val_ppl.keys()]).mean().item()
            
            self.log_dict({"val/val_xppl": val_xppl, "val/forget_xppl": forget_xppl, "val/forget_xma": forget_xma})

    def on_test_epoch_end(self, trainer, pl_module):
        if pl_module.hparams.task == "flores":
            test_ma = {k: v for k, v in trainer.logged_metrics.items() if "ma" in k and "forget" not in k and "x" not in k}
            test_xma = torch.stack([test_ma[k] for k in test_ma.keys()]).mean().item()

            forget_ma = {k: v for k, v in self.trainer.logged_metrics.items() if "ma" in k and "forget" in k and "x" not in k}
            forget_xma = torch.stack([forget_ma[k] for k in forget_ma.keys()]).mean().item()
            
            self.log_dict({"test/test_xma": test_xma, "test/forget_xma": forget_xma})

    def on_validation_end(self, trainer, pl_module):
        if pl_module.hparams.task == "flores":
            forget_ma_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" in k and "ma" in k})
            forget_ma_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            forget_ma_df.to_csv(f"{self.output_dir}/val_forget_ma.csv", index=False, mode="a")
            
            ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" not in k and "ppl" in k})
            ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            ppl_df.to_csv(f"{self.output_dir}/val_ppl.csv", index=False, mode="a")
            
            forget_ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" in k and "ppl" in k})
            forget_ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            forget_ppl_df.to_csv(f"{self.output_dir}/val_forget_ppl.csv", index=False, mode="a")
        else:
            raise ValueError(f"Task {pl_module.hparams.task} not supported.")

    def on_test_end(self, trainer, pl_module):
        if pl_module.hparams.task == "flores":
            ma_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" not in k and "ma" in k})
            ma_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            ma_df.to_csv(f"{self.output_dir}/test_ma.csv", index=False)
        
            forget_ma_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" in k and "ma" in k})
            forget_ma_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            forget_ma_df.to_csv(f"{self.output_dir}/test_forget_ma.csv", index=False)
        
            ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" not in k and "ppl" in k})
            ppl_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            ppl_df.to_csv(f"{self.output_dir}/test_ppl.csv", index=False)
            forget_ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" in k and "ppl" in k})
            forget_ppl_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            forget_ppl_df.to_csv(f"{self.output_dir}/test_forget_ppl.csv", index=False)
        else:
            raise ValueError(f"Task {pl_module.hparams.task} not supported.")


class CustomRichProgressBar(RichProgressBar):
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items