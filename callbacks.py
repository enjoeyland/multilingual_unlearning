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

            if args.method == "negtaskvector":
                self.mode = "max"
                if args.negtv_fit == "retain":
                    self.monitor = "val/retain_xma"
                    self.filename = "rxma={val/retain_xma:.4f}-rxppl={val/retain_xppl:.2f}-vxppl={val/val_xppl:.2f}"

        elif args.task == "xnli":
            self.monitor = "val_accuracy"
            self.mode = "max"
            self.filename = "best"
        
        if "sisa "in args.method:
            self.filename = f"shard{args.shard}-slice{args.sl}"
        elif args.method == "negtaskvector":
            if args.negtv_fit == "forget":
                self.filename = f"forget_{self.filename}"
            elif args.negtv_fit == "retain":
                self.filename = f"retain{args.retain_multiplier}_{self.filename}"

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
    def __init__(self, args):
        self.args = args

        self.output_dir = args.output_dir

        self.target = "forget"
        if args.negtv_fit == "retain":
            self.target = "retain"

    def _fit_write_params(self, f):
        f.write(f"\n\n{'='*50}\n")
        f.write(f"train\n")
        f.write(f"method: {self.args.method}\n")
        f.write(f"negtv_fit: {self.args.negtv_fit}\n")
        f.write(f"forget_num: {self.args.forget_num}\n")
        f.write(f"retention_multiplier: {self.args.retain_multiplier}\n")

    def _validate_write_params(self, f):
        f.write(f"\n\n{'='*50}\n")
        f.write(f"eval\n")
        f.write(f"forget_scaling_coef: {self.args.forget_scaling_coef}\n")
        f.write(f"retain_scaling_coef: {self.args.retain_scaling_coef}\n")

    def on_fit_start(self, trainer, pl_module):
        with open(f"{self.output_dir}/val_target_ma.csv", "a") as f:
            self._fit_write_params(f)
        with open(f"{self.output_dir}/val_ppl.csv", "a") as f:
            self._fit_write_params(f)
        with open(f"{self.output_dir}/val_target_ppl.csv", "a") as f:
            self._fit_write_params(f)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        if pl_module.hparams.task == "flores":
            target_ma = {k: v for k, v in trainer.logged_metrics.items() if "ma" in k and self.target in k and "x" not in k}
            target_xma = torch.stack([target_ma[k] for k in target_ma.keys()]).mean().item()

            val_ppl = {k: v for k, v in trainer.logged_metrics.items() if "ppl" in k and self.target not in k and "x" not in k}
            val_xppl = torch.stack([val_ppl[k] for k in val_ppl.keys()]).mean().item()

            target_ppl = {k: v for k, v in trainer.logged_metrics.items() if "ppl" in k and self.target in k and "x" not in k}
            target_xppl = torch.stack([target_ppl[k] for k in target_ppl.keys()]).mean().item()

            self.log_dict({f"val/{self.target}_xma": target_xma, "val/val_xppl": val_xppl, f"val/{self.target}_xppl": target_xppl}, sync_dist=True)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.args.task == "flores":
            test_ma = {k: v for k, v in trainer.logged_metrics.items() if "ma" in k and "forget" not in k and "x" not in k}
            test_xma = torch.stack([test_ma[k] for k in test_ma.keys()]).mean().item()

            forget_ma = {k: v for k, v in trainer.logged_metrics.items() if "ma" in k and "forget" in k and "x" not in k}
            forget_xma = torch.stack([forget_ma[k] for k in forget_ma.keys()]).mean().item()
            
            self.log_dict({"test/test_xma": test_xma, "test/forget_xma": forget_xma})

    def on_validation_end(self, trainer, pl_module):
        if self.args.task == "flores":
            if trainer.state.fn == "validate":
                with open(f"{self.output_dir}/val_target_ma.csv", "a") as f:
                    self._validate_write_params(f)
                with open(f"{self.output_dir}/val_ppl.csv", "a") as f:
                    self._validate_write_params(f)
                with open(f"{self.output_dir}/val_target_ppl.csv", "a") as f:
                    self._validate_write_params(f)
        
            target_ma_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if self.target in k and "ma" in k})
            target_ma_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            target_ma_df.to_csv(f"{self.output_dir}/val_target_ma.csv", index=False, mode="a")
            
            ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if self.target not in k and "ppl" in k})
            ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            ppl_df.to_csv(f"{self.output_dir}/val_ppl.csv", index=False, mode="a")
            
            target_ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if self.target in k and "ppl" in k})
            target_ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            target_ppl_df.to_csv(f"{self.output_dir}/val_target_ppl.csv", index=False, mode="a")
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