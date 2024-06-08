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
            self.filename = "fxma={val/forget_xma:.4f}-fxppl={val/forget_xppl:.2f}-xppl={val/val_xppl:.2f}"

            if args.method == "negtaskvector":
                self.mode = "max"
                if args.negtv_fit == "retain":
                    self.monitor = "val/retain_xma"
                    self.filename = "rxma={val/retain_xma:.4f}-rxppl={val/retain_xppl:.2f}-xppl={val/val_xppl:.2f}"
        
        elif "bmlama" in args.task:
            self.monitor = "val/forget_xpa"
            self.mode = "min"
            self.filename = "fxpa={val/forget_xpa:.4f}-fxppl={val/forget_xppl:.2f}-xppl={val/val_xppl:.2f}"
            
            if args.method == "negtaskvector":
                self.mode = "max"
                if args.negtv_fit == "retain":
                    self.monitor = "val/retain_xma"
                    self.filename = "rxpa={val/retain_xpa:.4f}-rxppl={val/retain_xppl:.2f}-xppl={val/val_xppl:.2f}"
        
        elif args.task == "xnli":
            self.monitor = "val_accuracy"
            self.mode = "max"
            self.filename = "best"
        else:
            raise ValueError(f"Task {args.task} not supported.")
        

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

    def _write_params(self, f, stage):
        f.write(f"\n\n{'='*50}\n")
        f.write(f"task: {self.args.task}\n")
        f.write(f"method: {self.args.method}\n")
        f.write(f"hyparams: {self.output_dir.split('/')[-1]}\n")

        f.write(f"{stage}\n")
        f.write(f"negtv_fit: {self.args.negtv_fit}\n")
        f.write(f"forget_num: {self.args.forget_num}\n")
        f.write(f"retain_multiplier: {self.args.retain_multiplier}\n")

        if not stage=="fit":
            f.write(f"forget_scaling_coef: {self.args.forget_scaling_coef}\n")
            f.write(f"retain_scaling_coef: {self.args.retain_scaling_coef}\n")

    def on_fit_start(self, trainer, pl_module):
        if self.args.task == "flores":
            with open(f"{self.output_dir}/target_ma.csv", "a") as f:
                self._write_params(f, "fit")
            with open(f"{self.output_dir}/ppl.csv", "a") as f:
                self._write_params(f, "fit")
            with open(f"{self.output_dir}/target_ppl.csv", "a") as f:
                self._write_params(f, "fit")

        elif "bmlama" in self.args.task:
            with open(f"{self.output_dir}/target_pa.csv", "a") as f:
                self._write_params(f, "fit")
            with open(f"{self.output_dir}/target_sent_ppl.csv", "a") as f:
                self._write_params(f, "fit")
            with open(f"{self.output_dir}/ppl.csv", "a") as f:
                self._write_params(f, "fit")
            with open(f"{self.output_dir}/target_ppl.csv", "a") as f:
                self._write_params(f, "fit")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        if self.args.task == "flores":
            target_ma = {k: v for k, v in trainer.logged_metrics.items() if "train" not in k and "ma" in k and self.target in k and "x" not in k}
            target_xma = torch.stack([target_ma[k] for k in target_ma.keys()]).mean().item()

            val_ppl = {k: v for k, v in trainer.logged_metrics.items() if "train" not in k and "ppl" in k and self.target not in k and "x" not in k}
            val_xppl = torch.stack([val_ppl[k] for k in val_ppl.keys()]).mean().item()

            target_ppl = {k: v for k, v in trainer.logged_metrics.items() if "train" not in k and "ppl" in k and self.target in k and "x" not in k}
            target_xppl = torch.stack([target_ppl[k] for k in target_ppl.keys()]).mean().item()

            self.log_dict({f"val/{self.target}_xma": target_xma, "val/val_xppl": val_xppl, f"val/{self.target}_xppl": target_xppl}, on_epoch=True , sync_dist=True)

        elif "bmlama" in self.args.task:
            target_pa = {k: v for k, v in trainer.logged_metrics.items() if "train" not in k and "pa" in k and self.target in k and "x" not in k}
            target_xpa = torch.stack([target_pa[k] for k in target_pa.keys()]).mean().item()

            val_ppl = {k: v for k, v in trainer.logged_metrics.items() if "train" not in k and "ppl" in k and self.target not in k and "x" not in k}
            val_xppl = torch.stack([val_ppl[k] for k in val_ppl.keys()]).mean().item()

            target_ppl = {k: v for k, v in trainer.logged_metrics.items() if "train" not in k and "ppl" in k and self.target in k and "x" not in k}
            target_xppl = torch.stack([target_ppl[k] for k in target_ppl.keys()]).mean().item()

            self.log_dict({f"val/{self.target}_xpa": target_xpa, "val/val_xppl": val_xppl, f"val/{self.target}_xppl": target_xppl}, on_epoch=True, sync_dist=True)

    def on_validation_end(self, trainer, pl_module):
        if self.args.task == "flores":
            if trainer.state.fn == "validate":
                with open(f"{self.output_dir}/target_ma.csv", "a") as f:
                    self._write_params(f, "eval")
                with open(f"{self.output_dir}/ppl.csv", "a") as f:
                    self._write_params(f, "eval")
                with open(f"{self.output_dir}/target_ppl.csv", "a") as f:
                    self._write_params(f, "eval")
        
            target_ma_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if self.target in k and "ma" in k})
            target_ma_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            target_ma_df.to_csv(f"{self.output_dir}/target_ma.csv", index=False, mode="a")
            
            ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if self.target not in k and "ppl" in k})
            ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            ppl_df.to_csv(f"{self.output_dir}/ppl.csv", index=False, mode="a")
            
            target_ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if self.target in k and "ppl" in k})
            target_ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            target_ppl_df.to_csv(f"{self.output_dir}/target_ppl.csv", index=False, mode="a")

        elif "bmlama" in self.args.task:
            if trainer.state.fn == "validate":
                with open(f"{self.output_dir}/target_pa.csv", "a") as f:
                    self._write_params(f, "eval")
                with open(f"{self.output_dir}/target_sent_ppl.csv", "a") as f:
                    self._write_params(f, "eval")
                with open(f"{self.output_dir}/ppl.csv", "a") as f:
                    self._write_params(f, "eval")
                with open(f"{self.output_dir}/target_ppl.csv", "a") as f:
                    self._write_params(f, "eval")
            
            target_pa_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" in k and "pa" in k})
            target_pa_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            target_pa_df.to_csv(f"{self.output_dir}/target_pa.csv", index=False, mode="a")
            
            target_sent_ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" in k and "sent_ppl" in k})
            target_sent_ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            target_sent_ppl_df.to_csv(f"{self.output_dir}/target_sent_ppl.csv", index=False, mode="a")

            ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if self.target not in k and "ppl" in k})
            ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            ppl_df.to_csv(f"{self.output_dir}/ppl.csv", index=False, mode="a")
            
            target_ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if self.target in k and "ppl" in k})
            target_ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            target_ppl_df.to_csv(f"{self.output_dir}/target_ppl.csv", index=False, mode="a")

        else:
            raise ValueError(f"Task {self.args.task} not supported.")

    def on_test_end(self, trainer, pl_module):
        with open(f"{self.output_dir}/ma.csv", "a") as f:
            self._write_params(f, "test")
        with open(f"{self.output_dir}/target_ma.csv", "a") as f:
            self._write_params(f, "test")
        with open(f"{self.output_dir}/ppl.csv", "a") as f:
            self._write_params(f, "test")
        with open(f"{self.output_dir}/target_ppl.csv", "a") as f:
            self._write_params(f, "test")

        if self.args.task == "flores":
            ma_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" not in k and "ma" in k})
            ma_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            ma_df.to_csv(f"{self.output_dir}/ma.csv", index=False)
        
            forget_ma_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" in k and "ma" in k})
            forget_ma_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            forget_ma_df.to_csv(f"{self.output_dir}/target_ma.csv", index=False)
        
            ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" not in k and "ppl" in k})
            ppl_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            ppl_df.to_csv(f"{self.output_dir}/ppl.csv", index=False)
           
            forget_ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" in k and "ppl" in k})
            forget_ppl_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            forget_ppl_df.to_csv(f"{self.output_dir}/target_ppl.csv", index=False)

        elif "bmlama" in self.args.task:
            pa_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" not in k and "pa" in k})
            pa_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            pa_df.to_csv(f"{self.output_dir}/pa.csv", index=False)

            forget_pa_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" in k and "pa" in k})
            forget_pa_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            forget_pa_df.to_csv(f"{self.output_dir}/target_pa.csv", index=False)

            sent_ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" not in k and "sent_ppl" in k})
            sent_ppl_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            sent_ppl_df.to_csv(f"{self.output_dir}/target_ppl.csv", index=False)

            forget_sent_ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" in k and "sent_ppl" in k})
            forget_sent_ppl_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            forget_sent_ppl_df.to_csv(f"{self.output_dir}/target_sent_ppl.csv", index=False)

            ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" not in k and "ppl" in k})
            ppl_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            ppl_df.to_csv(f"{self.output_dir}/ppl.csv", index=False)
           
            forget_ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" in k and "ppl" in k})
            forget_ppl_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            forget_ppl_df.to_csv(f"{self.output_dir}/target_ppl.csv", index=False)
        else:
            raise ValueError(f"Task {self.args.task} not supported.")
