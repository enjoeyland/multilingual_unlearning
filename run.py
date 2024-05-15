import os
import json
import wandb
import torch
import argparse
import numpy as np
import lightning as L

from glob import glob
from datasets import load_dataset

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.strategies import FSDPStrategy, DeepSpeedStrategy
from transformers.models.mt5.modeling_mt5 import MT5Block
from torch.distributed.fsdp import MixedPrecision

from config import add_arguments
from models import MultilingualModel, ShardEnsembleModel
from dataset import sizeOfShard

def main(args, model_path=None):
    L.seed_everything(args.seed, workers=True)

    name = "/".join(args.output_dir.split("/")[4:])
    if args.do_train and args.method in ["sisa", "sisa-retain"]:
        name += f"_sd{args.shard}"

    wandb_logger = WandbLogger(
        project="multilingual-unlearning",
        group="/".join(args.output_dir.split("/")[1:4]),
        name=name,
        config=args,
    )

    if model_path:
        model = MultilingualModel.load_from_checkpoint(model_path, hparams=args)
    elif args.method == "sisa" and args.do_eval:
        models = []
        for shard in range(args.shards):
            ckpt = os.path.join(args.output_dir, f"shard{shard}-slice{args.slices-1}.ckpt")
            models.append(MultilingualModel.load_from_checkpoint(ckpt, hparams=args))
        model = ShardEnsembleModel(models, args)
    else:
        model = MultilingualModel(args)



    if args.dp_strategy == "fsdp":
        strategy = FSDPStrategy(
            auto_wrap_policy={MT5Block},
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True) if args.bf16 else None,
            sharding_strategy="FULL_SHARD",
        )
    if "deepspeed" in args.dp_strategy and args.dp_config:
        strategy = DeepSpeedStrategy(config=args.dp_config)
    else:
        strategy = args.dp_strategy

    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        save_weights_only=True,
        save_last=False,
        save_on_train_epoch_end=True,
        dirpath=args.output_dir,
        filename=args.checkpoint_name,
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=args.max_tolerance, # if eval_steps is int, 3 * (int(float eval_steps *  train size / int eval_steps)+1) 1000 20정도하면 되지 않을까?
        mode="max",
    )

    trainer = L.Trainer(
        strategy=strategy,
        devices=find_usable_cuda_devices(args.world_size),
        precision="bf16-mixed" if args.bf16 else "32-true",
        gradient_clip_val=1.0 if args.dp_strategy != "fsdp" else None,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        max_epochs=args.epochs,
        val_check_interval=args.eval_steps,
        logger=wandb_logger,
        log_every_n_steps=args.logging_steps,
        callbacks=[checkpoint_callback, early_stopping],
        default_root_dir=args.output_dir,
        reload_dataloaders_every_n_epochs=0, # for unlearning
    )

    if args.do_train:
        trainer.fit(model, datamodule=model.datamodule)

    if args.do_eval:
        trainer.validate(model, datamodule=model.datamodule)


def is_passable(args, shards_idx, slice_size, shard, sl):
    shard_idx = np.array(shards_idx[shard])
    offset = sl * slice_size
    until = (sl + 1) * slice_size if sl < args.slices - 1 else len(shard_idx)
    slice_idx = shard_idx[offset:until]
 
    retain_start_idx = len(load_dataset(
        "json",
        data_files=os.path.join(
            args.data_dir,
            f"{args.task}/forget-{args.forget_ratio}.jsonl",
        ),
    )["train"])
 
    return np.all(slice_idx >= retain_start_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    args.train_batch_size = (args.world_size if args.dp_strategy in ["auto", "ddp"] else 1) * args.per_device_batch_size * args.gradient_accumulation_steps
    args.output_dir = f".checkpoints/{args.model_name}/{args.task}/{args.method}/" + \
                    f"BS{args.train_batch_size}_LR{args.learning_rate}_W{args.warmup_ratio}_S{args.seed}"
    
    if args.method in ["sisa", "sisa-retain"]:
        args.output_dir += f"_SD{args.shards}_SL{args.slices}"

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.do_train and args.method in ["sisa", "sisa-retain"]:
        epochs = args.epochs
        do_eval = args.do_eval
        args.do_eval = False

        # Recovery
        recovery_list = glob(f"{args.output_dir}/*.ckpt")
        if recovery_list:
            recovery_list.sort()
            lastmodel = recovery_list[-1].split("/")[-1]
            shard, sl = int(lastmodel.split("-")[0].split("d")[-1]), int(lastmodel.split("-")[1].split("e")[-1].split(".")[0])
            print(f"Recovery from {lastmodel}")
            if sl == args.slices - 1:
                shard += 1
                sl = 0
            else:
                sl += 1
        else:
            shard, sl = 0, 0

        if args.method == "sisa-retain":
            # load shards_idx from splitfile
            args.sisa_output_dir = f".checkpoints/{args.model_name}/{args.task}/sisa/" + \
                f"BS{args.train_batch_size}_LR{args.learning_rate}_W{args.warmup_ratio}_S{args.seed}_SD{args.shards}_SL{args.slices}"
            splitfile = os.path.join(f'{args.sisa_output_dir}', f'shard{args.shards}-splitfile.jsonl')
            if os.path.exists(splitfile):
                with open(splitfile) as f:
                    shards_idx = f.readlines()
                shards_idx = [json.loads(shard) for shard in shards_idx]
            else:
                raise FileNotFoundError(f"Splitfile {splitfile} not found.")

        for shard in range(shard, args.shards):
            args.shard = shard
            check_passable = sl == 0
            for sl in range(sl, args.slices):
                args.sl = sl

                if check_passable and args.method == "sisa-retain":
                    # 만약 forget 데이터가 한 slice에 없으면 pass
                    shard_size = sizeOfShard(splitfile, shard)
                    slice_size = shard_size // args.slices
                    if is_passable(args, shards_idx, slice_size, shard, sl):
                        print(f"Shard {shard} slice {sl} is passable")
                        continue
                    else:
                        check_passable = False
                        print(f"Shard {shard} slice {sl} has forget data")

                args.checkpoint_name = f"shard{shard}-slice{sl}" 

                avg_epochs_per_slice = (2 * epochs / (args.slices + 1)) # See paper for explanation.
                slice_epochs = int((sl + 1) * avg_epochs_per_slice) - int(sl * avg_epochs_per_slice)
                args.epochs = slice_epochs

                if sl == 0:
                    model_path = None
                else:
                    model_path = os.path.join(args.output_dir, f"shard{shard}-slice{sl-1}.ckpt")
                    if args.method == "sisa-retain" and not os.path.exists(model_path):
                        model_path = os.path.join(args.sisa_output_dir, f"shard{shard}-slice{sl-1}.ckpt")

                main(args, model_path=model_path)
            else:
                sl = 0
                wandb.finish()
        else:
            args.do_eval = do_eval
            if args.do_eval:
                args.do_train = False
                del args.shard, args.sl
                main(args)
    else:
        main(args)
