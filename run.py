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
from lightning.pytorch.strategies import FSDPStrategy
from transformers.models.mt5.modeling_mt5 import MT5Block
from torch.distributed.fsdp import MixedPrecision

from models import MultilingualModel, ShardEnsembleModel
from dataset import sizeOfShard


def add_arguments(parser):
    # Model arguments
    parser.add_argument("--model_name", default="mt5-base", help="Model name, default mt5-base")
    parser.add_argument("--model", default="google/mt5-base", help="Model to use, default google/mt5-base")
    parser.add_argument("--method", default="original", choices=["original", "sisa", "sisa-retain"], help="Method to use, default original")
    parser.add_argument("--cache_dir", help="Location of the cache, default None")

    # Data arguments
    parser.add_argument("--task", default="xnli", help="Name of the task")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--forget_ratio", type=float, default=0.01, help="Forget ratio, default 0.01")
    parser.add_argument("--data_dir", default="data/", help="Location of the data, default data/")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers to use, default 4")

    # Sharding arguments
    parser.add_argument("--shards", default=5, type=int, help="Number of shards to use, default 5")
    parser.add_argument("--slices", default=1, type=int, help="Number of slices to use, default 1")


    # Training arguments
    parser.add_argument("--do_train", action="store_true", help="Perform training")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dp_strategy", default="auto", choices=["auto", "ddp", "fsdp"], help="Distributed training strategy, default auto")
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"], help="Optimizer to use, default adamw")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate, default 0.001")
    parser.add_argument("--lr_scheduler_type", default="linear", choices=["linear", "cosine"], help="Learning rate scheduler type, default linear")
    parser.add_argument("--warmup_ratio", default=0.1, type=float, help="Warmup ratio, default 0.1")

    parser.add_argument("--epochs", default=20, type=int, help="Train for the specified number of epochs, default 20")
    parser.add_argument("--world_size", default=1, type=int, help="Number of GPUs to use, default 1")
    parser.add_argument("--per_device_batch_size", default=16, type=int, help="Batch size, default 16")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--evaluation_strategy", default="steps", choices=["steps", "epoch", "no"], help="Evaluation strategy, default steps")
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--max_tolerance", type=int, default=3)

    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--checkpoint_name", type=str, default="best")

    parser.add_argument("--do_eval", action="store_true", help="Perform evaluation on the validation set")
    parser.add_argument("--do_test", action="store_true", help="Perform evaluation on the test set")

def main(args, model_path=None):
    L.seed_everything(args.seed, workers=True)

    name = "/".join(args.output_dir.split("/")[4:])
    if args.do_train and args.method in ["sisa", "sisa-retain"]:
        name += f"_sd{args.shard}"

    wandb_logger = WandbLogger(
        project="multilinugal-unlearning",
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

    # # print model summary
    # from lightning.pytorch.utilities.model_summary import ModelSummary
    # print(ModelSummary(model, max_depth=5))
    # print(type(model.model.transformer.encoder))
    # quit()

    if args.dp_strategy == "fsdp":
        strategy = FSDPStrategy(
            auto_wrap_policy={MT5Block},
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True) if args.bf16 else None,
            sharding_strategy="FULL_SHARD",
        )        
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
        patience=args.max_tolerance,
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
            f"_{args.task}/forget-{args.forget_ratio}.jsonl",
        ),
    )["train"])
 
    return np.all(slice_idx >= retain_start_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    assert 2 * args.epochs >= args.slices + 1, "Not enough epochs per slice"


    args.train_batch_size = (args.world_size if args.dp_strategy != "fsdp" else 1) * args.per_device_batch_size * args.gradient_accumulation_steps
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
