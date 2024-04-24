import os
import json
import wandb
import argparse
import numpy as np
import lightning as L

from datasets import load_dataset

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.accelerators import find_usable_cuda_devices

from models import MultilingualModel
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
    parser.add_argument("--train", action="store_true", help="Perform SISA training on the shard")
    parser.add_argument("--test", action="store_true", help="Compute shard predictions")
    
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bf16", action="store_true")

    parser.add_argument("--optimizer", default="sgd", help="Optimizer, default sgd")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate, default 0.001")

    parser.add_argument("--epochs", default=20, type=int, help="Train for the specified number of epochs, default 20")
    parser.add_argument("--world_size", default=1, type=int, help="Number of GPUs to use, default 1")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size, default 16")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--evaluation_strategy", default="steps", choices=["steps", "epoch", "no"], help="Evaluation strategy, default steps")
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--max_tolerance", type=int, default=3)

    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--save_strategies", default="epoch", choices=["epoch", "no"], help="Save strategies, default epoch")

def main(args, model_path = None):
    L.seed_everything(args.seed, workers=True)

    name = "/".join(args.output_dir.split("/")[4:])
    if args.method in ["sisa", "sisa-retain"]:
        name += f"_sd{args.shard}"
    
    wandb_logger = WandbLogger(
        project="multilinugal-unlearning",
        group="/".join(args.output_dir.split("/")[1:4]),
        name=name,
        config=args,
    )
    if model_path is None:
        model = MultilingualModel(args)
    else:
        model = MultilingualModel.load_from_checkpoint(model_path, hparams=args)

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
        default_root_dir=args.output_dir,
        devices=find_usable_cuda_devices(args.world_size),
        precision="bf16-mixed" if args.bf16 else "32-true",
        max_epochs=args.epochs,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        log_every_n_steps=args.logging_steps,
        val_check_interval=args.eval_steps,
        callbacks=[checkpoint_callback, early_stopping],
        accumulate_grad_batches=args.gradient_accumulation_steps,
    )
    trainer.fit(model)

def is_passable(args, shards_idx, slice_size,  shard, sl):
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

    args.train_batch_size = args.world_size * args.batch_size * args.gradient_accumulation_steps

    os.makedirs(args.cache_dir, exist_ok=True)

    if args.method in ["sisa", "sisa-retain"]:
        args.output_dir = f".checkpoints/{args.model_name}/{args.task}/{args.method}/" + \
                    f"BS{args.train_batch_size}_LR{args.learning_rate}_E{args.epochs}_SD{args.shards}_SL{args.slices}_S{args.seed}"
        os.makedirs(args.output_dir, exist_ok=True)
        epochs = args.epochs

        if args.method == "sisa-retain":
            splitfile = os.path.join(f'{args.output_dir}', f'shard{args.shards}-splitfile.jsonl')
            if os.path.exists(splitfile):
                with open(splitfile) as f:
                    shards_idx = f.readlines()
                shards_idx = [json.loads(shard) for shard in shards_idx]
            else:
                raise FileNotFoundError(f"Splitfile {splitfile} not found.")

        for shard in range(args.shards):
            for sl in range(args.slices):
                if args.method == "sisa-retain":
                    # 만약 forget 데이터가 한 slice에 없으면 pass
                    shard_size = sizeOfShard(splitfile, shard)
                    slice_size = shard_size // args.slices
                    if is_passable(args, shards_idx, slice_size, shard, sl):
                        print(f"Shard {shard} slice {sl} is passable")
                        continue
                    args.checkpoint_name = f"shard{shard}-slice{sl}-retain"
                elif args.method == "sisa":
                    args.checkpoint_name = f"shard{shard}-slice{sl}" 

                avg_epochs_per_slice = (2 * epochs / (args.slices + 1)) # See paper for explanation.
                slice_epochs = int((sl + 1) * avg_epochs_per_slice) - int(sl * avg_epochs_per_slice)
                
                args.shard = shard
                args.sl = sl
                args.epochs = slice_epochs
                if sl == 0:
                    model_path = None
                else:
                    model_path = os.path.join(args.output_dir, f"shard{shard}-slice{sl-1}.ckpt")
                    if args.method == "sisa-retain":
                        model_path = os.path.join(args.output_dir, f"shard{shard}-slice{sl-1}-retain.ckpt")
                        if not os.path.exists(model_path):
                            model_path = os.path.join(args.output_dir, f"shard{shard}-slice{sl-1}.ckpt")

                main(args, model_path=model_path)
            else:
                wandb.finish()

    else:
        main(args)    