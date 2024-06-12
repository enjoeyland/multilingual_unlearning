import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import wandb
import torch
import argparse
import numpy as np
import lightning as L

from glob import glob
from datasets import load_dataset

from lightning.pytorch.loggers import WandbLogger

from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.strategies import FSDPStrategy, DeepSpeedStrategy
from transformers.models.mt5.modeling_mt5 import MT5Block
from torch.distributed.fsdp import MixedPrecision

from config import add_arguments
from models import MultilingualModel, ShardEnsembleModel
from callbacks import Callbacks, CustomMetricTracker

def main(args, model_path=None):
    L.seed_everything(args.seed, workers=True)

    name = "/".join(args.output_dir.split("/")[4:])
    if args.do_train and "sisa" in args.method:
        name += f"_sd{args.shard}"

    wandb_logger = WandbLogger(
        project="multilingual-unlearning",
        group="/".join(args.output_dir.split("/")[1:4]),
        name=name,
        config=args,
        mode=args.wandb_mode,
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
        if "mt5" in args.model_name:
            strategy = FSDPStrategy(
                auto_wrap_policy={MT5Block},
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True) if args.bf16 else None,
                sharding_strategy="FULL_SHARD",
            )
        else:
            raise NotImplementedError(f"FS-DP is not implemented for {args.model_name}")
    else:
        strategy = args.dp_strategy

    # Add this line to solve AttributeError: 'PeftModelForSequenceClassification' object has no attribute 'base_model'
    # which is caused by Initiate deepspeed both in lightning and peft
    if "deepspeed" in args.dp_strategy and args.use_lora:
        from contextlib import contextmanager

        @contextmanager
        def model_sharded_context(self):
            yield
        DeepSpeedStrategy.model_sharded_context = model_sharded_context


    cb = Callbacks(args)
    callbacks = [
        CustomMetricTracker(args),
        cb.get_checkpoint_callback(),
        cb.get_early_stopping(),
    ]

    trainer = L.Trainer(
        strategy=strategy,
        devices=find_usable_cuda_devices(args.world_size),
        precision="bf16-mixed" if args.bf16 else "32-true",
        gradient_clip_val=1.0 if args.dp_strategy != "fsdp" else None,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        max_epochs=args.epochs,
        val_check_interval=args.eval_steps,
        logger=wandb_logger,
        # logger=csv_logger,
        log_every_n_steps=args.logging_steps,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
        reload_dataloaders_every_n_epochs=0, # for unlearning
        num_sanity_val_steps=0,
    )

    if args.do_train:
        assert args.method != "negtaskvector", "Negtaskvector method is not supported for training"
        trainer.fit(model, datamodule=model.datamodule)

    if args.do_eval:
        assert args.method != "finetune", "Finetune method is not supported for evaluation"
        if args.method == "negtaskvector":
            model = create_negtaskvector_model(args)
        trainer.validate(model, datamodule=model.datamodule)
        
    if args.do_test:
        assert args.method != "finetune", "Finetune method is not supported for evaluation"
        if args.method == "negtaskvector":
            model = create_negtaskvector_model(args)
        trainer.test(model, datamodule=model.datamodule)

def create_negtaskvector_model(args):
    from task_vectors import TaskVector
    pretraind_model = MultilingualModel(args)
    pretraind_model.configure_model()

    saved_ckpt = glob(f"{args.output_dir}/*.ckpt")
    saved_ckpt = [item for item in saved_ckpt if "negtv" not in item.split("/")[-1]]
    
    forget_ckpt = [item for item in saved_ckpt if "forget" in item.split("/")[-1]]
    try:
        forget_ckpt = sorted(forget_ckpt, key=lambda x: float(x.split("/")[-1].split(".ckpt")[0].split("fxppl=")[-1].split("-")[0]))[0]
    except IndexError:
        print(forget_ckpt)
        raise FileNotFoundError(f"Forget ckpt not found in {args.output_dir}")
    except ValueError as e:
        print(forget_ckpt)
        print(e)
    forget_ckpt_metrics = forget_ckpt.split("/")[-1].split("-")[0]
    forget_tv = TaskVector(pretraind_model, forget_ckpt)

    model = (-forget_tv).apply_to(pretraind_model, scaling_coef=args.forget_scaling_coef)
    model_name = f"negtv_fs{args.forget_scaling_coef}_{forget_ckpt_metrics}"

    if args.retain_scaling_coef != 0:
        retain_ckpt = [item for item in saved_ckpt if f"retain{args.retain_multiplier}" in item.split("/")[-1]]
        assert retain_ckpt, f"Retain ckpt not found in {args.output_dir}"
        try:
            retain_ckpt = sorted(retain_ckpt, key=lambda x: float(x.split("/")[-1].split(".ckpt")[0].split("rxppl=")[-1].split("-")[0]))[0]
        except IndexError:
            print(retain_ckpt)
            raise FileNotFoundError(f"Retain ckpt not found in {args.output_dir}")
        except ValueError as e:
            print(retain_ckpt)
            print(e)
        retain_ckpt_metrics = retain_ckpt.split("/")[-1].split("-")[0]
        retain_tv = TaskVector(pretraind_model, retain_ckpt)

        model = retain_tv.apply_to(model, scaling_coef=args.retain_scaling_coef)
        model_name += f"-rs{args.retain_scaling_coef}_{retain_ckpt_metrics}"
    
    # model_path = os.path.join(args.output_dir, f"{model_name}.ckpt")
    # if not os.path.exists(model_path):
    #     torch.save(model, model_path)
    return model

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

    if not args.do_train:
        args.fit_target = "forget"

    if args.do_train and args.method in ["finetune", "negtaskvector"] and args.fit_target == "both":
        method = args.method
        args.method = "finetune"
        do_eval = args.do_eval
        args.do_eval = False
        do_test = args.do_test
        args.do_test = False
        args.fit_target = "forget"
        main(args)
        wandb.finish()
        
        args.fit_target = "retain"
        main(args)
        wandb.finish()

        if do_eval or do_test:
            args.method = method
            args.do_eval = do_eval
            args.do_test = do_test
            args.do_train = False
            args.fit_target = "forget"
            args.wandb_mode = "disabled"
            main(args)

    elif args.do_train and args.method in ["finetune", "negtaskvector"] and args.fit_target in ["forget", "retain"]:
        method = args.method
        args.method = "finetune"
        do_eval = args.do_eval
        args.do_eval = False
        do_test = args.do_test
        args.do_test = False
        main(args)
        wandb.finish()

        if do_eval or do_test:
            args.method = method
            args.do_eval = do_eval
            args.do_test = do_test
            args.do_train = False
            args.fit_target = "forget"
            args.wandb_mode = "disabled"
            main(args)

    elif args.do_train and "sisa" in args.method:
        from datamodules.sisa_dataset import sizeOfShard

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
