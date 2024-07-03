
def deepspeed_weights_only(strategy):
    # DeepSpeed Remove Optimizer State from Checkpoint
    if "deepspeed" in strategy:
        import os
        from deepspeed import comm as dist
        from deepspeed.utils import logger
        from deepspeed.runtime.engine import DeepSpeedEngine

        def save_checkpoint(self, save_dir, tag=None, client_state={}, save_latest=True, exclude_frozen_parameters=False):
            """Save training checkpoint

            Arguments:
                save_dir: Required. Directory for saving the checkpoint
                tag: Optional. Checkpoint tag used as a unique identifier for the checkpoint, global step is
                    used if not provided. Tag name must be the same across all ranks.
                client_state: Optional. State dictionary used for saving required training states in the client code.
                save_latest: Optional. Save a file 'latest' pointing to the latest saved checkpoint.
                exclude_frozen_parameters: Optional. Exclude frozen parameters from checkpointed state.
            Important: all processes must call this method and not just the process with rank 0. It is
            because each process needs to save its master weights and scheduler+optimizer states. This
            method will hang waiting to synchronize with other processes if it's called just for the
            process with rank 0.

            """
            if self._optimizer_has_ckpt_event_prologue():
                # Custom preparation for checkpoint save, if applicable
                self.optimizer.checkpoint_event_prologue()

            rank = self.local_rank if self.use_node_local_storage() else self.global_rank

            # This is to make sure the checkpoint names are created without collision
            # There seems to be issue creating them in parallel

            # Ensure save_dir directory exists
            if rank == 0:
                self.checkpoint_engine.makedirs(save_dir, exist_ok=True)
            dist.barrier()

            if tag is None:
                tag = f"global_step{self.global_steps}"

            # Ensure tag is a string
            tag = str(tag)
            self.checkpoint_engine.create(tag)

            # Ensure checkpoint tag is consistent across ranks
            self._checkpoint_tag_validation(tag)

            if self.has_moe_layers:
                self.save_non_zero_checkpoint = False
                self._create_checkpoint_file(save_dir, tag, False)
                self._save_moe_checkpoint(save_dir,
                                        tag,
                                        client_state=client_state,
                                        exclude_frozen_parameters=exclude_frozen_parameters)

            # We distribute the task of saving layer checkpoint files among
            # data parallel instances, so all procs should call _save_checkpoint.
            # All procs then call module_state_dict(), but only procs of data
            # parallel rank 0 save the general model params.
            if not self.has_moe_layers:
                self._create_checkpoint_file(save_dir, tag, False)
                self._save_checkpoint(save_dir,
                                    tag,
                                    client_state=client_state,
                                    exclude_frozen_parameters=exclude_frozen_parameters)

            if "optimizer_states" not in client_state: # save_weights_only = True
                self.save_zero_checkpoint = False

            if self.save_zero_checkpoint:
                self._create_zero_checkpoint_files(save_dir, tag)
                self._save_zero_checkpoint(save_dir, tag)

            if self.zero_has_nvme_offload():
                from shutil import copytree, disk_usage
                offload_dir = self.optimizer.optimizer_swapper.swap_folder
                offload_ckpt_dir = os.path.join(save_dir, tag, "offloaded_tensors")
                _, _, free = disk_usage(save_dir)
                logger.info(
                    f"Copying NVMe offload files from {offload_dir} to {offload_ckpt_dir}, {free / 1e9:,.2f} GB free on target filesystem..."
                )
                copytree(offload_dir,
                        offload_ckpt_dir,
                        ignore=lambda _, dir_list: list(filter(lambda x: 'gradient' in x, dir_list)),
                        dirs_exist_ok=False)
                _, _, free = disk_usage(save_dir)
                logger.info(f"Copying complete! {free / 1e9:,.2f} GB free on target filesystem")

            if self._optimizer_has_ckpt_event_epilogue():
                self.optimizer.checkpoint_event_epilogue()

            # Save latest checkpoint tag
            self.checkpoint_engine.commit(tag)
            if save_latest and rank == 0:
                with open(os.path.join(save_dir, 'latest'), 'w') as fd:
                    fd.write(tag)

            dist.barrier()

            return True
        DeepSpeedEngine.save_checkpoint = save_checkpoint

def update_deepspeed_initalize(strategy, use_lora):
    # Add this line to solve AttributeError: 'PeftModelForSequenceClassification' object has no attribute 'base_model'
    # which is caused by Initiate deepspeed both in lightning and peft
    if "deepspeed" in strategy and use_lora:
        from lightning.pytorch.strategies import DeepSpeedStrategy
        from contextlib import contextmanager

        @contextmanager
        def model_sharded_context(self):
            yield
        DeepSpeedStrategy.model_sharded_context = model_sharded_context

def installed_cuda_version():
    import subprocess
    import torch.utils.cpp_extension
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    assert cuda_home is not None, "CUDA_HOME does not exist, unable to compile CUDA op(s)"
    # Ensure there is not a cuda version mismatch between torch and nvcc compiler
    output = subprocess.check_output([cuda_home + "/bin/nvcc",
                                      "-V"],
                                     universal_newlines=True)
    output_split = output.split()
    release_idx = output_split.index("release")
    release = output_split[release_idx + 1].replace(',', '').split(".")
    # Ignore patch versions, only look at major + minor
    cuda_major, cuda_minor = release[:2]
    installed_cuda_version = ".".join(release[:2])
    return int(cuda_major), int(cuda_minor)