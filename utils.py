
def deepspeed_weights_only(strategy):
    # DeepSpeed Remove Optimizer State from Checkpoint
    if "deepspeed" in strategy:
        from deepspeed import comm as dist
        from deepspeed.utils import groups, logger
        from deepspeed.runtime.engine import DeepSpeedEngine
        from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine

        def _configure_checkpointing(self, dist_init_required):
            self.checkpoint_engine = TorchCheckpointEngine()

            if self._config is not None and self._config.nebula_config.enabled:
                try:
                    from deepspeed.runtime.checkpoint_engine.nebula_checkpoint_engine import \
                        NebulaCheckpointEngine
                    self.checkpoint_engine = NebulaCheckpointEngine(config_params=self._config.nebula_config)
                except ImportError as err:
                    logger.error(f"No torch_nebula was found! Will fall back to torch.save. Details: {err}")
                    self.checkpoint_engine = TorchCheckpointEngine()

            dp_rank = groups._get_sequence_data_parallel_rank()

            rank = self.local_rank if self.use_node_local_storage() else dp_rank

            # only the first data parallel process needs to store the model checkpoint
            # if you want to use node local storage this must be done by rank 0 on each
            # node
            self.save_non_zero_checkpoint = (rank == 0) or (self.zero_optimization_partition_weights()
                                                            and self.is_first_weights_partition_group())

        DeepSpeedEngine._configure_checkpointing = _configure_checkpointing

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