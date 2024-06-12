import os.path as osp

import torch



class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = self._get_state_dict(pretrained_checkpoint)
                finetuned_state_dict = self._get_state_dict(finetuned_checkpoint)

                self.vector = {}
                for key in pretrained_state_dict:
                    # if key not in finetuned_state_dict:
                    #     print(f'Warning: key {key} is present in the pretrained state dict but not in the finetuned state dict')
                    #     continue
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)
    
    def __truediv__(self, scalar):
        """Divide a task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] / scalar
        return TaskVector(vector=new_vector)

    def maximum(self, other):
        """Element-wise maximum of two task vectors."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = torch.maximum(self.vector[key], other.vector[key])
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = self._get_model(pretrained_checkpoint)
            pretrained_state_dict = self._get_state_dict(pretrained_checkpoint)
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                pretrained_state_dict[key] += scaling_coef * self.vector[key]
        return pretrained_model
    
    def _get_state_dict(self, ckpt):
        model = self._get_model(ckpt)
        if isinstance(model, dict) and 'state_dict' in model:
            state_dict = model['state_dict']
        elif isinstance(model, dict) and 'module' in model:
            state_dict = model['module']
        elif isinstance(model, torch.nn.Module):
            model.to('cuda:0')
            state_dict = model.state_dict()
        return state_dict
    
    def _get_model(self, ckpt):
        if isinstance(ckpt, str) and osp.exists(ckpt) and osp.isfile(ckpt):
            model = torch.load(ckpt)
        elif isinstance(ckpt, str) and osp.exists(ckpt) and osp.isdir(ckpt):
            model = torch.load(f"{ckpt}/checkpoint/mp_rank_00_model_states.pt")
        elif isinstance(ckpt, torch.nn.Module):
            model = ckpt
        return model

if __name__ == "__main__":
    import lightning as L
    from transformers import AutoModelForCausalLM

    ckpt = "/home/nas1_userA/minseokchoi20/kyunghyun/multilingual_unlearning/.checkpoints/xglm-564M/flores/negtaskvector/BS32_LR0.0003_W0.1_S42/fxma=0.6075-fxppl=10.01-vxppl=97.70.ckpt"

    class MultilingualModel(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = AutoModelForCausalLM.from_pretrained(
                "facebook/xglm-564M",
                cache_dir="../.cache",
                resume_download=True,
            )
    model = MultilingualModel()
    
    print(model.state_dict()["model.model.embed_tokens.weight"])

    task_vector = TaskVector(model, ckpt)
    new_model = (-task_vector).apply_to(model, scaling_coef=10)
    print(new_model.state_dict()["model.model.embed_tokens.weight"])
