import os
import json
import numpy as np

from torch.utils.data import Dataset, ConcatDataset

def shard_data(output_dir, data_length, shards):
    splitfile = os.path.join(f'{output_dir}', f'shard{shards}-splitfile.jsonl')
    if os.path.exists(splitfile):
        return splitfile

    indices = np.arange(data_length)
    np.random.shuffle(indices)
    partitions = np.split(
        indices,
        [t * (data_length // shards) for t in range(1, shards)],
    )

    with open(splitfile, 'w') as f:
        for i, partition in enumerate(partitions):
            f.write(json.dumps(partition.tolist()) + '\n')
    return splitfile


def get_shard(splitfile, shard):
    if os.path.exists(splitfile):
        with open(splitfile) as f:
            shards = f.readlines()
        shards = [json.loads(shard) for shard in shards]
        return np.array(shards[shard])
    else:
        raise FileNotFoundError(f"Splitfile {splitfile} not found.")

def sizeOfShard(splitfile, shard):
    '''
    Returns the size (in number of points) of the shard before any unlearning request.
    '''
    return get_shard(splitfile, shard).shape[0]

class ShardDataset(Dataset):
    def __init__(self, splitfile, shard, mixed_dataset, split, offset=0, until=None):
        self.shard = get_shard(splitfile, shard)
        self.mixed_dataset = mixed_dataset
        self.split = split
        self.offset = offset
        self.until = until if until is not None else len(self.shard)

        self.slice = self.shard[self.offset:self.until]
        if self.split == "retain":
            self.slice = self.slice[self.slice >= self.mixed_dataset.retain_start_idx]
        self.slice = self.slice.tolist()

    def __len__(self):
        return len(self.slice)

    def __getitem__(self, idx):
        if idx > len(self.slice):
            raise IndexError("Index out of the bounds of the data segment.")
        item, is_forget = self.mixed_dataset[self.slice[idx]]
        assert not is_forget
        return item
        
class MixedDataset(Dataset):
    def __init__(self, retain_data, forget_data):
        self.combined_data = ConcatDataset([forget_data, retain_data])
        self.retain_start_idx = len(forget_data)
        self.total_len = len(self.combined_data)
        self.forget_idx = list(range(self.retain_start_idx))

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data = self.combined_data[idx]
        is_forget = idx in self.forget_idx
        return data, is_forget