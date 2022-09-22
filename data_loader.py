import torch
from torch.utils.data import Dataset


class dataset(Dataset):
    """wrap in PyTorch Dataset"""
    def __init__(self, examples):
        """

        :param examples: examples returned by VUA_All_Processor or Verb_Processor
        """
        super(dataset, self).__init__()
        self.examples = examples

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


def collate_fn(examples):
    ids_ls, segs_ls, att_mask_ls, ids_rs, att_mask_rs, segs_rs, labels = map(list, zip(*examples))

    ids_ls = torch.tensor(ids_ls, dtype=torch.long)
    segs_ls = torch.tensor(segs_ls, dtype=torch.long)
    att_mask_ls = torch.tensor(att_mask_ls, dtype=torch.long)
    ids_rs = torch.tensor(ids_rs, dtype=torch.long)
    att_mask_rs = torch.tensor(att_mask_rs, dtype=torch.long)
    segs_rs = torch.tensor(segs_rs, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return ids_ls, segs_ls, att_mask_ls, ids_rs, att_mask_rs, segs_rs, labels
