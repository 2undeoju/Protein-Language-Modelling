# == DataLoader ==
import torch
import gzip, random
from Bio import SeqIO
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable, List, Optional, Union
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset, DataLoader

# Define ProteinMaskedDataset class
class ProteinMaskedDataset(Dataset):
    def __init__(self, input_ids_list, label_ids_list):
        self.inputs = input_ids_list
        self.labels = label_ids_list

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "labels": self.labels[idx]
        }
# Define default masking function for MLM

def default_mask_mlm(token_ids: List[int], mask_token_id: int, mask_prob: float = 0.25):
    # Standard 80/10/10 MLM
    import random
    vocab_upper = 22  # adjust to your vocab size - 1 for mLSTM path, or pass in if needed

    input_ids = list(token_ids)  # mutable copy
    labels = [-100] * len(input_ids)

    for i in range(len(input_ids)):
        if random.random() < mask_prob:
            original = input_ids[i]
            labels[i] = original  # supervise only masked positions

            r = random.random()
            if r < 0.8:
                input_ids[i] = mask_token_id
            elif r < 0.9:
                # replace with random token in vocab range [0, vocab_upper]
                input_ids[i] = random.randint(0, vocab_upper)
            else:
                # keep original token 10% of the time
                pass

    return input_ids, labels

# def default_mask_mlm(token_ids: List[int], mask_token_id: int, mask_prob: float = 0.25):
#     """Simple MLM masking: replace with mask_token_id and set labels; others = -100."""
#     input_ids = token_ids.copy()
#     labels = [-100] * len(token_ids)
#     for i in range(len(token_ids)):
#         if random.random() < mask_prob:
#             labels[i] = token_ids[i]
#             input_ids[i] = mask_token_id
#     # Ensure labels are in [0, 22] or -100
#     labels = [label if (label == -100 or label <= 22) else -100 for label in labels]
#     return input_ids, labels


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")


class StreamingProteinDataset(IterableDataset):
    """
    Stream proteins from one or more FASTA files.
    - tokenize_fn: callable(seq_str) -> List[int]
    - task: 'mlm' (masked LM) or 'clm' (causal LM)
    - mask_token_id: required for 'mlm'
    """
    def __init__(
        self,
        fasta_paths: Union[str, List[str]],
        tokenize_fn,
        task: str = "mlm",
        mask_token_id: Optional[int] = None,
        mask_prob: float = 0.25,
        max_len: Optional[int] = None,
        shuffle_window: int = 1024,
        seed: int = 0,
    ):
        super().__init__()
        self.fasta_paths = [fasta_paths] if isinstance(fasta_paths, str) else list(fasta_paths)
        self.tokenize_fn = tokenize_fn
        self.task = task
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.max_len = max_len
        self.shuffle_window = shuffle_window
        self.seed = seed



    def _records(self) -> Iterable[str]:
        for p in self.fasta_paths:
            with _open(p) as fh:
                for rec in SeqIO.parse(fh, "fasta"):
                    yield str(rec.seq)



    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        worker_id = wi.id if wi is not None else 0
        rng = random.Random(self.seed + worker_id)

        buf = []
        for seq in self._records():
            if self.max_len is not None:
                seq = seq[: self.max_len]

            token_ids = self.tokenize_fn(seq)
            if self.task == "mlm":
                assert self.mask_token_id is not None, "mask_token_id required for MLM"
                input_ids, labels = default_mask_mlm(token_ids, self.mask_token_id, self.mask_prob)
            elif self.task == "clm":
                if len(token_ids) < 2:
                    continue
                input_ids, labels = token_ids[:-1], token_ids[1:]
            else:
                raise ValueError(f"Unknown task: {self.task}")

            ex = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
            buf.append(ex)

            if len(buf) >= self.shuffle_window:
                j = rng.randrange(len(buf))
                yield buf.pop(j)

        while buf:
            j = rng.randrange(len(buf))
            yield buf.pop(j)



def collate_fn(batch, pad_id: int = 0, label_pad: int = -100, chunk_size: int = 64):
    """Pad to max length in the batch, then pad to chunk_size (like collate_fn)."""
    xs = [b["input_ids"] for b in batch]
    ys = [b["labels"] for b in batch]

    # Type assertions
    assert all(isinstance(x, torch.Tensor) and x.dtype == torch.long for x in xs), f"input_ids must be LongTensors, got {[ (type(x), getattr(x, 'dtype', None)) for x in xs ]}"
    assert all(isinstance(y, torch.Tensor) and y.dtype in (torch.long,) for y in ys), f"labels must be LongTensors, got {[ (type(y), getattr(y, 'dtype', None)) for y in ys ]}"

    input_ids_padded = pad_sequence(xs, batch_first=True, padding_value=pad_id)
    labels_padded = pad_sequence(ys, batch_first=True, padding_value=label_pad)


    from mlstm_utils import pad_tensor_to_chunk
    input_ids_padded = pad_tensor_to_chunk(input_ids_padded, chunk_size=chunk_size, pad_value=pad_id)
    labels_padded = pad_tensor_to_chunk(labels_padded, chunk_size=chunk_size, pad_value=label_pad)

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (input_ids_padded != pad_id).long()

    # === Add this assertion to check input_ids type and label range ===
    assert input_ids_padded.dtype in (torch.int32, torch.int64), (
        f"input_ids must be integer tensor, got dtype {input_ids_padded.dtype}"
    )
    #if input_ids_padded.dtype == torch.int64: # torch.long is int64  #pass

    valid_labels = labels_padded[labels_padded != label_pad]
    if valid_labels.numel() > 0:
        assert valid_labels.min().item() >= 0, (
            f"Label below zero! min={valid_labels.min().item()}"
        )

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "attention_mask": attention_mask,
        "lengths": torch.tensor([len(x) for x in xs], dtype=torch.long),
    }


def token_bucket(loader: DataLoader, max_tokens_per_batch: int, pad_id: int = 0, label_pad: int = -100):
    """
    Group single-sample items into variable-size batches up to a token budget.
    Usage:
        base = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0], ...)
        for batch in token_bucket(base, max_tokens_per_batch):
            ...
    """
    batch, tok_sum = [], 0
    for ex in loader:
        l = int(ex["input_ids"].shape[0])
        if batch and tok_sum + l > max_tokens_per_batch:
            yield collate_fn(batch, pad_id=pad_id, label_pad=label_pad)
            batch, tok_sum = [], 0
        batch.append(ex)
        tok_sum += l
    if batch:
        yield collate_fn(batch, pad_id=pad_id, label_pad=label_pad)


class TokenBucketDataLoader:
    """
    Thin wrapper that looks like a DataLoader but yields token-bucketed batches.
    Note: __len__ is undefined for streaming; drive training by steps_per_epoch.
    """
    def __init__(self, dataset: IterableDataset, max_tokens_per_batch: int, **dl_kwargs):
        # Important: batch_size=1 and collate_fn returns the single element
        self.inner = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=lambda x: x[0],
            **dl_kwargs
        )
        self.max_tokens = max_tokens_per_batch
        self.pad_id = dl_kwargs.get("pad_id", 0)
        self.label_pad = dl_kwargs.get("label_pad", -100)

    def __iter__(self):
        return token_bucket(self.inner, self.max_tokens, pad_id=self.pad_id, label_pad=self.label_pad)
