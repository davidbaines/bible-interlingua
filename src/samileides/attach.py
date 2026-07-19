"""Attach a new language to a frozen base run (plan.md, "Anchors & attach").

Two mechanisms, both freezing the base model trained *without* the new
language and learning only a thin new-language head on its NT:

- **graft** (control, Bérard 2021): append one target-tag embedding row (tied,
  so it also grows lm_head) and insert decoder bottleneck adapters; train only
  those on the new language's NT in the base's multi-source format. The `<2xx>`
  tag is a fresh id = old vocab size, prepended manually (SentencePiece never
  saw it), so no retokenisation of the frozen vocab is needed.

- **anchor_decoder**: initialise a decoder + embeddings from the base and train
  it to generate the new language's NT text from the *frozen per-verse anchor*
  (anchors.py), fed as a single-slot encoder memory via ``encoder_outputs=``.
  The base encoder is not used at attach time; the anchor is the interlingua.

Early stopping runs on a held-back NT dev split; the held-out OT is generated
and scored exactly once, at the end (train_attach.py).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .dataset import LABEL_PAD_ID
from .preprocess import SRC_COLUMN, TGT_COLUMN

# ----------------------------------------------------------------------------
# Graft: new tag row + decoder adapters, everything else frozen
# ----------------------------------------------------------------------------


class Adapter(nn.Module):
    """Houlsby bottleneck adapter: down-project, ReLU, up-project, residual."""

    def __init__(self, d_model: int, dim: int):
        super().__init__()
        self.down = nn.Linear(d_model, dim)
        self.up = nn.Linear(dim, d_model)
        nn.init.zeros_(self.up.weight)   # start as identity (up=0 -> residual only)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(torch.relu(self.down(x)))


def add_target_tag_row(model) -> int:
    """Append one embedding row for the new target tag; return its token id.

    Marian ties encoder/decoder embeddings and lm_head, so resizing the token
    embeddings grows all three. The new id is the old vocab size.
    """
    old = model.config.vocab_size
    model.resize_token_embeddings(old + 1)
    model.config.vocab_size = old + 1
    # Initialise the new row as the mean of existing rows (a neutral start).
    emb = model.get_input_embeddings().weight.data
    emb[old] = emb[:old].mean(0)
    return old


def graft_adapters(model, dim: int) -> list[nn.Module]:
    """Wrap each decoder layer's output with an adapter; return the adapters."""
    adapters = []
    for layer in model.model.decoder.layers:
        adapter = Adapter(model.config.d_model, dim).to(model.device)
        adapters.append(adapter)
        orig_forward = layer.forward

        def make(orig, ad):
            def forward(*a, **kw):
                out = orig(*a, **kw)
                if isinstance(out, tuple):
                    return (ad(out[0]),) + out[1:]
                return ad(out)
            return forward

        layer.forward = make(orig_forward, adapter)
    return adapters


def freeze_for_graft(model, new_tag_id: int, adapters: list[nn.Module]):
    """Freeze all base params; leave only the new embedding row + adapters
    trainable. The embedding row is trained via a gradient hook that zeroes
    every row except ``new_tag_id`` (the tied weight can't be split otherwise).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    emb = model.get_input_embeddings().weight
    emb.requires_grad_(True)

    def zero_other_rows(grad):
        mask = torch.zeros_like(grad)
        mask[new_tag_id] = 1.0
        return grad * mask

    emb.register_hook(zero_other_rows)
    for ad in adapters:
        for p in ad.parameters():
            p.requires_grad_(True)


def drop_target_tag(src: str) -> str:
    """Remove the leading ``<2xx>`` target tag; keep the rest verbatim.

    The graft path re-supplies the target tag as a fresh embedding id, so the
    (unknown-to-SentencePiece) ``<2new>`` prefix is stripped before encoding.
    """
    parts = src.split(" ", 1)
    return parts[1] if len(parts) == 2 and parts[0].startswith("<2") else src


class TagPrependDataset(Dataset):
    """Graft dataset: manually prepend the new tag id to the (un-tagged) source.

    ``pairs`` carry multi-source lines already, but their leading ``<2xx>`` tag
    is the *new* language's, which SentencePiece cannot encode. We strip that
    first token and prepend ``new_tag_id`` instead; the remaining ``<1src>``
    tags are in the frozen vocab and encode normally.
    """

    def __init__(self, pairs, sp, new_tag_id: int, max_len: int, max_src_len: int):
        self.src = pairs[SRC_COLUMN].tolist()
        self.tgt = pairs[TGT_COLUMN].tolist()
        self.sp = sp
        self.new_tag_id = new_tag_id
        self.max_len = max_len
        self.max_src_len = max_src_len
        self.eos = sp.eos_id()

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        body = drop_target_tag(self.src[idx])
        ids = [self.new_tag_id] + self.sp.encode(body, out_type=int)[: self.max_src_len - 2]
        ids.append(self.eos)
        labels = self.sp.encode(self.tgt[idx], out_type=int)[: self.max_len - 1]
        labels.append(self.eos)
        return {"input_ids": ids, "labels": labels}


# ----------------------------------------------------------------------------
# Anchor decoder: generate from a frozen per-verse anchor vector
# ----------------------------------------------------------------------------


class AnchorPairDataset(Dataset):
    """(anchor vector, target ids) pairs for the anchor-decoder attach."""

    def __init__(self, vrefs, targets, anchors: np.ndarray, vref_index: dict,
                 sp, max_len: int):
        self.vrefs = list(vrefs)
        self.tgt = list(targets)
        self.anchors = anchors
        self.vi = vref_index
        self.sp = sp
        self.max_len = max_len
        self.eos = sp.eos_id()

    def __len__(self):
        return len(self.vrefs)

    def __getitem__(self, idx):
        labels = self.sp.encode(self.tgt[idx], out_type=int)[: self.max_len - 1]
        labels.append(self.eos)
        anchor = self.anchors[self.vi[self.vrefs[idx]]]
        return {"anchor": anchor.astype(np.float32), "labels": labels}


class AnchorCollator:
    """Pad targets and stack anchors into a single-slot encoder memory.

    Emits the memory as a plain ``encoder_memory`` tensor ([B, 1, d]) — NOT a
    BaseModelOutput, which the DataLoader's pin_memory and the Trainer's
    device-move both choke on. ``AnchorTrainer`` (train_attach.py) wraps it into
    ``encoder_outputs`` at compute-loss time so the decoder cross-attends to the
    frozen anchor and the encoder is skipped. Builds ``decoder_input_ids`` the
    way ``dataset.Collator`` does (the label-smoother pops labels).
    """

    def __init__(self, pad_id: int, decoder_start_id: int | None = None):
        self.pad_id = pad_id
        self.decoder_start_id = pad_id if decoder_start_id is None else decoder_start_id

    def __call__(self, batch):
        tgt_max = max(len(b["labels"]) for b in batch)
        labels, dec_in, anchors = [], [], []
        for b in batch:
            t = b["labels"]
            labels.append(t + [LABEL_PAD_ID] * (tgt_max - len(t)))
            shifted = ([self.decoder_start_id] + t)[:tgt_max]
            dec_in.append(shifted + [self.pad_id] * (tgt_max - len(shifted)))
            anchors.append(b["anchor"])
        memory = torch.tensor(np.stack(anchors), dtype=torch.float32).unsqueeze(1)
        return {
            "encoder_memory": memory,
            "attention_mask": torch.ones(len(batch), 1, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "decoder_input_ids": torch.tensor(dec_in, dtype=torch.long),
        }


def freeze_encoder_only(model):
    """Freeze the (unused) encoder + shared embeddings; train the decoder.

    The anchor replaces the encoder, so its weights never update; the decoder
    and lm_head learn to render the new language from the anchor space.
    """
    for p in model.get_encoder().parameters():
        p.requires_grad_(False)


@torch.no_grad()
def generate_from_anchors(model, sp, device, vrefs, anchors, vref_index,
                          tag_id: int, beam: int = 5, max_length: int = 192,
                          length_penalty: float = 1.0, batch_size: int = 32):
    """Decode target text from frozen anchors (one memory slot per verse)."""
    from transformers.modeling_outputs import BaseModelOutput

    pad, eos, bos = sp.pad_id(), sp.eos_id(), sp.bos_id()
    special = {pad, eos, bos}
    hyps, truncated = [], 0
    for start in range(0, len(vrefs), batch_size):
        chunk = vrefs[start : start + batch_size]
        mem = torch.tensor(
            np.stack([anchors[vref_index[v]] for v in chunk]), dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        enc_out = BaseModelOutput(last_hidden_state=mem)
        attn = torch.ones(len(chunk), 1, dtype=torch.long, device=device)
        out = model.generate(
            encoder_outputs=enc_out, attention_mask=attn,
            num_beams=beam, length_penalty=length_penalty,
            max_length=max_length, early_stopping=True,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )
        for row in out.tolist():
            truncated += int(len(row) >= max_length and eos not in row)
            hyps.append(sp.decode([i for i in row if i not in special]))
    return hyps, truncated
