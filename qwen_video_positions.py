"""Qwen video position compatibility for Transformers 5.15.1.

The installed implementation casts seconds-per-grid to int before scaling,
collapsing fractional video intervals to zero. Keep fractional timing until
quantizing positions, as in Qwen's original MRoPE implementation. Scoped to
our model instance; no installed package files are modified.
"""
import itertools
from types import MethodType
import torch


def video_rope_index(self, input_ids, mm_token_type_ids, image_grid_thw=None,
                     video_grid_thw=None, second_per_grid_ts=None, attention_mask=None, **kwargs):
    merge = self.config.vision_config.spatial_merge_size
    rate = self.config.vision_config.tokens_per_second
    grids = {1: iter(image_grid_thw) if image_grid_thw is not None else iter(()),
             2: iter(video_grid_thw) if video_grid_thw is not None else iter(())}
    intervals = iter(second_per_grid_ts) if second_per_grid_ts is not None else itertools.repeat(1.)
    positions = torch.zeros((3, *input_ids.shape), dtype=input_ids.dtype, device=input_ids.device)
    deltas = []
    for batch, tokens in enumerate(input_ids):
        mask = attention_mask[batch].bool() if attention_mask is not None else torch.ones_like(tokens, dtype=torch.bool)
        types = mm_token_type_ids[batch][mask].tolist()
        blocks, cursor = [], 0
        for modality, group in itertools.groupby(types):
            length = sum(1 for _ in group)
            if modality == 0:
                block = torch.arange(length, device=tokens.device).expand(3, -1) + cursor
            else:
                t, h, w = map(int, next(grids[modality]))
                h //= merge; w //= merge
                interval = rate * float(next(intervals)) if modality == 2 else 1.
                temporal = (torch.arange(t, device=tokens.device) * interval).long()
                axes = torch.meshgrid(temporal, torch.arange(h, device=tokens.device),
                                      torch.arange(w, device=tokens.device), indexing='ij')
                block = torch.stack(axes).reshape(3, -1) + cursor
                if block.shape[1] != length:
                    raise ValueError('Qwen video grid does not match its vision tokens.')
            blocks.append(block)
            cursor = int(block.max()) + 1
        combined = torch.cat(blocks, dim=1)
        positions[:, batch, mask] = combined
        deltas.append(int(combined.max()) + 1 - input_ids.shape[1])
    return positions, torch.tensor(deltas, device=input_ids.device).unsqueeze(1)


def install_video_position_fix(model):
    # Explicit compatibility scope: do not silently override future runtimes.
    import transformers
    if transformers.__version__ == '5.15.1':
        model.model.get_rope_index = MethodType(video_rope_index, model.model)
