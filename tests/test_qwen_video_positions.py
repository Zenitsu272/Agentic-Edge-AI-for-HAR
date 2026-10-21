from types import SimpleNamespace
import torch
from qwen_video_positions import video_rope_index


def test_fractional_video_intervals_keep_time_order_instead_of_all_zero():
    model=SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2,tokens_per_second=2)))
    tokens=torch.ones((1,9),dtype=torch.long)
    types=torch.tensor([[0,0,2,2,2,2,2,2,0]])
    positions,deltas=video_rope_index(model,tokens,types,video_grid_thw=torch.tensor([[6,2,2]]),second_per_grid_ts=torch.tensor([.5]))
    assert positions[0,0,2:8].tolist()==[2,3,4,5,6,7]
    assert positions[:,0,-1].tolist()==[8,8,8]
    assert deltas.tolist()==[[0]]


def test_subsecond_positions_quantize_after_multiplication_and_preserve_padding():
    model=SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2,tokens_per_second=2)))
    tokens=torch.ones((1,11),dtype=torch.long)
    types=torch.tensor([[0,0,0,0,2,2,2,2,2,2,0]])
    mask=torch.tensor([[0,0,1,1,1,1,1,1,1,1,1]])
    positions,deltas=video_rope_index(model,tokens,types,video_grid_thw=torch.tensor([[6,2,2]]),second_per_grid_ts=torch.tensor([.25]),attention_mask=mask)
    assert positions[0,0,4:10].tolist()==[2,2,3,3,4,4]
    assert positions[:,:,:2].eq(0).all()
    assert positions[:,0,-1].tolist()==[5,5,5]
    assert deltas.tolist()==[[-5]]


def test_still_images_do_not_consume_video_timing():
    model=SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2,tokens_per_second=2)))
    tokens=torch.ones((1,6),dtype=torch.long)
    types=torch.tensor([[0,1,1,1,1,0]])
    positions,_=video_rope_index(model,tokens,types,image_grid_thw=torch.tensor([[1,4,4]]))
    assert positions[0,0].tolist()==[0,1,1,1,1,3]
