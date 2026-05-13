import torch
from transformers import VisionEncoderDecoderModel
import inspect

SPANISH_PATH = r"C:\Users\rdb104\Documents\caserepos\models\trocr_span"

model = VisionEncoderDecoderModel.from_pretrained(
    SPANISH_PATH,
    local_files_only=True,
    low_cpu_mem_usage=False,
    torch_dtype=torch.float32,
    ignore_mismatched_sizes=True
)

ep = model.decoder.model.decoder.embed_positions
print("weights:", ep.weights)
print("weights device (if not None):", ep.weights.device if ep.weights is not None else "N/A")
print()
print("get_embedding source:")
print(inspect.getsource(ep.get_embedding))
print()
print("ALL attributes on ep:")
for k, v in ep.__dict__.items():
    print(f"  {k}: {type(v).__name__} = {v if not isinstance(v, torch.Tensor) else f'tensor shape={v.shape} device={v.device}'}")