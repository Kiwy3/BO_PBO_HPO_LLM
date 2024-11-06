import torch
from lightning.pytorch.plugins.precision.bitsandbytes import BitsandbytesPrecision

def quantize_plug(quantize = "bnb.nf4", precision="16-true"):
    dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
    plugins = BitsandbytesPrecision(quantize[4:], dtype)
    return plugins