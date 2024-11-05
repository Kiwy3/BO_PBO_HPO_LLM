import gc
from collections import defaultdict
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Dict, Optional, Tuple, Union

import torch
from lightning.fabric.utilities.load import _NotYetLoadedTensor as NotYetLoadedTensor

from litgpt import Config
from litgpt.scripts.convert_hf_checkpoint import layer_template, load_param
from litgpt.utils import extend_checkpoint_dir, incremental_save, lazy_load

def qkv_split(
    param: Union[torch.Tensor, NotYetLoadedTensor], config: Config
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_per_kv = config.n_head // config.n_query_groups
    qs = []
    ks = []
    vs = []
    for chunk in torch.chunk(param, config.n_query_groups):
        split = torch.split(chunk, [config.head_size * q_per_kv, config.head_size, config.head_size])
        qs.append(split[0])
        ks.append(split[1])
        vs.append(split[2])
    q = torch.cat(qs)
    k = torch.cat(ks)
    v = torch.cat(vs)
    return q, k, v

def copy_weights_llama(
    config: Config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    untie_weights: bool = False,
    saver: Optional[incremental_save] = None,
) -> None:
    weight_map = {
        "transformer.wte.weight": "model.embed_tokens.weight",
        "transformer.h.{}.norm_1.weight": "model.layers.{l}.input_layernorm.weight",
        "transformer.h.{}.norm_1.bias": "model.layers.{l}.input_layernorm.bias",
        "transformer.h.{}.attn.proj.weight": "model.layers.{l}.self_attn.o_proj.weight",
        "transformer.h.{}.norm_2.weight": "model.layers.{l}.post_attention_layernorm.weight",
        "transformer.h.{}.norm_2.bias": "model.layers.{l}.post_attention_layernorm.bias",
        "transformer.ln_f.weight": "model.norm.weight",
        "transformer.ln_f.bias": "model.norm.bias",
        "lm_head.weight": "lm_head.weight",
    }
    if config.mlp_class_name == "LLaMAMoE":
        weight_map.update(
            {
                "transformer.h.{}.mlp.gate.weight": "model.layers.{l}.block_sparse_moe.gate.weight",
                "transformer.h.{}.mlp.experts.{}.fc_1.weight": "model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight",
                "transformer.h.{}.mlp.experts.{}.fc_2.weight": "model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight",
                "transformer.h.{}.mlp.experts.{}.proj.weight": "model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight",
            }
        )
    elif config.mlp_class_name in ("LLaMAMLP", "GemmaMLP"):
        weight_map.update(
            {
                "transformer.h.{}.mlp.fc_1.weight": "model.layers.{l}.mlp.gate_proj.weight",
                "transformer.h.{}.mlp.fc_2.weight": "model.layers.{l}.mlp.up_proj.weight",
                "transformer.h.{}.mlp.proj.weight": "model.layers.{l}.mlp.down_proj.weight",
            }
        )
    else:
        raise NotImplementedError
    
    for name, param in lit_weights.items():
        if name == "lm_head.weight" and untie_weights:
            continue
        if name.endswith(".attn.attn.weight"):
            from_name, l = layer_template(name, 2)
            q = "model.layers.{}.self_attn.q_proj.weight".format(l)
            k = "model.layers.{}.self_attn.k_proj.weight".format(l)
            v = "model.layers.{}.self_attn.v_proj.weight".format(l)
            qkv = load_param(param, name, None)
            qp, kp, vp = qkv_split(qkv, config)
            for to_name, param in zip((q, k, v), (qp, kp, vp)):
                if saver is not None:
                    param = saver.store_early(param)
                state_dict[to_name] = param
        else:
            if "transformer.h" in name:
                from_name, l = layer_template(name, 2)
                e = None
                if "mlp.experts" in name:
                    from_name, e = layer_template(from_name, 5)
                to_name = weight_map[from_name]
                to_name = to_name.format(l=l, e=e)
            elif name != "epoch":
                to_name = weight_map[name]
            param = load_param(param, name, None)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param

def check_conversion_supported(lit_weights: Dict[str, torch.Tensor]) -> None:
    if any("lora" in wn for wn in lit_weights):
        raise ValueError("Checkpoints with LoRA weights cannot be converted. Call `scripts/merge_lora.py` first.")
    if any("adapter" in wn or "gating_factor" in wn for wn in lit_weights):
        raise NotImplementedError("Converting adapter models is not supported.")

@torch.inference_mode()
def convert_checkpoint(checkpoint_dir: Path, output_dir: Path) -> None:
    """Convert a LitGPT trained checkpoint into a Hugging Face Transformers checkpoint."""
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    pprint(locals())

    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.pth"

    untie_weights = "Gemma" in config.name
    copy_fn = partial(copy_weights_llama, config, untie_weights=untie_weights)


    # initialize a new empty state dict to hold our new weights
    sd = {}
    with incremental_save(output_path) as saver:
        lit_weights = lazy_load(checkpoint_dir / "lit_model.pth")
        #lit_weights = lit_weights.get("state_dict", lit_weights)
        lit_weights = lit_weights.get("model", lit_weights)
        check_conversion_supported(lit_weights)
        copy_fn(sd, lit_weights, saver=saver)
        gc.collect()
        saver.save(sd)