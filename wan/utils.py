import torch, os, argparse
from safetensors import safe_open
from contextlib import contextmanager
import hashlib

@contextmanager
def init_weights_on_device(device = torch.device("meta"), include_buffers :bool = False):
    
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer
    
    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)
            
    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper
    
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}
    
    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)

def load_state_dict_from_folder(file_path, torch_dtype=None):
    state_dict = {}
    for file_name in os.listdir(file_path):
        if "." in file_name and file_name.split(".")[-1] in [
            "safetensors", "bin", "ckpt", "pth", "pt"
        ]:
            state_dict.update(load_state_dict(os.path.join(file_path, file_name), torch_dtype=torch_dtype))
    return state_dict


def load_state_dict(file_path, torch_dtype=None, device="cpu"):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype, device=device)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, device=device)


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    state_dict = {}
    with safe_open(file_path, framework="pt", device=str(device)) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, device="cpu"):
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


def search_for_embeddings(state_dict):
    embeddings = []
    for k in state_dict:
        if isinstance(state_dict[k], torch.Tensor):
            embeddings.append(state_dict[k])
        elif isinstance(state_dict[k], dict):
            embeddings += search_for_embeddings(state_dict[k])
    return embeddings


def search_parameter(param, state_dict):
    for name, param_ in state_dict.items():
        if param.numel() == param_.numel():
            if param.shape == param_.shape:
                if torch.dist(param, param_) < 1e-3:
                    return name
            else:
                if torch.dist(param.flatten(), param_.flatten()) < 1e-3:
                    return name
    return None


def build_rename_dict(source_state_dict, target_state_dict, split_qkv=False):
    matched_keys = set()
    with torch.no_grad():
        for name in source_state_dict:
            rename = search_parameter(source_state_dict[name], target_state_dict)
            if rename is not None:
                print(f'"{name}": "{rename}",')
                matched_keys.add(rename)
            elif split_qkv and len(source_state_dict[name].shape)>=1 and source_state_dict[name].shape[0]%3==0:
                length = source_state_dict[name].shape[0] // 3
                rename = []
                for i in range(3):
                    rename.append(search_parameter(source_state_dict[name][i*length: i*length+length], target_state_dict))
                if None not in rename:
                    print(f'"{name}": {rename},')
                    for rename_ in rename:
                        matched_keys.add(rename_)
    for name in target_state_dict:
        if name not in matched_keys:
            print("Cannot find", name, target_state_dict[name].shape)


def search_for_files(folder, extensions):
    files = []
    if os.path.isdir(folder):
        for file in sorted(os.listdir(folder)):
            files += search_for_files(os.path.join(folder, file), extensions)
    elif os.path.isfile(folder):
        for extension in extensions:
            if folder.endswith(extension):
                files.append(folder)
                break
    return files


def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def split_state_dict_with_prefix(state_dict):
    keys = sorted([key for key in state_dict if isinstance(key, str)])
    prefix_dict = {}
    for key in  keys:
        prefix = key if "." not in key else key.split(".")[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(key)
    state_dicts = []
    for prefix, keys in prefix_dict.items():
        sub_state_dict = {key: state_dict[key] for key in keys}
        state_dicts.append(sub_state_dict)
    return state_dicts


def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()

def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--save_steps", type=int, default=500, help="Number of checkpoint saving invervals.")
    parser.add_argument("--seed", type=int, default=20917, help="Random seed.")

    parser.add_argument("--t5_path", type=str, default=None, help="Path to the T5 model.")
    parser.add_argument("--vae_path", type=str, default=None, help="Path to the VAE model.")
    parser.add_argument("--dit_paths", type=str, nargs="+", default=None, help="Paths to load diffusion models.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer.")
    parser.add_argument("--extra_inputs", type=str, nargs="+", default=[], help="Additional model inputs, comma-separated.")
    parser.add_argument("--add_control_adapter", default=False, action="store_true", help="Whether to enable control adapter.")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--eta_min", type=float, default=0.0, help="Eta min for the learning rate scheduler.")
    parser.add_argument("--eta_max", type=float, default=1.0, help="Eta max for the learning rate scheduler.")
    parser.add_argument("--num_steps", type=int, default=200000, help="Number of steps.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, nargs="+", default=[], help="Remove prefix in ckpt, e.g., pipe.dit.")
    parser.add_argument("--wrap_with_prefix", default=False, action="store_true", help="Wrap state dict with prefix.")
    parser.add_argument("--trainable_models", type=str, nargs="+", default=[], help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--use_reentrant", default=False, action="store_true", help="Whether to use reentrant in gradient checkpointing. If usint Zero3, this should be True.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint.")

    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--condition_video_base_path", type=str, default=None, help="Base path of the condition video.")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--sample_size", type=int, nargs=2, default=[960,960], help="Sample size of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--height_division_factor", type=int, default=16, choices=[8, 16], help="Height division factor of the images or videos.")
    parser.add_argument("--width_division_factor", type=int, default=16, choices=[8, 16], help="Width division factor of the images or videos.")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--prompt_key", type=str, default="long_caption", help="Prompt key in the metadata.")
    parser.add_argument("--video_key", type=str, default="video_path", help="Video key in the metadata.")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of the dataset.")
    parser.add_argument("--end_idx", type=int, default=None, help="End index of the dataset.")
    parser.add_argument("--video_backend", type=str, default="decord", help="Video backend.", choices=["decord", "pyav"])
    parser.add_argument("--metadata_folder", type=str, default="json", help="Metadata folder of the latent dataset.")
    parser.add_argument("--latent_folder", type=str, default="latents", help="Latent folder of the latent dataset.")
    parser.add_argument("--metadata_paths", type=str, nargs="*", default=[], help="Metadata folder of the latent dataset. This is used for the case when metadata is not in the base path. Specially designed for blob storage.")
    parser.add_argument("--dataset_weights", type=float, nargs="*", default=[], help="Weights of the metadata folders of the latent dataset.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size of the latent dataset.")
    parser.add_argument("--buffer_size", type=int, default=100, help="Buffer size of the latent dataset.")
    parser.add_argument("--min_latents_num", type=int, default=1, help="Min latents number of the latent dataset.")
    parser.add_argument("--shuffle", default=False, action="store_true", help="Whether to shuffle the dataset.")
    parser.add_argument("--disable_score_condition", default=False, action="store_true", help="Whether to disable score condition.")
    parser.add_argument("--finalize_partial_buckets", default=False, action="store_true", help="Whether to finalize partial buckets.")
    parser.add_argument("--dataset_base_path", type=str, nargs="*", default=[], help="Base path of the dataset.")
    parser.add_argument("--dataset_depth_path", type=str, nargs="*", default=[], help="Base path of the depth dataset.")
    parser.add_argument("--additional_keys", type=str, nargs="+", default=[], help="Additional keys in the metadata.")
    parser.add_argument("--dataset_num_workers", type=int, default=12, help="Number of workers for data loading.")
    parser.add_argument("--dataset_prefetch_factor", type=int, default=8, help="Number of batches loaded in advance by each worker.")

    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")

    parser.add_argument("--enable_vace", default=False, action="store_true", help="Whether to enable VACE.")
    parser.add_argument("--vace_layers", type=int, nargs="+", default=[], help="Which layers VACE is added to.")
    parser.add_argument("--vace_in_dim", type=int, default=96, help="Input dimension of VACE.")
    parser.add_argument("--vace_camera_adapter_in_dim", type=int, default=24, help="Input dimension of VACE camera adapter.")
    parser.add_argument("--vace_camera_adapter_out_dim", type=int, default=48, help="Output dimension of VACE camera adapter. If concat, set 48. else set equal to feature dim.")
    parser.add_argument("--vace_camera_add_method", type=str, default="concat", help="Add method of VACE camera adapter.", choices=["concat", "add"])
    parser.add_argument("--vace_scale", type=float, default=1.0, help="Scale of VACE.")
    parser.add_argument("--vace_path", type=str, default=None, help="Path of the VACE model.")

    parser.add_argument("--ar_num_hist_latents", type=int, default=0, help="Number of history latents for AR.")
    parser.add_argument("--ar_max_hist_timestep_boundary", type=float, default=1.0, help="Max timestep boundary for history latents for AR.")
    parser.add_argument("--ar_min_hist_timestep_boundary", type=float, default=1.0, help="Min timestep boundary for history latents for AR, default to 1.0 which means no noise")
    parser.add_argument("--ar_uncond_p", type=float, default=0.0, help="Uncond probability for AR.")

    parser.add_argument("--enable_ref_image_embedding", default=False, action="store_true", help="Whether to enable ref image embedding.")
    parser.add_argument("--ref_image_uncond_p", type=float, default=0.1, help="Uncond probability for ref image embedding.")
    parser.add_argument("--ref_on_vace", default=False, action="store_true", help="Whether to enable ref on VACE.")
    
    parser.add_argument("--log_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--log_time_interval", type=float, default=60.0, help="Log metrics every N seconds (in addition to step-based logging).")
    parser.add_argument("--enable_wandb", default=False, action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb_project", type=str, default="wan_video_training", help="Wandb project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name (auto-generated if not specified).")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Wandb run id (auto-generated if not specified).")

    parser.add_argument("--prompt_uncond_p", default=0.1, type=float, help="Prompt uncond probability.")
    parser.add_argument("--img_uncond_p", default=0.1, type=float, help="Image uncond probability.")

    parser.add_argument("--add_geo_feat_adapter", default=False, action="store_true", help="Whether to enable geo feat adapter.")
    
    return parser