# Copyright (c) Facebook, Inc. and its affiliates.
# Inspired from https://github.com/gaoyuezhou/dino_wm
# Licensed under the MIT License
import os
import warnings

import torch
import torch.nn as nn

# Suppress xFormers availability warnings from DINOv2
warnings.filterwarnings("ignore", message="xFormers is not available")

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


class DinoEncoder(nn.Module):
    def __init__(self, name, feature_key, causal_enc=False, pretrained=True):
        super().__init__()
        self.name = name
        if self.name.startswith("dinov2"):
            print(f"Loading DINOv2 encoder via torch.hub: {name}", flush=True)
            self.base_model = torch.hub.load("facebookresearch/dinov2", name, pretrained=pretrained)
            print(f"Finished loading DINOv2 encoder: {name}", flush=True)
        elif self.name.startswith("dinov3"):
            print(f"Loading DINOv3 encoder: {name} (pretrained={pretrained})", flush=True)
            pretrained_ckpt_root = os.environ.get("JEPAWM_OSSCKPT") if pretrained else None
            # Try to infer dinov3 path from JEPAWM_HOME, NAVSIM_DEVKIT_ROOT, or default
            jepawm_home = os.environ.get("JEPAWM_HOME")
            if not jepawm_home:
                # Fallback to devkit root if available
                jepawm_home = os.environ.get("NAVSIM_DEVKIT_ROOT", os.path.expanduser("~"))
            dinov3_path = os.path.join(jepawm_home, "dinov3")

            if not os.path.exists(os.path.join(dinov3_path, "hubconf.py")):
                try:
                    # Fallback to loading from github
                    print(f"Loading DINOv3 encoder via torch.hub GitHub fallback: {name}", flush=True)
                    self.base_model = torch.hub.load("facebookresearch/dinov3", name, pretrained=pretrained)
                    print(f"Finished loading DINOv3 encoder via GitHub fallback: {name}", flush=True)
                except ModuleNotFoundError as e:
                    if "termcolor" in str(e):
                        raise ModuleNotFoundError(
                            "The facebookresearch/dinov3 repository requires the 'termcolor' package to be installed. "
                            "Please run `pip install termcolor` in your 'ad-e2e-jepa_py310' conda environment and try running the script again."
                        ) from e
                    else:
                        raise e
                except Exception as e:
                    # Provide an option to load the model from the local downloaded weights if the github 403 error is still occurring
                    local_weights_path = os.path.join(
                        pretrained_ckpt_root or os.path.join(jepawm_home, "pretrained_models"),
                        "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                    )
                    if pretrained and os.path.exists(local_weights_path) and "vitl16" in self.name:
                        # The github repo directly imports load_state_dict_from_url so mocking it on torch.hub doesn't work.
                        # Instead, mock torch.hub.download_url_to_file to just copy the local file to the cache destination.
                        import shutil
                        original_download = torch.hub.download_url_to_file
                        def mock_download(url, dst, hash_prefix=None, progress=True):
                            shutil.copy(local_weights_path, dst)
                        try:
                            torch.hub.download_url_to_file = mock_download
                            print(f"Loading DINOv3 encoder via cached local weights: {local_weights_path}", flush=True)
                            self.base_model = torch.hub.load("facebookresearch/dinov3", name, pretrained=pretrained)
                            print(f"Finished loading DINOv3 encoder via cached local weights: {name}", flush=True)
                        finally:
                            torch.hub.download_url_to_file = original_download
                    else:
                        raise FileNotFoundError(
                            f"Could not find local dinov3 repository at {dinov3_path} "
                            "and fallback to github failed. Please clone facebookresearch/dinov3 "
                            "to that location or set JEPAWM_HOME environment variable appropriately."
                        ) from e
            else:
                if "vitl16" in self.name:
                    print(f"Loading DINOv3 encoder from local repo: {dinov3_path}", flush=True)
                    self.base_model = torch.hub.load(
                        dinov3_path,
                        name,
                        source="local",
                        pretrained=pretrained,
                        weights=f"{pretrained_ckpt_root}/{name}_pretrain_lvd1689m-8aa4cbdd.pth" if pretrained_ckpt_root else None,
                    )
                    print(f"Finished loading DINOv3 encoder from local repo: {name}", flush=True)
                else:
                    print(f"Loading DINOv3 encoder from local repo: {dinov3_path}", flush=True)
                    self.base_model = torch.hub.load(
                        dinov3_path,
                        name,
                        source="local",
                        pretrained=pretrained,
                        weights=f"{pretrained_ckpt_root}/dinov3/{name}_pretrain_lvd1689m.pth" if pretrained_ckpt_root else None,
                    )
                    print(f"Finished loading DINOv3 encoder from local repo: {name}", flush=True)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)  # dummy patch dim
        return emb
