# models_loader.py
import sys
import os
import torch

def get_edgeface_model(edgeface_dir, model_name='edgeface_s_gamma_05', pretrained=True, device='cpu'):
    """
    edgeface_dir: path to the cloned edgeface repository
    model_name: accepted names from repo (see README). Default 'edgeface_s_gamma_05'
    """
    # add repo path for import
    sys.path.insert(0, edgeface_dir)
    try:
        from backbones import get_model
    except Exception as e:
        raise ImportError(f"Couldn't import EdgeFace backbones. Make sure edgeface dir is correct. Err: {e}")

    model = get_model(model_name)
    model.to(device)
    if pretrained:
        # try to load checkpoint from repo's checkpoints folder if available
        ckpt_path = os.path.join(edgeface_dir, 'checkpoints', f'{model_name}.pt')
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
        else:
            # try torch.hub as fallback
            try:
                import torch as _torch
                model = _torch.hub.load('otroshi/edgeface', model_name, source='github', pretrained=True).to(device)
            except Exception:
                print("Pretrained weights not found locally; proceeding with random init.")
    return model