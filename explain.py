"""Experimental spatial Grad-CAM engine. Not validated on research checkpoints.

Accepts a *separate* model copy and explicit spatial target layer. Never invoke
on the shared inference cache; architecture-specific layer selection and real
checkpoint validation are prerequisites to enabling this in the UI.
"""
def spatial_gradcam(model, layer, tensor, class_index):
    import torch
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError('A single NCHW image tensor is required')
    activation=[]
    handle=layer.register_forward_hook(lambda module, args, output: activation.append(output))
    was_training=model.training
    try:
        model.eval()
        x=tensor.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logits=model(x)
            if not torch.is_tensor(logits) or logits.ndim!=2 or not 0<=class_index<logits.shape[1]:
                raise ValueError('Invalid class index or classifier output')
            if not torch.isfinite(logits).all():
                raise ValueError('Non-finite classifier output')
            if len(activation)!=1 or not torch.is_tensor(activation[0]) or activation[0].ndim!=4:
                raise ValueError('Target layer must emit a single NCHW spatial activation')
            features=activation[0]
            gradients=torch.autograd.grad(logits[0,class_index],features)[0]
            heat=(gradients.mean(dim=(2,3),keepdim=True)*features).sum(dim=1).relu()
            if not torch.isfinite(heat).all() or float((heat.max()-heat.min()).detach())<=1e-12:
                raise ValueError('No informative heatmap for this target')
            heat=(heat-heat.min())/(heat.max()-heat.min())
            return torch.nn.functional.interpolate(heat[:,None],size=x.shape[-2:],mode='bilinear',align_corners=False)[0,0].detach().cpu()
    finally:
        handle.remove()
        model.train(was_training)
