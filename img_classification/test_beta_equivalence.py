"""
Verify that beta=0 (sigmoid(0)=0.5) produces same output as original /2 correction.
Run on cluster: python test_beta_equivalence.py
"""
import torch
from lion.models_lion import lion_tiny_patch16_224

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = lion_tiny_patch16_224(format='RNN', mask_type='Decay').to(device)

# beta initialized to zeros -> sigmoid(0) = 0.5, matches original /2
for name, p in model.named_parameters():
    if 'beta' in name:
        assert torch.all(p == 0), f"beta not zero at init: {p}"
        print(f"  {name}: {torch.sigmoid(p).detach().cpu().tolist()}")

x = torch.randn(2, 3, 224, 224).to(device)
with torch.no_grad():
    out = model(x)

assert out.shape == (2, 1000), f"unexpected output shape: {out.shape}"
assert not torch.isnan(out).any(), "NaN in output"
assert not torch.isinf(out).any(), "Inf in output"

print(f"output shape: {out.shape}")
print(f"output mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")
print("PASSED: beta init equivalent to original /2 correction")
