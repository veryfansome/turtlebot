"""World model architecture: the baseline, pet_learn.model.WorldModel (MLP encoder to a state z,
residual MLP latent dynamics, one MLP head per prediction target, a pose head).

CONTRACT (arch axis). Expose build(input_dim, **params) -> torch.nn.Module with:
- z_dim: int, the size of the state;
- encode(x): inputs (B, input_dim) -> states (B, z_dim);
- rollout(z0, actions): z0 (B, z_dim), actions (B, h, 2) -> states (B, h+1, z_dim), z0 first;
- decode(zs, current): states (B, h+1, z_dim) and each target's current reading (B, 1, size) ->
  {target: (B, h+1, size)}, one entry per dataset.TARGETS name; continuous targets in the
  standardized units of `current` (predicting a change from it is the baseline's choice, not a
  requirement), bump and slip as logits;
- pose(z0, zs): z0 (B, 1, z_dim) and states (B, h, z_dim) -> pose changes (B, h, 3).
The harness scores the model only through these, on held-out windows it never trains on. The
driver (train_driver) builds on encode() and z_dim, so keep them meaningful: a state the policy
can read. Everything must be differentiable and deterministic given torch's seed.
"""
from pet_learn.model import WorldModel


def build(input_dim, z=64, hidden=128, enc_layers=2, dyn_layers=2, head_layers=1, dropout=0.0):
    """z: the state's size; hidden: every hidden layer's width; enc_layers, dyn_layers, head_layers: hidden
    layers in the encoder, the dynamics and each head; dropout after each hidden layer (none by default)."""
    model = WorldModel(input_dim, z_dim=z, hidden=hidden, enc_layers=enc_layers, dyn_layers=dyn_layers,
                       head_layers=head_layers, dropout=dropout)
    model.z_dim = z
    return model
