"""Alternative action heads for LeRobot's Diffusion Policy network (lerobot 0.6.1).

The network is unchanged: same ResNet-18 observation encoder, same 1-D conditional U-Net, same
parameter count. Only the training objective (`DiffusionModel.compute_loss`) and the sampler
(`DiffusionModel.conditional_sample`) are swapped, by monkeypatching before LeRobot's train or eval
entry point runs. Checkpoints keep `type: diffusion` in config.json, so every run directory carries
a `t3_head.txt` marker; `t3_launch.py` reads it and applies the same patch at eval time.

Heads
- "diffusion": no patch (stock DDPM, epsilon prediction, 100 train timesteps).
- "flow": rectified flow / conditional flow matching with a linear path (Lipman et al. 2022,
  arXiv 2210.02747; Liu et al. 2022). x_t = (1 - t) * x0 + t * x1, x0 ~ N(0, I) noise,
  x1 = normalized action chunk, t ~ U(0, 1). The U-Net regresses the constant velocity x1 - x0 with
  MSE. Sampling integrates dx/dt = v(x, t) from t=0 to t=1 with `num_inference_steps` explicit
  Euler steps (default 100, same as DDPM's default), then clips to [-1, 1] like the stock sampler.
  The U-Net's sinusoidal timestep embedding gets t * 100, so its input range matches the 0..99
  integer timesteps the stock head uses.
- "mse": plain behaviour cloning with the same network. The U-Net gets an all-zeros "noisy action"
  and timestep 0, and regresses the action chunk directly with MSE. Deterministic: one output per
  observation. This is the explicit-regression baseline for the multimodality probe.
"""

import torch
import torch.nn.functional as F

T_SCALE = 100.0  # maps t in [0, 1] onto the stock head's timestep range


def _cond_and_target(self, batch):
    from lerobot.utils.constants import ACTION, OBS_STATE

    assert batch[OBS_STATE].shape[1] == self.config.n_obs_steps
    assert batch[ACTION].shape[1] == self.config.horizon
    global_cond = self._prepare_global_conditioning(batch)
    return global_cond, batch[ACTION]


def _masked_mean(self, loss, batch):
    if self.config.do_mask_loss_for_padding:
        mask = (~batch["action_is_pad"]).unsqueeze(-1)
        return (loss * mask).sum() / (mask.sum() * loss.shape[-1]).clamp_min(1)
    return loss.mean()


# ---------------- flow matching ----------------
def flow_compute_loss(self, batch):
    global_cond, x1 = _cond_and_target(self, batch)
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
    tt = t[:, None, None]
    xt = (1.0 - tt) * x0 + tt * x1
    v_pred = self.unet(xt, t * T_SCALE, global_cond=global_cond)
    loss = F.mse_loss(v_pred, x1 - x0, reduction="none")
    return _masked_mean(self, loss, batch)


@torch.no_grad()
def flow_conditional_sample(self, batch_size, global_cond=None, generator=None, noise=None):
    p = next(self.parameters())
    shape = (batch_size, self.config.horizon, self.config.action_feature.shape[0])
    x = noise if noise is not None else torch.randn(shape, dtype=p.dtype, device=p.device, generator=generator)
    n = int(self.num_inference_steps)
    dt = 1.0 / n
    for i in range(n):
        t = torch.full((batch_size,), i * dt, dtype=x.dtype, device=x.device)
        x = x + dt * self.unet(x, t * T_SCALE, global_cond=global_cond)
    return x.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)


# ---------------- plain MSE behaviour cloning ----------------
def mse_compute_loss(self, batch):
    global_cond, x1 = _cond_and_target(self, batch)
    zeros_t = torch.zeros(x1.shape[0], device=x1.device, dtype=x1.dtype)
    pred = self.unet(torch.zeros_like(x1), zeros_t, global_cond=global_cond)
    loss = F.mse_loss(pred, x1, reduction="none")
    return _masked_mean(self, loss, batch)


@torch.no_grad()
def mse_conditional_sample(self, batch_size, global_cond=None, generator=None, noise=None):
    p = next(self.parameters())
    shape = (batch_size, self.config.horizon, self.config.action_feature.shape[0])
    x = torch.zeros(shape, dtype=p.dtype, device=p.device)
    t = torch.zeros((batch_size,), dtype=p.dtype, device=p.device)
    out = self.unet(x, t, global_cond=global_cond)
    return out.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)


def apply(head: str) -> None:
    """Patch lerobot's DiffusionModel in this process. head in {diffusion, flow, mse}."""
    if head in (None, "", "diffusion"):
        return
    from lerobot.policies.diffusion import modeling_diffusion as md

    if head == "flow":
        md.DiffusionModel.compute_loss = flow_compute_loss
        md.DiffusionModel.conditional_sample = flow_conditional_sample
    elif head == "mse":
        md.DiffusionModel.compute_loss = mse_compute_loss
        md.DiffusionModel.conditional_sample = mse_conditional_sample
    else:
        raise ValueError(f"unknown head {head!r}")
    print(f"[t3_heads] DiffusionModel patched with head={head}", flush=True)


def apply_to_instance(diffusion_model, head: str) -> None:
    """Per-object version of `apply` (used by the probe, which loads several heads in one process)."""
    import types

    if head in (None, "", "diffusion"):
        return
    fns = {"flow": (flow_compute_loss, flow_conditional_sample), "mse": (mse_compute_loss, mse_conditional_sample)}
    loss_fn, sample_fn = fns[head]
    diffusion_model.compute_loss = types.MethodType(loss_fn, diffusion_model)
    diffusion_model.conditional_sample = types.MethodType(sample_fn, diffusion_model)
