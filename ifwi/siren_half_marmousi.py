"""
================================================================================
SIREN-IFWI on Half Marmousi (94 x 144) — paper-faithful, corrected pipeline
================================================================================

Fixes the failures seen in the previous notebooks:
  * The SIREN is ALWAYS pretrained on the smooth initial model first, so the
    inversion starts from the real initial model (not a flat ~mean block).
  * Pretrain and IFWI use IDENTICAL mean/std normalization.
  * Manual pretrain loop avoids the buggy netOpt='IRN' path in ifwi_modules.py.
  * 8001 IFWI epochs (paper count), resumable, best-loss checkpoint tracked.
  * Clean 3-panel deliverable: truth | prediction | |error|  + MAE/RMS.

Run this on the ModelWhale H100 environment. Paths assume the standard layout:
    /home/mw/project/IFWI_Project/Codes/   (rnn_fd.py, ifwi_modules.py, generator.py, plot_functions.py)
    /home/mw/project/IFWI_Project/Data/    (vel_marmousi_376x1151.csv, vel_marmousi_smooth400_376x1151.csv)
    /home/mw/project/IFWI_Project/checkpoints/

The file is organized as numbered CELLS — paste each into its own notebook cell.
================================================================================
"""

# ============================================================================
# CELL 1 — Environment, data loading, half-model geometry
# ============================================================================
import os, sys, time, copy, glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

PROJECT = "/home/mw/project/IFWI_Project"
sys.path.insert(0, os.path.join(PROJECT, "Codes"))

print("=" * 55)
print("Python :", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU    :", torch.cuda.get_device_name(0))
print("=" * 55)

# reproducibility
torch.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(3)
os.environ["PYTHONHASHSEED"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---- load full Marmousi + smoothed initial ----
vmodel_full = np.array(pd.read_csv(os.path.join(PROJECT, "Data/vel_marmousi_376x1151.csv")))
v_init_full = np.array(pd.read_csv(os.path.join(PROJECT, "Data/vel_marmousi_smooth400_376x1151.csv")))
v_init_full = gaussian_filter(v_init_full, sigma=50)            # paper-faithful smoothing

# ---- subsample by 4, then take the LEFT HALF -> 94 x 144 ----
sample_interval = 4
dz = 15.0
vmodel_res = vmodel_full[::sample_interval, ::sample_interval]
v_init_res = v_init_full[::sample_interval, ::sample_interval]
nx_half = vmodel_res.shape[1] // 2                              # 144
vmodel_half = vmodel_res[:, :nx_half]
v_init_half = v_init_res[:, :nx_half]
nz, nx = vmodel_half.shape
print("\nHalf-Marmousi shape :", nz, "x", nx)
print("Grid spacing        : {:.1f} m".format(dz))

vp_tensor = torch.from_numpy(vmodel_half[None]).float().to(device)   # truth   [1, nz, nx]
vi_tensor = torch.from_numpy(v_init_half[None]).float().to(device)   # initial [1, nz, nx]
nv = 1

# ---- acquisition geometry: 13 surface shots ----
xs = torch.arange(10, nx - 5, 10, dtype=torch.long).unsqueeze(0)     # [1, ns]
ns = xs.shape[1]
xr = torch.arange(0, nx, 1, dtype=torch.long).unsqueeze(0).unsqueeze(0).repeat(nv, ns, 1)
zs = torch.full((nv, ns), 1, dtype=torch.long)                       # source depth 1 grid
zr = torch.full((nv, ns, nx), 2, dtype=torch.long)                   # receiver depth 2 grid
print("Number of shots     :", ns, "| spacing {:.0f} m".format((xs[0, 1] - xs[0, 0]).item() * dz))
print("True  vp range      : {:.0f} - {:.0f} m/s".format(vp_tensor.min().item(), vp_tensor.max().item()))
print("Init  vp range      : {:.0f} - {:.0f} m/s".format(vi_tensor.min().item(), vi_tensor.max().item()))

extent = [0, nx * dz / 1000, nz * dz / 1000, 0]


# ============================================================================
# CELL 2 — Wavelet + stability / dispersion checks
# ============================================================================
from generator import wGenerator

freq = 8.0
dt = 0.0019
nt = 1000
npad = 15
t = dt * torch.arange(0, nt, dtype=torch.float32)
wavelet = wGenerator(t, freq).ricker().to(device)

vmax = vp_tensor.max().item()
vmin = vp_tensor.min().item()
stab = vmax * dt / dz
disp = vmin / (freq * dz)
print("Stability  vmax*dt/dz = {:.4f}  (must be < {:.4f})  -> {}".format(
    stab, 1 / np.sqrt(2), "PASS" if stab < 1 / np.sqrt(2) else "FAIL"))
print("Dispersion vmin/(f*dz) = {:.2f}  (want > 5)        -> {}".format(
    disp, "PASS" if disp >= 5 else "FAIL"))


# ============================================================================
# CELL 3 — Forward modeling -> observed shot gathers (the inversion target)
# ============================================================================
from rnn_fd import rnn2D

forward_rnn = rnn2D(nz, nx, zs, xs, zr, xr, dz, dt,
                    npad=npad, order=2, vmax=vmax, log_para=1e-6,
                    freeSurface=True, dtype=torch.float32, device=device).to(device)
t0 = time.time()
with torch.no_grad():
    _, _, shots, _ = forward_rnn(vmodel=vp_tensor.to(device), segment_wavelet=wavelet)
print("Forward modeling done in {:.1f}s | shots {} | range {:.3f}..{:.3f}".format(
    time.time() - t0, tuple(shots.shape), shots.min().item(), shots.max().item()))


# ============================================================================
# CELL 4 — PRETRAIN the SIREN on the smooth initial model   (THE KEY FIX)
# ----------------------------------------------------------------------------
# Without this, IFWI2D starts the inversion from a flat ~mean velocity block.
# We use a manual loop (the netOpt='IRN' path in ifwi_modules.py is buggy).
# mean/std are computed once here and REUSED unchanged in the IFWI stage.
# ============================================================================
from ifwi_modules import IFWI2D

# Paper-faithful normalization (true-model statistics, as in the paper notebook).
# Pretraining still forces the start point to be the smooth initial model.
NORM_MEAN = (vp_tensor / 1000).mean()
NORM_STD = (vp_tensor / 1000).std()
print("Normalization  mean={:.4f}  std={:.4f}  km/s".format(NORM_MEAN.item(), NORM_STD.item()))

CKPT = os.path.join(PROJECT, "checkpoints")
os.makedirs(CKPT, exist_ok=True)
pretrain_ckpt = os.path.join(CKPT, "HalfMarmousi_SIREN_PRETRAIN_nz{}_nx{}.pth".format(nz, nx))

# Build an IFWI2D purely to get an IRN (SIREN) with the right architecture.
ifwi_pretrain = IFWI2D(
    mean=NORM_MEAN, std=NORM_STD,
    neuron=[2, 128, 128, 128, 128, 1], omega_0=30, prob=0.2,
    activation="sine", bias=True, dropout=False, outermost_linear=True,
    nz=nz, nx=nx, zs=zs, xs=xs, zr=zr, xr=xr, dz=dz, dt=dt,
    npad=npad, order=2, vmax=vp_tensor.max(), log_para=1e-6,
    segment_size=len(t), vpadding=None, freeSurface=True,
    dtype=torch.float32, device=device, pretrained=None, netOpt="IFWI")

target = (vi_tensor / 1000)                                # smooth initial, km/s, [1, nz, nx]
opt = torch.optim.Adam(ifwi_pretrain.vel_net.parameters(), lr=1e-4)
MAX_PRE = 2001
t0 = time.time()
for ep in range(MAX_PRE):
    opt.zero_grad()
    out, _ = ifwi_pretrain.vel_net(ifwi_pretrain.coords)    # [1, nz, nx, 1] normalized
    vpred_km = (out * NORM_STD + NORM_MEAN).squeeze(-1)     # [1, nz, nx] km/s
    loss = ((vpred_km - target) ** 2).mean()
    loss.backward()
    opt.step()
    if ep % 200 == 0 or ep == MAX_PRE - 1:
        print("  pretrain epoch {:5d}  loss {:.4e}".format(ep, loss.item()))

torch.save({"state_dict": ifwi_pretrain.vel_net.state_dict()}, pretrain_ckpt)
with torch.no_grad():
    out, _ = ifwi_pretrain.vel_net(ifwi_pretrain.coords)
    pre_km = (out * NORM_STD + NORM_MEAN).squeeze().cpu().numpy()
mae_pre = np.abs(pre_km - vi_tensor.squeeze().cpu().numpy() / 1000).mean()
print("Pretrain done in {:.1f}s | SIREN-vs-init MAE = {:.4f} km/s (need < 0.05)".format(
    time.time() - t0, mae_pre))
assert mae_pre < 0.05, "Pretrain did not converge to the initial model — do not start IFWI."


# ============================================================================
# CELL 5 — IFWI from the pretrained SIREN  (8001 epochs, resumable)
# ----------------------------------------------------------------------------
# Re-running this cell automatically RESUMES from the latest checkpoint, so a
# ~12-14h / 8001-epoch run can be done in chunks. alpha=0 = paper Section 0.
# ============================================================================
MAX_ITER = 1001     # SANITY TEST first (~2 h). If the predicted range climbs past ~4 km/s, set this to 8001 and re-run (it resumes).
LR = 1e-4
ALPHA = 0                                                   # paper Section 0 (no TV), as you chose

save_prefix = os.path.join(
    CKPT, "HalfMarmousi_SIREN_IFWI_nz{}_nx{}_ns{}_dz{:.0f}_freq{:.0f}_lr{:.0e}-".format(
        nz, nx, ns, dz, freq, LR))

ifwi_model = IFWI2D(
    mean=NORM_MEAN, std=NORM_STD,
    neuron=[2, 128, 128, 128, 128, 1], omega_0=30, prob=0.2,
    activation="sine", bias=True, dropout=False, outermost_linear=True,
    nz=nz, nx=nx, zs=zs, xs=xs, zr=zr, xr=xr, dz=dz, dt=dt,
    npad=npad, order=2, vmax=vp_tensor.max(), log_para=1e-6,
    segment_size=len(t), vpadding=None, freeSurface=True,
    dtype=torch.float32, device=device,
    pretrained=pretrain_ckpt,                               # <-- start from the smooth initial model
    netOpt="IFWI")

# auto-resume from the newest checkpoint if one exists
existing = sorted(glob.glob(save_prefix + "checkpoint-*.pth"),
                  key=lambda p: int(p.split("checkpoint-")[1].split(".pth")[0]))
resume = existing[-1] if existing else None
print("Resuming from:", os.path.basename(resume) if resume else "scratch (epoch 0)")

t0 = time.time()
train_loss, vpred = ifwi_model.train(
    MaxIter=MAX_ITER, vmodel=None, wavelet=wavelet, shots=shots,
    alpha=ALPHA, option=0, log_interval=100, learning_rate=LR,
    wandb=None, resume_file_name=resume, save_file_name=save_prefix)
print("\nIFWI finished in {:.2f} hours".format((time.time() - t0) / 3600))
np.save(os.path.join(PROJECT, "Data/siren_loss_half_8001.npy"), np.array(train_loss, dtype=object))


# ============================================================================
# CELL 6 — Evaluate BEST checkpoint + supervisor's 3-panel figure
# ----------------------------------------------------------------------------
# After the 1001-epoch test: read "Predicted range" below.
#   * climbing toward ~4.5-5.5 km/s in the deep section  -> set MAX_ITER=8001
#     in CELL 5 and re-run CELL 5 (auto-resumes from 1001), then re-run CELL 6.
#   * still capped near ~3.9 km/s  -> single-freq cycle-skipping confirmed;
#     ping me and we switch on the frequency-continuation fallback.
# ============================================================================
ckpts = sorted(glob.glob(save_prefix + "checkpoint-*.pth"))
best_ckpt, best_loss = None, 1e18
for c in ckpts:
    d = torch.load(c, map_location=device)
    if d["best_loss"] < best_loss:
        best_loss, best_ckpt = d["best_loss"], c
print("Best checkpoint:", os.path.basename(best_ckpt), "| best_loss {:.4e}".format(best_loss))

vpred_tensor, _ = ifwi_model.predict(resume_file_name=best_ckpt, best=True)
true_km = vp_tensor.squeeze().cpu().numpy() / 1000
init_km = vi_tensor.squeeze().cpu().numpy() / 1000
vpred_km = vpred_tensor.squeeze().cpu().detach().numpy() / 1000
err_km = np.abs(true_km - vpred_km)

mae_init = np.abs(true_km - init_km).mean()
mae_pred = err_km.mean()
rms_init = np.sqrt(((true_km - init_km) ** 2).mean())
rms_pred = np.sqrt(((true_km - vpred_km) ** 2).mean())
print("\n               Initial    Predicted")
print("MAE (km/s) :   {:.4f}     {:.4f}".format(mae_init, mae_pred))
print("RMS (km/s) :   {:.4f}     {:.4f}".format(rms_init, rms_pred))
print("MAE improvement : {:.1f}%".format((mae_init - mae_pred) / mae_init * 100))
print("Predicted range : {:.3f} - {:.3f} km/s (truth 1.500 - 5.500)".format(vpred_km.min(), vpred_km.max()))
np.save(os.path.join(PROJECT, "Data/siren_vpred_half_best.npy"), vpred_km)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
im0 = ax[0].imshow(true_km, vmin=1.5, vmax=5.5, extent=extent, aspect=1, cmap="RdBu_r")
ax[0].set(title="(a) True model", xlabel="Distance (km)", ylabel="Depth (km)")
plt.colorbar(im0, ax=ax[0], label="km/s")
im1 = ax[1].imshow(vpred_km, vmin=1.5, vmax=5.5, extent=extent, aspect=1, cmap="RdBu_r")
ax[1].set(title="(b) SIREN-IFWI prediction", xlabel="Distance (km)")
plt.colorbar(im1, ax=ax[1], label="km/s")
im2 = ax[2].imshow(err_km, vmin=0, vmax=err_km.max(), extent=extent, aspect=1, cmap="hot_r")
ax[2].set(title="(c) Pointwise |error|", xlabel="Distance (km)")
plt.colorbar(im2, ax=ax[2], label="km/s")
plt.suptitle("SIREN-IFWI | Half Marmousi | ns={} | MAE={:.4f} RMS={:.4f} km/s".format(
    ns, mae_pred, rms_pred), fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(CKPT, "SIREN_half_result_3panel.png"), dpi=150, bbox_inches="tight")
plt.show()
