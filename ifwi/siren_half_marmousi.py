# SIREN-IFWI on Half Marmousi (94 x 144)
# Run cell by cell. Paste output here after each cell so we can verify before moving on.
# Checkpoints saved every 100 epochs -> re-running Cell 5 auto-resumes from the latest one.

# ─── CELL 1 ── environment, data, geometry ────────────────────────────────────
import os, sys, time, glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

PROJECT = "/home/mw/project/IFWI_Project"
sys.path.insert(0, os.path.join(PROJECT, "Codes"))

print("Python :", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU    :", torch.cuda.get_device_name(0))

torch.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(3)
os.environ["PYTHONHASHSEED"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vmodel_full = np.array(pd.read_csv(os.path.join(PROJECT, "Data/vel_marmousi_376x1151.csv")))
v_init_full = np.array(pd.read_csv(os.path.join(PROJECT, "Data/vel_marmousi_smooth400_376x1151.csv")))
v_init_full = gaussian_filter(v_init_full, sigma=50)

vmodel_res = vmodel_full[::4, ::4]
v_init_res = v_init_full[::4, ::4]
nx_half = vmodel_res.shape[1] // 2
vmodel_half = vmodel_res[:, :nx_half]
v_init_half = v_init_res[:, :nx_half]
nz, nx = vmodel_half.shape
dz = 15.0

vp_tensor = torch.from_numpy(vmodel_half[None]).float().to(device)
vi_tensor = torch.from_numpy(v_init_half[None]).float().to(device)
nv = 1

xs = torch.arange(10, nx - 5, 10, dtype=torch.long).unsqueeze(0)
ns = xs.shape[1]
xr = torch.arange(0, nx, 1, dtype=torch.long).unsqueeze(0).unsqueeze(0).repeat(nv, ns, 1)
zs = torch.full((nv, ns), 1, dtype=torch.long)
zr = torch.full((nv, ns, nx), 2, dtype=torch.long)

extent = [0, nx * dz / 1000, nz * dz / 1000, 0]

print("\nGrid shape  :", nz, "x", nx, "| dz =", dz, "m")
print("Shots       :", ns, "| spacing {:.0f} m".format((xs[0,1]-xs[0,0]).item() * dz))
print("True  vp    : {:.0f} - {:.0f} m/s".format(vp_tensor.min().item(), vp_tensor.max().item()))
print("Initial vp  : {:.0f} - {:.0f} m/s".format(vi_tensor.min().item(), vi_tensor.max().item()))

# ── what to check ─────────────────────────────────────────────────────────────
# Grid shape should be 94 x 144
# CUDA should be True
# True vp roughly 1500 - 5500 m/s
# Paste this output before running Cell 2


# ─── CELL 2 ── wavelet, stability, dispersion ─────────────────────────────────
from generator import wGenerator

freq = 8.0
dt   = 0.0019
nt   = 1000
npad = 15
t = dt * torch.arange(0, nt, dtype=torch.float32)
wavelet = wGenerator(t, freq).ricker().to(device)

vmax = vp_tensor.max().item()
vmin = vp_tensor.min().item()
stab = vmax * dt / dz
disp = vmin / (freq * dz)
print("Stability  vmax*dt/dz = {:.4f}  (< {:.4f}) -> {}".format(stab, 1/np.sqrt(2), "PASS" if stab < 1/np.sqrt(2) else "FAIL"))
print("Dispersion vmin/(f*dz) = {:.2f}  (> 5)     -> {}".format(disp, "PASS" if disp >= 5 else "FAIL"))

# ── what to check ─────────────────────────────────────────────────────────────
# Both lines must say PASS before moving on


# ─── CELL 3 ── forward model on true velocity -> observed shots ───────────────
from rnn_fd import rnn2D

forward_rnn = rnn2D(nz, nx, zs, xs, zr, xr, dz, dt,
                    npad=npad, order=2, vmax=vmax, log_para=1e-6,
                    freeSurface=True, dtype=torch.float32, device=device).to(device)
t0 = time.time()
with torch.no_grad():
    _, _, shots, _ = forward_rnn(vmodel=vp_tensor, segment_wavelet=wavelet)
print("Forward done in {:.1f}s | shots shape {} | range {:.3f}..{:.3f}".format(
    time.time()-t0, tuple(shots.shape), shots.min().item(), shots.max().item()))

# ── what to check ─────────────────────────────────────────────────────────────
# shots shape should be [1, 13, 1000, 144]
# range should be non-zero (e.g. -0.05..0.05 order of magnitude)


# ─── CELL 4 ── pretrain SIREN on smooth initial model ─────────────────────────
# This is the key fix. Without pretraining, IFWI starts from a flat block
# (~mean velocity everywhere) instead of the smooth initial model.
from ifwi_modules import IFWI2D

NORM_MEAN = (vp_tensor / 1000).mean()
NORM_STD  = (vp_tensor / 1000).std()
print("Norm  mean={:.4f}  std={:.4f}  km/s".format(NORM_MEAN.item(), NORM_STD.item()))

CKPT = os.path.join(PROJECT, "checkpoints")
os.makedirs(CKPT, exist_ok=True)
pretrain_ckpt = os.path.join(CKPT, "siren_pretrain_nz{}_nx{}.pth".format(nz, nx))

ifwi_pre = IFWI2D(
    mean=NORM_MEAN, std=NORM_STD,
    neuron=[2, 128, 128, 128, 128, 1], omega_0=30, prob=0.2,
    activation="sine", bias=True, dropout=False, outermost_linear=True,
    nz=nz, nx=nx, zs=zs, xs=xs, zr=zr, xr=xr, dz=dz, dt=dt,
    npad=npad, order=2, vmax=vmax, log_para=1e-6,
    segment_size=len(t), vpadding=None, freeSurface=True,
    dtype=torch.float32, device=device, pretrained=None, netOpt="IFWI")

target = vi_tensor / 1000
opt = torch.optim.Adam(ifwi_pre.vel_net.parameters(), lr=1e-4)
t0 = time.time()
for ep in range(2001):
    opt.zero_grad()
    out, _ = ifwi_pre.vel_net(ifwi_pre.coords)
    vpred_km = (out * NORM_STD + NORM_MEAN).squeeze(-1)
    loss = ((vpred_km - target) ** 2).mean()
    loss.backward()
    opt.step()
    if ep % 200 == 0 or ep == 2000:
        print("  ep {:5d}  loss {:.4e}".format(ep, loss.item()))

torch.save({"state_dict": ifwi_pre.vel_net.state_dict()}, pretrain_ckpt)

with torch.no_grad():
    out, _ = ifwi_pre.vel_net(ifwi_pre.coords)
    pre_km = (out * NORM_STD + NORM_MEAN).squeeze().cpu().numpy()
mae_pre = np.abs(pre_km - vi_tensor.squeeze().cpu().numpy() / 1000).mean()
print("Pretrain done in {:.1f}s | MAE vs initial = {:.4f} km/s".format(time.time()-t0, mae_pre))
assert mae_pre < 0.05, "Pretrain didn't converge — do NOT proceed to Cell 5."

# ── what to check ─────────────────────────────────────────────────────────────
# Loss should drop from ~1e-1 down to ~1e-3 or lower by ep 2000
# Final MAE vs initial must be < 0.05 (assert will stop you if not)


# ─── CELL 5 ── IFWI (resumable, saves every 100 epochs) ───────────────────────
# Set MAX_ITER=1001 for a quick sanity run first (~2h on H100 MIG).
# After that, change to 8001 and re-run — it auto-resumes from epoch 1001.
MAX_ITER = 1001
LR       = 1e-4
ALPHA    = 0

save_prefix = os.path.join(CKPT,
    "siren_ifwi_nz{}_nx{}_ns{}_dz{:.0f}_freq{:.0f}_lr{:.0e}-".format(
        nz, nx, ns, dz, freq, LR))

ifwi_model = IFWI2D(
    mean=NORM_MEAN, std=NORM_STD,
    neuron=[2, 128, 128, 128, 128, 1], omega_0=30, prob=0.2,
    activation="sine", bias=True, dropout=False, outermost_linear=True,
    nz=nz, nx=nx, zs=zs, xs=xs, zr=zr, xr=xr, dz=dz, dt=dt,
    npad=npad, order=2, vmax=vmax, log_para=1e-6,
    segment_size=len(t), vpadding=None, freeSurface=True,
    dtype=torch.float32, device=device,
    pretrained=pretrain_ckpt,
    netOpt="IFWI")

existing = sorted(glob.glob(save_prefix + "checkpoint-*.pth"),
                  key=lambda p: int(p.split("checkpoint-")[1].split(".pth")[0]))
resume = existing[-1] if existing else None
print("Resume from:", os.path.basename(resume) if resume else "scratch")

t0 = time.time()
train_loss, vpred = ifwi_model.train(
    MaxIter=MAX_ITER, vmodel=None, wavelet=wavelet, shots=shots,
    alpha=ALPHA, option=0, log_interval=100, learning_rate=LR,
    wandb=None, resume_file_name=resume, save_file_name=save_prefix)
print("\nIFWI finished in {:.2f} h".format((time.time()-t0)/3600))
np.save(os.path.join(PROJECT, "Data/siren_loss_half.npy"), np.array(train_loss, dtype=object))

# ── what to check ─────────────────────────────────────────────────────────────
# Loss should decrease over epochs (even slowly)
# After 1001 epochs run Cell 6 and check "Predicted range":
#   if deep velocities climbing toward 4.5-5.5 km/s -> good, set MAX_ITER=8001 and re-run
#   if stuck at ~3.9 km/s -> cycle-skipping, ping me before continuing


# ─── CELL 6 ── evaluate best checkpoint + 3-panel figure ─────────────────────
ckpts = sorted(glob.glob(save_prefix + "checkpoint-*.pth"))
best_ckpt, best_loss_val = None, 1e18
for c in ckpts:
    d = torch.load(c, map_location=device)
    if d["best_loss"] < best_loss_val:
        best_loss_val, best_ckpt = d["best_loss"], c
print("Best checkpoint:", os.path.basename(best_ckpt), "| loss {:.4e}".format(best_loss_val))

vpred_tensor, _ = ifwi_model.predict(resume_file_name=best_ckpt, best=True)
true_km  = vp_tensor.squeeze().cpu().numpy() / 1000
init_km  = vi_tensor.squeeze().cpu().numpy() / 1000
vpred_km = vpred_tensor.squeeze().cpu().detach().numpy() / 1000
err_km   = np.abs(true_km - vpred_km)

mae_i = np.abs(true_km - init_km).mean()
mae_p = err_km.mean()
rms_i = np.sqrt(((true_km - init_km)**2).mean())
rms_p = np.sqrt(((true_km - vpred_km)**2).mean())

print("\n              Initial    Predicted")
print("MAE (km/s) : {:.4f}    {:.4f}".format(mae_i, mae_p))
print("RMS (km/s) : {:.4f}    {:.4f}".format(rms_i, rms_p))
print("MAE improvement : {:.1f}%".format((mae_i - mae_p) / mae_i * 100))
print("Predicted range : {:.3f} - {:.3f} km/s  (truth 1.500 - 5.500)".format(vpred_km.min(), vpred_km.max()))

np.save(os.path.join(PROJECT, "Data/siren_vpred_half_best.npy"), vpred_km)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
im0 = ax[0].imshow(true_km,  vmin=1.5, vmax=5.5, extent=extent, aspect=1, cmap="RdBu_r")
ax[0].set(title="(a) True model",          xlabel="Distance (km)", ylabel="Depth (km)")
plt.colorbar(im0, ax=ax[0], label="km/s")
im1 = ax[1].imshow(vpred_km, vmin=1.5, vmax=5.5, extent=extent, aspect=1, cmap="RdBu_r")
ax[1].set(title="(b) SIREN-IFWI prediction", xlabel="Distance (km)")
plt.colorbar(im1, ax=ax[1], label="km/s")
im2 = ax[2].imshow(err_km, vmin=0, vmax=err_km.max(), extent=extent, aspect=1, cmap="hot_r")
ax[2].set(title="(c) Pointwise |error|",   xlabel="Distance (km)")
plt.colorbar(im2, ax=ax[2], label="km/s")
plt.suptitle("SIREN-IFWI  Half Marmousi  ns={}  MAE={:.4f}  RMS={:.4f} km/s".format(ns, mae_p, rms_p))
plt.tight_layout()
out_fig = os.path.join(CKPT, "SIREN_half_result_3panel.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved to:", out_fig)
