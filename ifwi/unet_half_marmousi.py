# U-Net IFWI on Half Marmousi (94 x 144)
# Companion to siren_half_marmousi.py for the SIREN-vs-U-Net comparison.
# Input is a linear initial model (image); U-Net outputs the velocity model and
# plugs into the SAME rnn2D physics + same 3->5->8 Hz frequency continuation.
#
# RUN SIREN Cells 1, 2, 3 FIRST (same kernel). They load: vp_tensor, vi_tensor,
# forward_rnn, t, wGenerator, nz, nx, dz, dt, npad, vmax, device, extent, ns,
# PROJECT, CKPT. These cells reuse all of that.

# ─── UCELL 1 ── linear initial model + U-Net + pretrain to the initial model ──
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList(); self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f)); ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))
        self.final = nn.Conv2d(features[0], out_channels, 1)
    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x); s = skips[i // 2]
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([s, x], dim=1); x = self.ups[i + 1](x)
        return self.final(x)

# same normalization as SIREN (true-model stats) so the two methods are comparable
NORM_MEAN = (vp_tensor / 1000).mean()
NORM_STD  = (vp_tensor / 1000).std()

# linear initial model: velocity increases linearly with depth (supervisor's choice)
v_top = vi_tensor.squeeze().cpu().numpy()[0].mean()
v_bot = vi_tensor.squeeze().cpu().numpy()[-1].mean()
lin_col = np.linspace(v_top, v_bot, nz)
v_lin = np.repeat(lin_col[:, None], nx, axis=1)
vlin_tensor = torch.from_numpy(v_lin[None]).float().to(device)
print("Linear init range: {:.0f} - {:.0f} m/s".format(v_lin.min(), v_lin.max()))

# U-Net input = normalized linear model; output = normalized velocity
unet_in = ((vlin_tensor / 1000 - NORM_MEAN) / NORM_STD).unsqueeze(1)   # [1,1,nz,nx]
unet = UNet(1, 1).to(device)
print("U-Net params:", sum(p.numel() for p in unet.parameters()))

unet_pretrain = os.path.join(CKPT, "unet_pretrain_nz{}_nx{}.pth".format(nz, nx))
target = vlin_tensor / 1000
opt = torch.optim.Adam(unet.parameters(), lr=1e-3)
unet.train()
t0 = time.time()
for ep in range(1001):
    opt.zero_grad()
    out = unet(unet_in)
    vpred_km = (out.squeeze(1) * NORM_STD + NORM_MEAN)
    loss = ((vpred_km - target) ** 2).mean()
    loss.backward(); opt.step()
    if ep % 100 == 0 or ep == 1000:
        print("  pretrain ep {:4d}  loss {:.4e}".format(ep, loss.item()))
torch.save({"state_dict": unet.state_dict()}, unet_pretrain)

with torch.no_grad():
    pre = (unet(unet_in).squeeze() * NORM_STD + NORM_MEAN).cpu().numpy()
mae_pre = np.abs(pre - v_lin / 1000).mean()
print("U-Net pretrain done in {:.1f}s | MAE vs linear init = {:.4f} km/s".format(time.time()-t0, mae_pre))
assert mae_pre < 0.05, "U-Net pretrain didn't converge — do not proceed."

# what to check: loss should fall to ~1e-3 or lower; MAE < 0.05 km/s


# ─── UCELL 2 ── U-Net IFWI with frequency continuation (resumable) ────────────
LR = 1e-4
schedule = [(3.0, 1500), (5.0, 1500), (8.0, 2000)]

def latest_ckpt(prefix):
    ex = sorted(glob.glob(prefix + "ckpt-*.pth"),
                key=lambda p: int(p.split("ckpt-")[1].split(".pth")[0]))
    return ex[-1] if ex else None

prev_weights = unet_pretrain
t_all = time.time()
for si, (fq, niter) in enumerate(schedule):
    stage_final = os.path.join(CKPT, "unet_stage{}_freq{:.0f}_final.pth".format(si, fq))
    prefix = os.path.join(CKPT, "unet_ifwi_stage{}_freq{:.0f}-".format(si, fq))
    if os.path.exists(stage_final):
        print("Stage {} ({:.0f} Hz) already finished -> skipping".format(si, fq))
        prev_weights = stage_final
        continue

    wav = wGenerator(t, fq).ricker().to(device)
    with torch.no_grad():
        _, _, shots_fq, _ = forward_rnn(vmodel=vp_tensor, segment_wavelet=wav)

    unet = UNet(1, 1).to(device)
    unet.load_state_dict(torch.load(prev_weights)["state_dict"])
    opt = torch.optim.Adam(unet.parameters(), lr=LR)

    start = 0
    best = 1e18
    resume = latest_ckpt(prefix)
    if resume:
        ck = torch.load(resume)
        unet.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["optimizer"])
        start = ck["epoch"] + 1; best = ck["best"]
        print("Stage {} ({:.0f} Hz) resume from epoch {}".format(si, fq, start))
    else:
        print("Stage {} ({:.0f} Hz, {} epochs) start".format(si, fq, niter))

    unet.train()
    t0 = time.time()
    for ep in range(start, niter):
        opt.zero_grad()
        out = unet(unet_in)
        vpred_ms = (out.squeeze(1) * NORM_STD + NORM_MEAN) * 1000
        _, _, pred, _ = forward_rnn(vmodel=vpred_ms, segment_wavelet=wav)
        loss = ((pred - shots_fq) ** 2).mean()
        loss.backward(); opt.step()
        if loss.item() < best:
            best = loss.item()
            torch.save({"state_dict": unet.state_dict()}, stage_final.replace("_final", "_best"))
        if ep % 100 == 0 or ep == niter - 1:
            print("  ep {:4d}  loss {:.4e}".format(ep, loss.item()))
            torch.save({"epoch": ep, "best": best, "state_dict": unet.state_dict(),
                        "optimizer": opt.state_dict()}, prefix + "ckpt-{}.pth".format(ep))
    print("Stage {} done in {:.2f} h".format(si, (time.time()-t0)/3600))
    torch.save({"state_dict": unet.state_dict()}, stage_final)
    prev_weights = stage_final

print("\nAll U-Net stages finished in {:.2f} h".format((time.time()-t_all)/3600))


# ─── UCELL 3 ── evaluate best + 3-panel figure ────────────────────────────────
best_w = os.path.join(CKPT, "unet_stage2_freq8_best.pth")
unet = UNet(1, 1).to(device)
unet.load_state_dict(torch.load(best_w)["state_dict"])
unet.train()   # single-image overfit: keep batchnorm in train mode
with torch.no_grad():
    vpred_km = (unet(unet_in).squeeze() * NORM_STD + NORM_MEAN).cpu().numpy()

true_km = vp_tensor.squeeze().cpu().numpy() / 1000
lin_km  = v_lin / 1000
err_km  = np.abs(true_km - vpred_km)
mae_i = np.abs(true_km - lin_km).mean()
mae_p = err_km.mean()
rms_i = np.sqrt(((true_km - lin_km) ** 2).mean())
rms_p = np.sqrt(((true_km - vpred_km) ** 2).mean())
print("              Initial    Predicted")
print("MAE (km/s) : {:.4f}    {:.4f}".format(mae_i, mae_p))
print("RMS (km/s) : {:.4f}    {:.4f}".format(rms_i, rms_p))
print("MAE improvement : {:.1f}%".format((mae_i - mae_p) / mae_i * 100))
print("Predicted range : {:.3f} - {:.3f} km/s  (truth 1.500 - 5.500)".format(vpred_km.min(), vpred_km.max()))
np.save(os.path.join(PROJECT, "Data/unet_vpred_half_best.npy"), vpred_km)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
im0 = ax[0].imshow(true_km, vmin=1.5, vmax=5.5, extent=extent, aspect=1, cmap="RdBu_r")
ax[0].set(title="(a) True model", xlabel="Distance (km)", ylabel="Depth (km)")
plt.colorbar(im0, ax=ax[0], label="km/s")
im1 = ax[1].imshow(vpred_km, vmin=1.5, vmax=5.5, extent=extent, aspect=1, cmap="RdBu_r")
ax[1].set(title="(b) U-Net-IFWI prediction", xlabel="Distance (km)")
plt.colorbar(im1, ax=ax[1], label="km/s")
im2 = ax[2].imshow(err_km, vmin=0, vmax=err_km.max(), extent=extent, aspect=1, cmap="hot_r")
ax[2].set(title="(c) Pointwise |error|", xlabel="Distance (km)")
plt.colorbar(im2, ax=ax[2], label="km/s")
plt.suptitle("U-Net-IFWI  Half Marmousi  ns={}  MAE={:.4f}  RMS={:.4f} km/s".format(ns, mae_p, rms_p))
plt.tight_layout()
plt.savefig(os.path.join(CKPT, "UNET_half_result_3panel.png"), dpi=150, bbox_inches="tight")
plt.show()
