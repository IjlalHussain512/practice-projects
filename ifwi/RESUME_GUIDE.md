# How to resume this work later (without rerunning training)

All trained weights live in `/home/mw/project/IFWI_Project/checkpoints/`
(persistent ModelWhale project storage). Kernel memory is lost between
sessions, but the disk files are not. Re-running the fast cells rebuilds the
variables; the training cells skip/resume from the saved checkpoints.

## Routine to get back to results (~2 min, no training)

1. Reopen the environment, start the kernel.
2. Run the fast setup cells:
   - SIREN `Cell 1, 2, 3` (data, wavelet, physics engine)
   - SIREN `Cell 4` (re-saves pretrain, ~9 s)
   - U-Net `UCELL 1` (rebuilds linear model + U-Net class, ~30 s)
3. Run the training cell (`Cell 5` / `UCELL 2`):
   - prints `already finished -> skipping` for every completed stage
   - zero epochs rerun
4. Run the eval/comparison cells to regenerate any figure instantly.

## Editing / extending

- **More epochs in a stage**: raise that stage's epoch count AND delete its
  `*_stageN_*_final.pth`. It resumes from the last checkpoint, runs only the
  new epochs.
- **Redo one stage**: delete that stage's `*_stageN_*` checkpoints + final
  file. Earlier stages stay cached.
- **Change geometry/frequency (upstream)**: invalidates everything — delete
  all `*_final.pth` and `*ckpt*` files and rerun.

## Before closing a session, verify persistence

```python
import glob, os
CKPT = "/home/mw/project/IFWI_Project/checkpoints"
print("Persistent:", CKPT.startswith("/home/mw/project"))
for p in sorted(glob.glob(os.path.join(CKPT, "*_final.pth"))):
    print(" ", os.path.basename(p), "{:.1f} MB".format(os.path.getsize(p)/1e6))
```

Expect 6 `*_final.pth` files: 3 SIREN (stage0/1/2 freq3/5/8) + 3 U-Net.
If they are listed under `/home/mw/project/...`, the work is safe to close.

## What is in git (survives independently of ModelWhale)

- `ifwi/siren_half_marmousi.py`, `ifwi/unet_half_marmousi.py` — the code
- `ifwi/RESULTS.md` — results writeup
- `ifwi/RESUME_GUIDE.md` — this file

The trained weights (`.pth`) and figures (`.png`) are NOT in git — they live
on ModelWhale storage. If you want them backed up off ModelWhale, download
the `checkpoints/` folder.
