"""
KrishiAI — EfficientNet-B3 trainer for the FULL 38-class PlantVillage dataset.
Designed to run on a Kaggle GPU notebook, but also runs locally.

═══════════════════════════════════════════════════════════════════════
KAGGLE SETUP (do this first in the notebook):
  1. Settings → Accelerator → GPU (T4 x2 or P100).
  2. Settings → Internet → ON   (needed to download pretrained weights).
  3. Add data → search "plantvillage" → add:
        abdallahalidev/plantvillage-dataset   (the full 38-class spMohanty mirror)
  4. First cell:
        !pip install -q timm onnx onnxruntime
  5. Then paste/run this script.

Outputs land in /kaggle/working/output/ — download these into your repo's output/ :
        model.onnx          ← served by FastAPI (main.py)
        class_names.json     ← index → class-name map (MUST match the model)
        best_model.pth       ← PyTorch checkpoint (for future fine-tuning)
        metrics.json         ← accuracy / macro-F1 / per-class report
        confusion_matrix.png
        training_curves.png
═══════════════════════════════════════════════════════════════════════
"""

import os, json, time, random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler

import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ─────────────────────────────────────────────────────────────
# CONFIG  — tweak here
# ─────────────────────────────────────────────────────────────
IMG_SIZE     = 300            # EfficientNet-B3 native res — MUST match main.py preprocessing
BATCH_SIZE   = 32             # drop to 16 if you hit CUDA OOM
EPOCHS       = 5             # PlantVillage converges fast; 5 already hits ~99.5%
LR           = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
VAL_FRAC     = 0.10
TEST_FRAC    = 0.10
SEED         = 42
MODEL_NAME   = "efficientnet_b3"

OUTPUT_DIR = Path("/kaggle/working/output" if os.path.isdir("/kaggle/working") else "output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Must match main.py — ImageNet stats, RGB, [N,3,300,300]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────
def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

seed_everything(SEED)


# ─────────────────────────────────────────────────────────────
# LOCATE DATASET
# The full dataset usually extracts to:
#   /kaggle/input/plantvillage-dataset/plantvillage dataset/color
# but folder names vary, so we auto-detect the "color" root that holds the
# class sub-folders. Falls back to a local ./PlantVillage for offline runs.
# ─────────────────────────────────────────────────────────────
def find_data_dir() -> str:
    explicit = [
        "/kaggle/input/plantvillage-dataset/plantvillage dataset/color",
        "/kaggle/input/plantvillage-dataset/color",
        "PlantVillage", "./PlantVillage",
    ]
    for c in explicit:
        if os.path.isdir(c):
            return c

    # Walk /kaggle/input and pick the dir with the most class-like subfolders,
    # preferring one literally named "color" (avoids grayscale/segmented copies).
    best, best_count = None, 0
    for root in ["/kaggle/input"]:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _ in os.walk(root):
            subdirs = [d for d in dirnames
                       if os.path.isdir(os.path.join(dirpath, d))]
            if len(subdirs) >= 15:                       # looks like a class root
                score = len(subdirs) + (1000 if os.path.basename(dirpath).lower() == "color" else 0)
                if score > best_count:
                    best, best_count = dirpath, score
    if best:
        return best
    raise FileNotFoundError(
        "Could not find the PlantVillage 'color' folder. "
        "Add the 'abdallahalidev/plantvillage-dataset' dataset on Kaggle."
    )

DATA_DIR = find_data_dir()
print(f"[data] using: {DATA_DIR}")
print(f"[device] {DEVICE}")


# ─────────────────────────────────────────────────────────────
# TRANSFORMS
#   • train: plain 300×300 resize (geometry matches inference) + light aug
#   • eval : EXACT mirror of main.py inference (Resize 300×300 → normalize)
# ─────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),               # leaves have no canonical orientation
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # field-lighting robustness
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ─────────────────────────────────────────────────────────────
# STRATIFIED 80/10/10 SPLIT
# Build one ImageFolder for labels, then split indices per-class so every
# class appears in train/val/test. Apply the right transform per split.
# ─────────────────────────────────────────────────────────────
base = datasets.ImageFolder(DATA_DIR)               # no transform — just for targets/classes
class_names = base.classes
num_classes = len(class_names)
targets = np.array(base.targets)
print(f"[data] {len(base)} images across {num_classes} classes")
assert num_classes == 38, f"Expected 38 classes, found {num_classes}. Check DATA_DIR points at the full dataset."

idx = np.arange(len(base))
train_idx, temp_idx = train_test_split(
    idx, test_size=(VAL_FRAC + TEST_FRAC), random_state=SEED, stratify=targets)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
    random_state=SEED, stratify=targets[temp_idx])

# Two ImageFolders pointing at the same data, different transforms.
train_full = datasets.ImageFolder(DATA_DIR, transform=train_tf)
eval_full  = datasets.ImageFolder(DATA_DIR, transform=eval_tf)

train_ds = Subset(train_full, train_idx)
val_ds   = Subset(eval_full,  val_idx)
test_ds  = Subset(eval_full,  test_idx)
print(f"[split] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

workers = min(4, os.cpu_count() or 2)
def make_loader(ds, shuffle):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=workers, pin_memory=(DEVICE == "cuda"),
                      persistent_workers=workers > 0)

train_loader = make_loader(train_ds, True)
val_loader   = make_loader(val_ds,   False)
test_loader  = make_loader(test_ds,  False)


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler    = GradScaler(enabled=(DEVICE == "cuda"))


# ─────────────────────────────────────────────────────────────
# TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(loader):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with autocast(enabled=(DEVICE == "cuda")):
            out = model(x)
            loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total

history = {"train_loss": [], "val_loss": [], "val_acc": []}
best_acc, best_state = 0.0, None

for epoch in range(1, EPOCHS + 1):
    model.train()
    running, t0 = 0.0, time.time()
    for bi, (x, y) in enumerate(train_loader, 1):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(DEVICE == "cuda")):
            out  = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * x.size(0)
        if bi % 100 == 0:
            print(f"  epoch {epoch} | batch {bi}/{len(train_loader)} | loss {loss.item():.4f}")
    scheduler.step()

    train_loss = running / len(train_ds)
    val_loss, val_acc = evaluate(val_loader)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    print(f"[epoch {epoch}/{EPOCHS}] train_loss={train_loss:.4f} "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({time.time()-t0:.0f}s)")

    if val_acc > best_acc:
        best_acc = val_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, OUTPUT_DIR / "best_model.pth")
        print(f"  ✓ new best (val_acc={best_acc:.4f}) saved")

# restore best weights for testing + export
if best_state is not None:
    model.load_state_dict(best_state)
print(f"[done] best val_acc = {best_acc:.4f}")


# ─────────────────────────────────────────────────────────────
# TEST-SET METRICS
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_preds(loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        with autocast(enabled=(DEVICE == "cuda")):
            out = model(x)
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)

y_true, y_pred = collect_preds(test_loader)
test_acc  = float((y_true == y_pred).mean())
macro_f1  = float(f1_score(y_true, y_pred, average="macro"))
report    = classification_report(y_true, y_pred, target_names=class_names,
                                  output_dict=True, zero_division=0)
per_class = {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")}
             for c in class_names}
print(f"[test] accuracy={test_acc:.4f}  macro_f1={macro_f1:.4f}")


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
# training curves
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history["train_loss"], label="train")
ax[0].plot(history["val_loss"],   label="val")
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(history["val_acc"], color="green")
ax[1].set_title("Val accuracy"); ax[1].set_xlabel("epoch")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=120); plt.close(fig)

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(cm, cmap="Greens")
ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
ax.set_xticklabels(class_names, rotation=90, fontsize=6)
ax.set_yticklabels(class_names, fontsize=6)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=120); plt.close(fig)


# ─────────────────────────────────────────────────────────────
# ONNX EXPORT  (single file, dynamic batch, opset 17)
# Export on CPU so the artifact is portable; main.py runs it CPU-only.
# ─────────────────────────────────────────────────────────────
model_cpu = model.to("cpu").eval()
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
onnx_path = OUTPUT_DIR / "model.onnx"
torch.onnx.export(
    model_cpu, dummy, str(onnx_path),
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17, do_constant_folding=True,
)
print(f"[onnx] exported → {onnx_path}")

# ─── verify ONNX matches PyTorch + measure CPU latency ───
import onnxruntime as ort
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
with torch.no_grad():
    torch_out = model_cpu(dummy).numpy()
onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
max_diff = float(np.abs(torch_out - onnx_out).max())
print(f"[onnx] max |torch - onnx| = {max_diff:.2e}  (should be < 1e-3)")

# CPU latency over 50 runs
runs = 50
_ = sess.run(None, {"input": dummy.numpy()})  # warmup
t0 = time.time()
for _ in range(runs):
    sess.run(None, {"input": dummy.numpy()})
infer_ms = (time.time() - t0) / runs * 1000


# ─────────────────────────────────────────────────────────────
# SAVE class_names.json + metrics.json
# class_names order == model output order — main.py relies on this.
# ─────────────────────────────────────────────────────────────
with open(OUTPUT_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

metrics = {
    "accuracy": round(test_acc, 4),
    "macro_f1": round(macro_f1, 4),
    "inference_ms_per_image": round(infer_ms, 2),
    "throughput_images_per_sec": round(1000 / infer_ms, 1),
    "total_test_images": int(len(y_true)),
    "num_classes": num_classes,
    "onnx_torch_max_diff": max_diff,
    "per_class": per_class,
}
with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n[✓] All artifacts written to", OUTPUT_DIR.resolve())
for p in sorted(OUTPUT_DIR.iterdir()):
    print("   ", p.name, f"({p.stat().st_size/1e6:.1f} MB)")
