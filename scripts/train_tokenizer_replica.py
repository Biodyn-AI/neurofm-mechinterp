"""Train a BrainTokenizer from scratch (released architecture and objective, new seed).

The model, loss (mean absolute errors of the waveform and of the spectral amplitude, exp(-PCC), commitment
and 0.5 times the mean absolute error of the spectral phase, with 25 % of channels masked from the encoder
and 0.1 input noise) and the EMA k-means RVQ are the upstream ones (`braintokenizer/model.py`); the
optimiser follows `braintokenizer/config.py` (AdamW, betas 0.5/0.9, eps 1e-5, lr 2e-4, codebook lr 3e-4,
weight decay 0.01, cosine schedule with 10 % warm-up, minimum ratio 0.05). The data volume, the batch
size and the number of steps are reduced.

Training data are the processed 10-s segments of a subject-disjoint training split of the given
cohorts (split fixed by --split-seed and shared by all replicas, written to split.json); every step
draws one cohort uniformly at random and a batch of its training segments (channel counts differ
between cohorts). MEG batches use a random subset of --meg-channels sensors per step, and EEG batches
one of --eeg-channels sensors if given (the pipeline uses 32 and 16; the architecture is montage-agnostic
and the objective already hides a random 25 % of sensors).
Held-out subjects are never seen in training.

Three changes to the upstream recipe are enabled by default:
(1) initialisation (--init fixed): SEANet convolution biases are set to zero and the weights of
the sensor-position MLP and of the encoder key projection are drawn with standard deviation 0.2
instead of 0.02. With the upstream initialisation the encoder output is dominated by a constant
vector, attention over sensors is uniform, all sources coincide and training does not leave this
state; (2) dead-code replacement (disabled by --no-proper-dead-reset): the upstream EMA codebook replaces unused
codewords only in `embed`, which the same forward pass overwrites from `embed_avg/cluster_size`, so
replacements never take effect; the fix also resets `embed_avg` and `cluster_size`; (3) for numerical
stability, the gradient norm is clipped at 1 (--grad-clip), training runs in single precision instead of
bfloat16, and the warm-up is linear instead of the upstream logarithmic one.

Writes `<out>/model_cfg.json`, `<out>/init.pt` (the weights at step 0, loaded as `<out>@init`), `<out>/final.pt`,
`<out>/train_log.csv` and `<out>/split.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neurofm_mechinterp.cohorts import get_cohort
from neurofm_mechinterp.loading import _configure_upstream_imports, _load_config
from neurofm_mechinterp.paths import CHECKPOINT_ROOT, REPO_ROOT, RESULTS_ROOT
from neurofm_mechinterp.replicas import add_decoder_source_embedding


def subject_split(cohorts: list[str], train_fraction: float, seed: int) -> dict:
    """Stratified (by group) subject split per cohort."""
    rng = random.Random(seed)
    split = {}
    for name in cohorts:
        cohort = get_cohort(name)
        subjects = sorted({r["subject"] for r in cohort.load_metadata()})
        groups = cohort.groups()
        by_group: dict[str, list[str]] = {}
        for s in subjects:
            by_group.setdefault(groups.get(s, "unknown"), []).append(s)
        train, held = [], []
        for members in by_group.values():
            members = sorted(members)
            rng.shuffle(members)
            k = int(round(train_fraction * len(members)))
            train += members[:k]
            held += members[k:]
        split[name] = {"train": sorted(train), "heldout": sorted(held)}
    return split


def load_segments(cohort_name: str, subjects: set[str]) -> dict[str, torch.Tensor]:
    """Load all segments of the given subjects into memory (x as float16)."""
    rows = [r for r in get_cohort(cohort_name).load_metadata() if r["subject"] in subjects]
    xs, pos, types = [], None, None
    for row in rows:
        payload = torch.load(REPO_ROOT / row["path"], map_location="cpu", weights_only=True)
        xs.append(payload["x"].to(torch.float16))
        pos, types = payload["pos"], payload["sensor_type"]
    return {"x": torch.stack(xs), "pos": pos, "sensor_type": types}


@torch.no_grad()
def apply_fixed_init(model) -> None:
    """Zero SEANet conv biases; std 0.2 for the sensor-position MLP and the encoder key projection."""
    for name, param in model.named_parameters():
        if "seanet" in name and name.endswith(".bias") and "lstm" not in name:
            param.zero_()
    for module in model.sensor_embed.pos_embedding_layer:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.2)
    torch.nn.init.trunc_normal_(model.encoder.k_proj.weight, std=0.2)


def install_proper_dead_reset(model) -> None:
    """Make EMA dead-code replacement effective by also resetting embed_avg and cluster_size."""
    import types

    from model_utils.vq import sample_vectors  # noqa: PLC0415

    def replace(self, samples, mask):
        new = sample_vectors(samples, self.codebook_size)
        weight = max(float(self.threshold_ema_dead_code), 1.0)
        self.embed.data.copy_(torch.where(mask[..., None], new, self.embed))
        self.embed_avg.data.copy_(torch.where(mask[..., None], new * weight, self.embed_avg))
        self.cluster_size.data.copy_(torch.where(mask, torch.full_like(self.cluster_size, weight), self.cluster_size))

    for layer in model.quantizer.rvq.layers:
        layer._codebook.replace_ = types.MethodType(replace, layer._codebook)


def lr_factor(step: int, total: int, warmup_ratio: float = 0.1, warmup_min: float = 0.1, cos_min: float = 0.05) -> float:
    warmup = max(1, int(total * warmup_ratio))
    if step < warmup:
        return warmup_min + (1.0 - warmup_min) * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return cos_min + (1.0 - cos_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def heldout_mse(model, data: dict, device: torch.device, n: int, batch: int) -> float:
    model.eval()
    idx = torch.arange(min(n, data["x"].shape[0]))
    total, count = 0.0, 0
    for i in range(0, len(idx), batch):
        x = data["x"][idx[i:i + batch]].float().to(device)
        b = x.shape[0]
        pos = data["pos"].unsqueeze(0).expand(b, -1, -1).to(device)
        st = data["sensor_type"].unsqueeze(0).expand(b, -1).to(device)
        out = model.visualize(x, pos, st)
        total += float(((out["x_rec"] - out["x"]) ** 2).mean(dim=(1, 2, 3)).sum())
        count += b
    model.train()
    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cohorts", nargs="+", default=["pd_ucsd", "pd_iowa", "dementia", "mdd_meg"])
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-eeg", type=int, default=16)
    parser.add_argument("--batch-meg", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--codebook-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--n-sources", type=int, default=None, help="override the number of latent sources")
    parser.add_argument("--meg-channels", type=int, default=64, help="random sensor subset per MEG training batch")
    parser.add_argument("--eeg-channels", type=int, default=None, help="random sensor subset per EEG training batch (default: all)")
    parser.add_argument("--init", choices=["fixed", "upstream"], default="fixed")
    parser.add_argument("--no-proper-dead-reset", action="store_true")
    parser.add_argument("--decoder-source-embedding", action="store_true",
                        help="variant: learned per-source embedding added to the decoder's key/value input")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or RESULTS_ROOT / "tokenizer_replicas" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    split = subject_split(args.cohorts, args.train_fraction, args.split_seed)
    (out / "split.json").write_text(json.dumps(split, indent=1))

    _configure_upstream_imports()
    from braintokenizer.model import BrainTokenizer  # noqa: PLC0415

    config = _load_config(CHECKPOINT_ROOT / "braintokenizer")
    if args.n_sources is not None:
        config["n_neuro"] = args.n_sources
    if args.decoder_source_embedding:
        config["decoder_source_embedding"] = True
    (out / "model_cfg.json").write_text(json.dumps(config, indent=2))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    model = BrainTokenizer(**config)
    if args.init == "fixed":
        apply_fixed_init(model)
    if not args.no_proper_dead_reset:
        install_proper_dead_reset(model)
    if args.decoder_source_embedding:
        add_decoder_source_embedding(model)
    torch.save(model.state_dict(), out / "init.pt")  # step-0 weights (geometry at initialization)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.get_parameters_groups(lr=args.lr, codebook_lr=args.codebook_lr, weight_decay=args.weight_decay),
        betas=(0.5, 0.9), eps=1e-5,
    )
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    train_data, held_data = {}, {}
    for name in args.cohorts:
        train_data[name] = load_segments(name, set(split[name]["train"]))
        held_data[name] = load_segments(name, set(split[name]["heldout"]))
        print(f"[replica] {name}: {train_data[name]['x'].shape[0]} train / {held_data[name]['x'].shape[0]} held-out segments", flush=True)

    rng = np.random.default_rng(args.seed)
    log, start = [], time.time()
    running: dict[str, float] = {}
    for step in range(args.steps):
        factor = lr_factor(step, args.steps)
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * factor
        name = args.cohorts[rng.integers(len(args.cohorts))]
        data = train_data[name]
        bsz = args.batch_meg if get_cohort(name).modality == "MEG" else args.batch_eeg
        idx = torch.from_numpy(rng.integers(0, data["x"].shape[0], size=bsz))
        x, pos, st = data["x"][idx].float(), data["pos"], data["sensor_type"]
        n_keep = args.meg_channels if get_cohort(name).modality == "MEG" else args.eeg_channels
        if n_keep is not None and n_keep < x.shape[1]:
            channels = torch.from_numpy(np.sort(rng.choice(x.shape[1], size=n_keep, replace=False)))
            x, pos, st = x[:, channels], pos[channels], st[channels]
        x = x.to(device)
        pos = pos.unsqueeze(0).expand(bsz, -1, -1).to(device)
        st = st.unsqueeze(0).expand(bsz, -1).to(device)
        output, _ = model(x, pos, st)
        loss = output["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        for key in ("loss", "time_loss", "pcc", "amp_loss", "phase_loss", "commitment_loss"):
            running[key] = running.get(key, 0.0) + float(output[key].detach())
        if (step + 1) % args.log_every == 0:
            entry = {"step": step + 1, "lr_factor": factor, "seconds": time.time() - start,
                     **{k: v / args.log_every for k, v in running.items()}}
            running = {}
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                for cname in args.cohorts:
                    bs = 4 if get_cohort(cname).modality == "MEG" else args.batch_eeg  # full-montage MEG eval
                    entry[f"heldout_mse_{cname}"] = heldout_mse(model, held_data[cname], device, n=64, batch=bs)
                torch.save(model.state_dict(), out / "final.pt")
            log.append(entry)
            pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
            print(f"[replica seed{args.seed}] step {step + 1}/{args.steps} loss {entry['loss']:.3f} pcc {entry['pcc']:.3f} "
                  f"({entry['seconds']:.0f}s)" + "".join(f" {k}={v:.3f}" for k, v in entry.items() if k.startswith("heldout")), flush=True)
    torch.save(model.state_dict(), out / "final.pt")
    print(f"[replica seed{args.seed}] done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
