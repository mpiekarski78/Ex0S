"""Train the frozen species prior: stripped Shakespeare + NOTE-use, no lord/love facts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_memory.byte_lm import LMConfig, TinyByteLM, hash_lm, save_lm
from three_memory.bytes_util import encode_bytes
from three_memory.corpus import build_train_text, load_shakespeare, make_note_example, strip_probe_facts


@torch.no_grad()
def note_follow_acc(model: TinyByteLM, device: torch.device, rng: np.random.Generator, n: int = 64) -> float:
    ok = 0
    for _ in range(n):
        ex = make_note_example(rng)
        body, _, rest = ex.partition("\n")
        rest = rest.rstrip("\n")
        target = rest[-1]
        prefix = rest[:-1]
        ctx = body + "\n" + prefix
        ids = encode_bytes(ctx)
        t = torch.tensor([ids], dtype=torch.long, device=device)
        if t.size(1) < 64:
            pad = torch.full((1, 64 - t.size(1)), ord("\n"), dtype=torch.long, device=device)
            t = torch.cat([pad, t], dim=1)
        elif t.size(1) > 64:
            t = t[:, -64:]
        logits, _ = model(t)
        pred = int(logits[0, -1].argmax().item())
        ok += int(pred == ord(target))
    return ok / n


def packed_note_batch(rng: np.random.Generator, batch: int, block: int):
    """Aligned NOTE→copy examples. Last input byte is end of prefix; target is ch."""
    xs = np.full((batch, block), ord("\n"), dtype=np.int64)
    ys = np.full((batch, block), ord("\n"), dtype=np.int64)
    last_target = np.zeros(batch, dtype=np.int64)
    for i in range(batch):
        ex = make_note_example(rng)
        body, _, rest = ex.partition("\n")
        rest = rest.rstrip("\n")
        ch = rest[-1]
        pfx = rest[:-1]
        ctx = f"{body}\n{pfx}"  # no answer byte in the input
        ids = encode_bytes(ctx)
        tgt = encode_bytes(ctx[1:] + ch)  # next-byte labels; last label is ch
        if len(ids) > block:
            ids = ids[-block:]
            tgt = tgt[-block:]
        xs[i, block - len(ids) :] = ids
        ys[i, block - len(tgt) :] = tgt
        last_target[i] = ord(ch)
    return xs, ys, last_target


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--shakespeare", type=str, default="")
    p.add_argument("--plain", action="store_true", help="Language only: no NOTE-copy training (v2 prior).")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()
    if not args.out:
        args.out = str(
            REPO_ROOT / "checkpoints" / ("prior_plain.pt" if args.plain else "prior.pt")
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    raw = load_shakespeare(Path(args.shakespeare) if args.shakespeare else None)
    if args.plain:
        text = strip_probe_facts(raw)
    else:
        text = build_train_text(raw, rng, n_notes=4000)
    stripped = strip_probe_facts(raw)
    low = stripped.lower()
    assert "lord" not in low and "love" not in low and "my lo" not in stripped
    data = np.array(encode_bytes(text), dtype=np.int64)

    cfg = LMConfig(seed=args.seed)
    model = TinyByteLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    losses = []
    n = len(data) - args.block - 1
    model.train()
    for step in range(1, args.steps + 1):
        ix = rng.integers(0, n, size=args.batch)
        x_lang = np.stack([data[i : i + args.block] for i in ix])
        y_lang = np.stack([data[i + 1 : i + 1 + args.block] for i in ix])
        if args.plain:
            x = torch.tensor(x_lang, dtype=torch.long, device=device)
            y = torch.tensor(y_lang, dtype=torch.long, device=device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            loss_note_v = float("nan")
        else:
            nx, ny, nlast = packed_note_batch(rng, args.batch, args.block)
            x = torch.tensor(np.concatenate([x_lang, nx], 0), dtype=torch.long, device=device)
            y = torch.tensor(np.concatenate([y_lang, ny], 0), dtype=torch.long, device=device)
            logits, _ = model(x)
            loss_all = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            note_logits = logits[args.batch :, -1]
            loss_note = F.cross_entropy(
                note_logits, torch.tensor(nlast, dtype=torch.long, device=device)
            )
            loss = loss_all + 4.0 * loss_note
            loss_note_v = float(loss_note.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if step == 1 or step % 100 == 0 or step == args.steps:
            model.eval()
            acc = note_follow_acc(model, device, rng)
            model.train()
            print(
                f"step {step:4d} loss={losses[-1]:.3f} note_ce={loss_note_v:.3f} note_acc={acc:.3f}",
                flush=True,
            )

    model.eval()
    acc = note_follow_acc(model, device, rng, n=200)
    extra = {
        "final_loss": losses[-1],
        "note_follow_acc": acc,
        "steps": args.steps,
        "device": str(device),
        "weight_hash": hash_lm(model),
        "corpus_bytes": int(len(data)),
        "stripped": True,
        "plain": bool(args.plain),
        "note_copy_trained": not bool(args.plain),
    }
    save_lm(model, args.out, extra=extra)
    meta_path = Path(args.out).with_suffix(".json")
    meta_path.write_text(json.dumps(extra, indent=2) + "\n", encoding="utf-8")
    print("saved", args.out)
    print(json.dumps(extra, indent=2))
    if not args.plain and acc < 0.7:
        raise SystemExit(f"NOTE-follow accuracy too low ({acc:.3f}); prior cannot use S.")
    if args.plain and acc >= 0.5:
        print("WARNING: plain prior still follows NOTE; check corpus leakage.")


if __name__ == "__main__":
    main()
