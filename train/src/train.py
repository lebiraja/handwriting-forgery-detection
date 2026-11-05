import argparse, copy, json, math, os, random, signal, time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .data import BatchSampler, PageDataset
from .metrics import verification_metrics
from .model import PageEncoder, feature_distill, relational_distill, supcon_loss

STOP = {"flag": False}
signal.signal(signal.SIGTERM, lambda *_: STOP.update(flag=True))
signal.signal(signal.SIGINT, lambda *_: STOP.update(flag=True))
MEAN, STD = 0.485, 0.229


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_teacher(path, dev):
    from transformers import Dinov2Model
    t = Dinov2Model.from_pretrained(path, torch_dtype=torch.float16).to(dev).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


@torch.no_grad()
def teacher_embed(teacher, x):  # x: [N,1,H,W] ink-high -> [N,384]
    img = (1.0 - x).repeat(1, 3, 1, 1)
    img = F.interpolate(img, size=126, mode="bilinear", align_corners=False)
    img = ((img - MEAN) / STD).half()
    return teacher(pixel_values=img).pooler_output.float()


@torch.no_grad()
def embed_pages(model, ds, dev, bs=8, workers=8):
    model.eval()
    dl = DataLoader(ds, batch_size=bs, num_workers=workers)
    embs, ws, dsn = [], [], []
    for x, w, d in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            embs.append(model(x.to(dev, non_blocking=True)).float().cpu())
        ws += list(w); dsn += list(d)
    return torch.cat(embs).numpy(), ws, dsn


def val_metrics(model, ds, dev, workers):
    e, w, d = embed_pages(model, ds, dev, workers=workers)
    out, d = {}, np.array(d); w = np.array(w)
    for name in sorted(set(d)):
        m = d == name
        out[name] = verification_metrics(e[m], list(w[m]))
    out["mean_eer"] = float(np.mean([v["eer"] for v in out.values()]))
    return out


@torch.no_grad()
def mine_neighbours(model, ds, dev, workers, n=20):
    e, w, _ = embed_pages(model, ds, dev, workers=workers)
    ws = sorted(set(w)); w = np.array(w)
    cen = np.stack([e[w == x].mean(0) for x in ws]); cen /= np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9
    sim = cen @ cen.T
    return {ws[i]: [ws[j] for j in np.argsort(-sim[i])[1:n + 1]] for i in range(len(ws))}


def update_ema(ema, model, decay):
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
        for be, bm in zip(ema.buffers(), model.buffers()):
            be.copy_(bm)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--set", nargs="*", action="extend", default=[])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for kv in a.set:  # e.g. --set seed=1 out_dir=/app/runs/x distill.enabled=false
        k, v = kv.split("="); cur = cfg
        *path, last = k.split(".")
        for p in path:
            cur = cur[p]
        cur[last] = yaml.safe_load(v)
    out = cfg["out_dir"]; os.makedirs(out, exist_ok=True)
    yaml.safe_dump(cfg, open(f"{out}/config.yaml", "w"))
    seed_all(cfg["seed"]); dev = "cuda"
    torch.backends.cudnn.benchmark = True
    tr = PageDataset(cfg["manifest"], "train", cfg["k"], cfg["size"], True, cfg["datasets"], cfg.get("augment", True))
    va = PageDataset(cfg["manifest"], "val", cfg["k_eval"], cfg["size"], False, cfg["datasets"])
    sampler = BatchSampler(tr, cfg["writers_per_batch"], cfg["cvl_frac"], cfg["p_mine"], cfg["seed"], cfg.get("content_matched", True))
    dl = DataLoader(tr, batch_sampler=sampler, num_workers=cfg["num_workers"], persistent_workers=True, prefetch_factor=4)
    model = PageEncoder(cfg["embed_dim"], pool=cfg["pool"]).to(dev)
    ema = copy.deepcopy(model).eval()
    dis = cfg["distill"]; teacher = load_teacher(dis["teacher"], dev) if dis["enabled"] else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(sampler); warm = cfg["warmup_epochs"] * len(sampler)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler()
    ep0, best, bad, nan_count, oom_count = 0, 1.0, 0, 0, 0
    shuf = bool(cfg.get('shuffle_labels'))  # leak test
    if a.resume and os.path.exists(f"{out}/last.pt"):
        ck = torch.load(f"{out}/last.pt", map_location=dev)
        assert ck.get("seed", cfg["seed"]) == cfg["seed"], "refusing to resume a checkpoint from a different seed"
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"]); ep0, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed at epoch {ep0}", flush=True)
    t0 = time.time(); log = open(f"{out}/metrics.jsonl", "a")
    for ep in range(ep0, cfg["epochs"]):
        if ep % cfg["mining_every"] == 0 and ep > 0 and cfg["p_mine"] > 0:
            tr.train = False; sampler.set_neighbours(mine_neighbours(ema, tr, dev, cfg["num_workers"])); tr.train = True
        model.train(); tot, n = 0.0, 0; te = time.time()
        for x, w, _ in dl:
            if STOP["flag"] or (time.time() - t0) / 60 > cfg["time_limit_min"]:
                break
            try:
                x = x.to(dev, non_blocking=True)
                lab = torch.tensor([hash(s) % (2**31) for s in w], device=dev)
                if shuf: lab = lab[torch.randperm(len(lab), device=dev)]  # break page<->writer pairing
                with torch.autocast("cuda", dtype=torch.float16):
                    page, f, _ = model(x, return_all=True)
                loss = supcon_loss(page.float(), lab, cfg["temperature"])
                if teacher is not None:
                    with torch.autocast("cuda", dtype=torch.float16):
                        t = teacher_embed(teacher, x.flatten(0, 1)).view(x.shape[0], x.shape[1], -1)
                    ff = f.float()
                    l_feat = feature_distill(ff, model.distill_head, t)
                    l_rel = relational_distill(ff.flatten(0, 1), t.flatten(0, 1)) + relational_distill(page.float(), F.normalize(t.mean(1), dim=-1))
                    loss = loss + dis["w_feat"] * l_feat + dis["w_rel"] * l_rel
                if not torch.isfinite(loss):
                    nan_count += 1; opt.zero_grad(set_to_none=True)
                    if nan_count > 20: raise RuntimeError("too many non-finite losses")
                    continue
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                update_ema(ema, model, cfg["ema_decay"] if ep >= cfg["warmup_epochs"] else 0.9)
                tot += loss.item(); n += 1
            except torch.cuda.OutOfMemoryError:
                oom_count += 1; torch.cuda.empty_cache(); tr.k = max(8, tr.k // 2)
                print(f"OOM: patches per page -> {tr.k}", flush=True)
                if oom_count > 4: raise
        rec = {"epoch": ep, "loss": tot / max(1, n), "steps": n, "sec": round(time.time() - te, 1), "nan": nan_count, "oom": oom_count, "lr": opt.param_groups[0]["lr"]}
        if (ep + 1) % cfg["eval_every"] == 0 or ep == cfg["epochs"] - 1:
            vm = val_metrics(ema, va, dev, cfg["num_workers"]); rec["val"] = vm
            if vm["mean_eer"] < best:
                best, bad = vm["mean_eer"], 0; torch.save(ema.state_dict(), f"{out}/best.pt")
                json.dump(vm, open(f"{out}/best_val.json", "w"))
            else:
                bad += cfg["eval_every"]
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": ep, "best": best, "seed": cfg["seed"]}, f"{out}/last.pt")
        if STOP["flag"] or (time.time() - t0) / 60 > cfg["time_limit_min"]:
            print("stopped early (signal or time limit); resume with --resume", flush=True); break
        if bad >= cfg["patience"]:
            print("early stop: patience exhausted", flush=True); break
    print(json.dumps({"done": True, "best_mean_val_eer": best, "minutes": round((time.time() - t0) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
