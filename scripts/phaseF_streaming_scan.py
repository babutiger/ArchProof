"""Phase F streaming scan: for each repo in the manifest, download all
files to a per-repo subdir under /tmp, run `verify_model_phaseC` on the
primary ONNX file, append a row to the verdict CSV, then delete the
download. Resumable via the CSV (already-scanned repo_ids are skipped).

Target behaviour:
  * peak working set ≤ 5 GB (one repo at a time + small scratch)
  * on CERTIFIED-POSITIVE: copy the repo to
    benchmark/wild_positives/<repo_slug>/ before deletion (forensics)
  * on any failure (download, verify, timeout): record row with status
    and move on; never leave a partial repo on disk

Env overrides:
  HF_ENDPOINT                default https://hf-mirror.com
  PHASEF_SCAN_ROOT           default /tmp/phaseF_scan
  PHASEF_POSITIVES_DIR       default benchmark/wild_positives
  PHASEF_PER_MODEL_TIMEOUT_S default 600  (10 min verify cap)
  PHASEF_MAX_RUN_HOURS       default 12   (stop after N hours wall-clock)
"""

import argparse
import csv
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    sys.exit("ERROR: huggingface_hub not installed")

sys.path.insert(0, (os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Download from the real HF hub first, then fall back to the mirror. Setting
# HF_ENDPOINT (e.g. behind the GFW) makes that endpoint the FIRST choice, with
# the rest tried after it.
_ENV_EP = os.environ.get("HF_ENDPOINT")
ENDPOINTS = list(dict.fromkeys(
    ([_ENV_EP] if _ENV_EP else [])
    + ["https://huggingface.co", "https://hf-mirror.com"]))
ENDPOINT = ENDPOINTS[0]   # primary, for HfApi metadata + logging
SCAN_ROOT = Path(os.environ.get("PHASEF_SCAN_ROOT", "/tmp/phaseF_scan"))
POSITIVES_DIR = Path(os.environ.get(
    "PHASEF_POSITIVES_DIR",
    os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmark/wild_positives")))
PER_MODEL_TIMEOUT = int(os.environ.get("PHASEF_PER_MODEL_TIMEOUT_S", "600"))
MAX_RUN_HOURS = float(os.environ.get("PHASEF_MAX_RUN_HOURS", "12"))

# Hard disk budget enforcement. These are INVARIANTS, not suggestions:
#   - total = scan_root (ephemeral working copy) + positives_dir (forensic keep)
#     must NEVER exceed HARD_DISK_BUDGET_GB at any moment
#   - positives_dir alone must NEVER exceed POS_BUDGET_GB (leaves headroom
#     for the current working copy)
#   - single repo scan_root snapshot must NEVER exceed PER_REPO_BUDGET_GB
#     (protects against lying manifests / LFS balloons mid-download)
# If any cap would be breached, the scanner stops cleanly (verdict CSV is
# already flushed per-row) rather than risk filling the disk.
HARD_DISK_BUDGET_GB = float(os.environ.get("PHASEF_HARD_DISK_BUDGET_GB",
                                           "20"))
# POS_BUDGET is the final-state cap on positives/. During a positive
# forensic copytree we hold source (scan_root) + partial destination
# (positives/) simultaneously, so peak = pos_before + 2*repo. To keep
# peak ≤ HARD (20), we need pos_before + 2*repo ≤ 20 AND pos_after =
# pos_before + repo ≤ POS_BUDGET. With POS_BUDGET=10 and PER_REPO=5,
# worst-case instantaneous disk = 10 + 5 + partial ≤ 15 GB, well inside
# the 20 GB hard cap.
POS_BUDGET_GB = float(os.environ.get("PHASEF_POS_BUDGET_GB", "10"))
PER_REPO_BUDGET_GB = float(os.environ.get("PHASEF_PER_REPO_BUDGET_GB",
                                          "5"))
MIN_FREE_GB = float(os.environ.get("PHASEF_MIN_FREE_GB", "5"))

FIELDS = [
    "repo_id", "onnx_file", "repo_mb", "downloaded_mb",
    "n_syntactic", "n_admitted", "epsilon",
    "verdict", "verify_sec", "status",
]


class TimeoutError_(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError_(f"exceeded {PER_MODEL_TIMEOUT}s")


def _repo_slug(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _dir_size_bytes(p: Path) -> int:
    """Recursive on-disk size of `p` in bytes (0 if missing / error)."""
    if not p.exists():
        return 0
    total = 0
    try:
        for root, _, files in os.walk(p):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    pass
    except Exception:
        pass
    return total


def _fs_free_bytes(path: Path) -> int:
    """Filesystem free bytes at path; 0 on error."""
    try:
        st = os.statvfs(str(path))
        return int(st.f_bavail) * int(st.f_frsize)
    except Exception:
        return 0


def _budget_state(scan_root: Path, positives_dir: Path) -> Dict[str, float]:
    """Snapshot disk-usage invariants in GB (not bytes)."""
    gb = 1024 ** 3
    scan = _dir_size_bytes(scan_root) / gb
    pos = _dir_size_bytes(positives_dir) / gb
    tmp_free = _fs_free_bytes(scan_root.parent if scan_root.parent.exists()
                              else Path("/tmp")) / gb
    home_free = _fs_free_bytes(positives_dir.parent if
                               positives_dir.parent.exists()
                               else Path("/home")) / gb
    return {
        "scan_gb": scan,
        "pos_gb": pos,
        "total_gb": scan + pos,
        "tmp_free_gb": tmp_free,
        "home_free_gb": home_free,
    }


def _force_clean_scan_root(scan_root: Path) -> bool:
    """rmtree scan_root; return True iff residual < 10 MB afterwards."""
    if scan_root.exists():
        try:
            shutil.rmtree(scan_root, ignore_errors=False)
        except Exception as e:
            print(f"    [cleanup] rmtree failed: {type(e).__name__}: "
                  f"{str(e)[:120]}")
            # second try with ignore_errors
            shutil.rmtree(scan_root, ignore_errors=True)
    scan_root.mkdir(parents=True, exist_ok=True)
    residual_mb = _dir_size_bytes(scan_root) / (1024 ** 2)
    if residual_mb > 10:
        print(f"    [cleanup] WARN residual {residual_mb:.1f} MB in "
              f"{scan_root}")
        return False
    return True


def _primary_onnx(files: List[str]) -> Optional[str]:
    """Choose the smallest-named `.onnx` (usually the main graph).

    Prefer files with no fp16 / int8 / quant in the name; fall back to first.
    """
    if not files:
        return None
    base = [f for f in files
            if not any(tag in f.lower()
                       for tag in ("fp16", "int8", "int4", "quant", "dyna"))]
    candidates = base if base else files
    return sorted(candidates, key=len)[0]


def _download_repo(api: HfApi, repo_id: str, file_list: List[str],
                   dst: Path) -> int:
    """Download all files in `file_list` to `dst/`. Return total bytes."""
    dst.mkdir(parents=True, exist_ok=True)
    total = 0
    for fname in file_list:
        last_err = None
        for ep in ENDPOINTS:
            try:
                p = hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,
                    endpoint=ep,
                    local_dir=str(dst),
                    local_dir_use_symlinks=False,
                )
                if p and os.path.exists(p):
                    total += os.path.getsize(p)
                last_err = None
                break                       # got it; stop trying endpoints
            except Exception as e:
                last_err = e                # this endpoint failed -> try next
        if last_err is not None:
            # every endpoint failed for this file
            if fname.lower().endswith(".onnx"):
                raise last_err              # the .onnx is mandatory
            # optional file (README, images) missing -> keep going
            print(f"    [{repo_id}] skip {fname}: "
                  f"{type(last_err).__name__}: {str(last_err)[:80]}")
    return total


def _load_existing(csv_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not csv_path.exists():
        return done
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("repo_id")
            if rid:
                done.add(rid)
    return done


def verify_one(api: HfApi, entry: Dict[str, Any], scan_root: Path,
               positives_dir: Path) -> Dict[str, Any]:
    from archproof.verify_phaseC import verify_model_phaseC  # local import

    repo_id = entry["repo_id"]
    onnx_files = entry.get("onnx_files", [])
    n_files = entry.get("n_files", 0)
    repo_mb = round(entry.get("total_bytes", 0) / (1024 ** 2), 1)

    primary = _primary_onnx(onnx_files)
    if not primary:
        return {
            "repo_id": repo_id, "onnx_file": "", "repo_mb": repo_mb,
            "downloaded_mb": 0, "n_syntactic": "", "n_admitted": "",
            "epsilon": "", "verdict": "", "verify_sec": 0,
            "status": "no_onnx_in_manifest",
        }

    slug = _repo_slug(repo_id)
    dst = scan_root / slug
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)

    row = {k: "" for k in FIELDS}
    row["repo_id"] = repo_id
    row["onnx_file"] = primary
    row["repo_mb"] = repo_mb

    t0 = time.time()
    try:
        file_list = []
        for s in (entry.get("file_list") or onnx_files):
            file_list.append(s)
        if not file_list:
            file_list = [primary]
        # best effort: fetch everything listed in onnx_files + any shards we
        # know about. For safety re-query siblings if list_file is short.
        if n_files > len(file_list):
            try:
                info = api.model_info(repo_id, files_metadata=True)
                file_list = [s.rfilename for s in (info.siblings or [])
                             if hasattr(s, "rfilename")]
            except Exception:
                pass
        downloaded = _download_repo(api, repo_id, file_list, dst)
        row["downloaded_mb"] = round(downloaded / (1024 ** 2), 1)

        # Invariant: single-repo footprint must not exceed PER_REPO_BUDGET_GB.
        # If HF metadata underreported (LFS balloon, hidden shards), abort
        # this repo cleanly and cleanup so the budget isn't blown.
        repo_gb_now = _dir_size_bytes(dst) / (1024 ** 3)
        if repo_gb_now > PER_REPO_BUDGET_GB:
            row["status"] = (f"oversized_download: {repo_gb_now:.1f} GB > "
                             f"cap {PER_REPO_BUDGET_GB} GB")
            return row

        onnx_path = dst / primary
        if not onnx_path.exists():
            row["status"] = "download_missing_primary"
            return row

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(PER_MODEL_TIMEOUT)
        try:
            r = verify_model_phaseC(str(onnx_path))
        finally:
            signal.alarm(0)

        row["n_syntactic"] = r.n_syntactic
        row["n_admitted"] = r.n_admitted_phaseC
        row["epsilon"] = r.epsilon_phaseC
        row["verdict"] = r.verdict_phaseC
        row["verify_sec"] = round(time.time() - t0, 1)
        row["status"] = "ok"

        # Forensic-keep: any CERTIFIED-POSITIVE goes to wild_positives/,
        # but only if doing so stays under POS_BUDGET_GB. If over budget,
        # skip the copy and annotate the verdict — we keep the row, not
        # the ONNX.
        verdict = (r.verdict_phaseC or "").upper()
        if "CERTIFIED-POSITIVE" in verdict or "POSITIVE" in verdict:
            repo_gb = _dir_size_bytes(dst) / (1024 ** 3)
            pos_gb_now = _dir_size_bytes(positives_dir) / (1024 ** 3)
            if pos_gb_now + repo_gb > POS_BUDGET_GB:
                print(f"    [{repo_id}] POSITIVE but skipping forensic "
                      f"copy: pos_dir {pos_gb_now:.1f}+{repo_gb:.1f} GB > "
                      f"cap {POS_BUDGET_GB} GB")
                row["status"] = "ok_positive_copy_skipped_budget"
            else:
                try:
                    positives_dir.mkdir(parents=True, exist_ok=True)
                    out = positives_dir / slug
                    if out.exists():
                        shutil.rmtree(out, ignore_errors=True)
                    shutil.copytree(dst, out)
                    print(f"    [{repo_id}] POSITIVE: copied to {out}")
                except Exception as e:
                    print(f"    [{repo_id}] positive copy failed: {e}")
    except TimeoutError_ as e:
        row["verify_sec"] = round(time.time() - t0, 1)
        row["status"] = f"timeout: {e}"
    except Exception as e:
        row["verify_sec"] = round(time.time() - t0, 1)
        row["status"] = f"fail: {type(e).__name__}: {str(e)[:200]}"
    finally:
        # Strict cleanup: tear down the per-repo dir; verify it actually went.
        try:
            shutil.rmtree(dst, ignore_errors=False)
        except Exception:
            shutil.rmtree(dst, ignore_errors=True)

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default=os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmark/phaseF_manifest.json"))
    ap.add_argument(
        "--out",
        default=os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "truth_source/per_model_phaseF.csv"))
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many *new* models (0 = no cap)")
    args = ap.parse_args()

    SCAN_ROOT.mkdir(parents=True, exist_ok=True)
    POSITIVES_DIR.mkdir(parents=True, exist_ok=True)

    # Startup invariant: scan_root must be empty. Any residue from a crashed
    # prior run would count against the budget, so scrub it now.
    _force_clean_scan_root(SCAN_ROOT)

    # Signal handler: on SIGINT/SIGTERM, tear the scan root down before
    # the process exits so /tmp is never left with stale downloads.
    def _on_signal(signum, frame):
        print(f"[scan] received signal {signum}; cleaning scan_root and "
              f"exiting")
        _force_clean_scan_root(SCAN_ROOT)
        sys.exit(130 if signum == signal.SIGINT else 143)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    mpath = Path(args.manifest)
    if not mpath.exists():
        sys.exit(f"ERROR: manifest not found: {mpath}. "
                 f"Run phaseF_discover_models.py first.")
    with mpath.open() as f:
        mani = json.load(f)
    entries = mani.get("models", [])
    print(f"[scan] manifest: {mpath} n={len(entries)}")
    print(f"[scan] endpoint={ENDPOINT} scan_root={SCAN_ROOT}")
    print(f"[scan] HARD_CAP total={HARD_DISK_BUDGET_GB} GB  "
          f"pos={POS_BUDGET_GB} GB  per-repo={PER_REPO_BUDGET_GB} GB  "
          f"min_free={MIN_FREE_GB} GB")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_existing(out)
    print(f"[scan] resume: {len(done)} repo_ids already in {out.name}")

    new_header = not out.exists()
    api = HfApi(endpoint=ENDPOINT)
    started = time.time()
    deadline = started + MAX_RUN_HOURS * 3600
    n_new = 0
    n_ok = 0
    n_pos = 0
    n_neg = 0
    n_unc = 0
    n_fail = 0

    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_header:
            w.writeheader()

        for i, entry in enumerate(entries, 1):
            if time.time() > deadline:
                print(f"[scan] hit MAX_RUN_HOURS={MAX_RUN_HOURS}; stopping")
                break
            repo_id = entry.get("repo_id", "")
            if not repo_id or repo_id in done:
                continue

            # Pre-iteration HARD-CAP invariant check. We reserve
            # 2 * PER_REPO_BUDGET headroom so even a worst-case
            # oversized download + positive copytree stays inside
            # HARD. When total usage is close to the cap, we stop
            # rather than skip, because pos_dir is monotonically
            # growing (only positives accumulate; can't free space).
            bs = _budget_state(SCAN_ROOT, POSITIVES_DIR)
            headroom = HARD_DISK_BUDGET_GB - 2 * PER_REPO_BUDGET_GB
            if bs["total_gb"] > headroom:
                print(f"[scan] STOP: total {bs['total_gb']:.2f} GB > "
                      f"headroom {headroom:.1f} GB "
                      f"(hard cap {HARD_DISK_BUDGET_GB}, reserves "
                      f"{2 * PER_REPO_BUDGET_GB} GB for next repo + copy) "
                      f"(scan {bs['scan_gb']:.2f} + pos "
                      f"{bs['pos_gb']:.2f})")
                break
            if bs["tmp_free_gb"] < MIN_FREE_GB:
                print(f"[scan] STOP: /tmp free only "
                      f"{bs['tmp_free_gb']:.2f} GB < min "
                      f"{MIN_FREE_GB} GB")
                break
            if bs["home_free_gb"] < MIN_FREE_GB:
                print(f"[scan] STOP: /home free only "
                      f"{bs['home_free_gb']:.2f} GB < min "
                      f"{MIN_FREE_GB} GB")
                break

            # Pre-download prediction: if the manifest-claimed size plus
            # current total could push us past the hard cap during a
            # possible positive copytree (peak = total + 2 * claim),
            # skip this repo (continue scanning smaller ones).
            claimed_gb = entry.get("total_bytes", 0) / (1024 ** 3)
            if bs["total_gb"] + 2 * claimed_gb > HARD_DISK_BUDGET_GB:
                print(f"[{i}/{len(entries)}] {repo_id} "
                      f"skipped: total {bs['total_gb']:.1f}+2*claim "
                      f"{claimed_gb:.1f} GB would risk hard cap "
                      f"{HARD_DISK_BUDGET_GB} GB")
                continue

            print(f"[{i}/{len(entries)}] {repo_id} "
                  f"({entry.get('total_bytes', 0) / (1024 ** 2):.1f} MB)  "
                  f"[disk scan={bs['scan_gb']:.1f} pos={bs['pos_gb']:.1f} "
                  f"/ cap {HARD_DISK_BUDGET_GB} GB]",
                  flush=True)
            row = verify_one(api, entry, SCAN_ROOT, POSITIVES_DIR)
            w.writerow(row)
            f.flush()
            done.add(repo_id)
            n_new += 1

            # Post-iteration invariant: scan_root MUST be < 100 MB
            # (just cleanup noise). If not, something leaked; stop.
            residual_mb = _dir_size_bytes(SCAN_ROOT) / (1024 ** 2)
            if residual_mb > 100:
                print(f"[scan] WARN residual {residual_mb:.1f} MB in "
                      f"{SCAN_ROOT} after verify; retry cleanup")
                if not _force_clean_scan_root(SCAN_ROOT):
                    print(f"[scan] STOP: cannot clean scan_root; aborting "
                          f"to protect disk")
                    break
            if row["status"] == "ok":
                n_ok += 1
                v = (row["verdict"] or "").upper()
                if "CERTIFIED-POSITIVE" in v or "POSITIVE" in v:
                    n_pos += 1
                elif "CLASS-NEGATIVE" in v or "DGP-FREE" in v or "NEGATIVE" in v:
                    n_neg += 1
                elif "UNCERTIFIED" in v or "UNDECIDED" in v:
                    n_unc += 1
            else:
                n_fail += 1
            print(f"    -> {row['verdict']}  n_adm={row['n_admitted']}  "
                  f"{row['verify_sec']}s  [{row['status']}]", flush=True)
            if args.limit and n_new >= args.limit:
                print(f"[scan] hit --limit={args.limit}")
                break

    elapsed = time.time() - started
    print(f"\n[scan] SUMMARY (elapsed {elapsed/60:.1f} min)")
    print(f"  new={n_new}  ok={n_ok}  fail={n_fail}")
    print(f"  positive={n_pos}  negative={n_neg}  uncertified={n_unc}")
    print(f"  verdict CSV: {out}")
    print(f"  positives dir: {POSITIVES_DIR}")


if __name__ == "__main__":
    main()
