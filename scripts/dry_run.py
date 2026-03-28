"""
dry_run.py — Répétition générale chronométrée
Mesure le temps de chaque étape du pipeline de soumission.
"""
import sys, os, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

# Force UTF-8 sur Windows
ENV = dict(os.environ)
ENV["PYTHONIOENCODING"] = "utf-8"
ENV["PYTHONUTF8"]       = "1"

TRAIN = "data/_dryrun_train.csv"
TEST  = "data/_dryrun_test.csv"
OUT   = "artifacts/dryrun"

def run_step(name, cmd):
    t0  = time.time()
    r   = subprocess.run([sys.executable] + cmd, capture_output=True, text=True, env=ENV, encoding="utf-8", errors="replace")
    sec = round(time.time() - t0, 2)
    ok  = r.returncode == 0
    err = r.stderr[-300:] if not ok else ""
    print(f"  {'OK' if ok else 'FAIL':4s}  {sec:6.1f}s  {name}")
    if not ok:
        print(f"       ERROR: {err.strip()}")
    return {"step": name, "seconds": sec, "ok": ok, "error": err[:200] if not ok else ""}

steps = []
t_start = time.time()
os.makedirs(OUT, exist_ok=True)

print(f"\n{'─'*60}")
print(f"  RÉPÉTITION GÉNÉRALE — {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"{'─'*60}")
print(f"  Train : {TRAIN}")
print(f"  Test  : {TEST}\n")

# ── Step 1 : inspect_dataset ──────────────────────────────────
steps.append(run_step("inspect_dataset", [
    "scripts/inspect_dataset.py", TRAIN
]))

# ── Step 2 : run_cutdown balanced (mode urgence) ─────────────
steps.append(run_step("run_cutdown [balanced, 3-fold]", [
    "scripts/run_cutdown.py",
    "--train", TRAIN, "--test", TEST,
    "--profile", "balanced",
    "--cv-folds", "3",
    "--out", OUT,
]))

# ── Step 3 : run_cutdown conservative ────────────────────────
steps.append(run_step("run_cutdown [conservative, 3-fold]", [
    "scripts/run_cutdown.py",
    "--train", TRAIN, "--test", TEST,
    "--profile", "conservative",
    "--cv-folds", "3",
    "--out", OUT,
]))

# ── Step 4 : submission_factory (3 profils) ───────────────────
steps.append(run_step("submission_factory [3 profils, 3-fold]", [
    "scripts/submission_factory.py",
    "--train", TRAIN, "--test", TEST,
    "--cv-folds", "3",
    "--out", OUT,
]))

# ── Vérification des fichiers générés ────────────────────────
expected_files = [
    os.path.join(OUT, "cutdown_balanced.csv"),
    os.path.join(OUT, "cutdown_conservative.csv"),
    os.path.join(OUT, "submission_conservative.csv"),
    os.path.join(OUT, "submission_balanced.csv"),
    os.path.join(OUT, "submission_aggressive.csv"),
    os.path.join(OUT, "factory_report.json"),
]
file_checks = []
for f in expected_files:
    exists = os.path.exists(f)
    size   = os.path.getsize(f) if exists else 0
    file_checks.append({"file": os.path.basename(f), "exists": exists, "bytes": size})
    print(f"  {'✅' if exists else '❌'}  {os.path.basename(f):45s}  {size:6d} bytes")

# ── Rapport final ─────────────────────────────────────────────
total = time.time() - t_start
n_ok  = sum(1 for s in steps if s["ok"])
n_kof = len(steps) - n_ok

print(f"\n{'─'*60}")
print(f"  RÉSUMÉ")
print(f"{'─'*60}")
print(f"  Étapes réussies : {n_ok}/{len(steps)}")
print(f"  Temps total     : {total:.1f}s ({total/60:.1f} min)")
print(f"  {'─'*30}")
for s in steps:
    status = "✅" if s["ok"] else "❌"
    print(f"  {status}  {s['step']:40s}  {s['seconds']:5.1f}s")
print()

# ── Points de friction ────────────────────────────────────────
friction = []
for s in steps:
    if not s["ok"]:
        friction.append(f"ERREUR - {s['step']}: {s['error']}")
    elif s["seconds"] > 60:
        friction.append(f"LENT - {s['step']} ({s['seconds']:.0f}s > 60s attendu)")

if friction:
    print(f"  ⚠️  POINTS DE FRICTION ({len(friction)}) :")
    for f in friction:
        print(f"  • {f}")
else:
    print("  ✅  Aucun point de friction — pipeline fluide !")

# ── Export JSON ───────────────────────────────────────────────
report = {
    "generated_at":     datetime.now().isoformat(),
    "train":            TRAIN,
    "test":             TEST,
    "elapsed_seconds":  round(total, 1),
    "elapsed_minutes":  round(total/60, 2),
    "steps":            steps,
    "files_generated":  file_checks,
    "friction_points":  friction,
    "n_steps_ok":       n_ok,
    "n_steps_fail":     n_kof,
}
report_path = os.path.join(OUT, "dry_run_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
print(f"\n  Rapport JSON : {report_path}")
print(f"{'─'*60}\n")
