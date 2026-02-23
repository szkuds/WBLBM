"""Generate sbatch run scripts grouping generated variant Python files into batches of 10.

Reads the manifest `DB_run/26_02_14_DB_run_phase_plots/generated_variants_manifest.txt` and
creates run scripts `11-20_run.sh`, `21-30_run.sh`, ..., `131-140_run.sh` in the same directory
as the manifest. Each generated script mirrors the headers from `01-10_run.sh`.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "DB_run" / "26_02_14_DB_run_phase_plots"
MANIFEST = OUT_DIR / "generated_variants_manifest.txt"
TEMPLATE = OUT_DIR / "01-10_run.sh"

# Read manifest
lines = [ln.strip() for ln in MANIFEST.read_text().splitlines() if ln.strip()]
# Convert absolute paths to basenames
basenames = [Path(p).name for p in lines]

# Read template header to reuse SBATCH headers
template_header = TEMPLATE.read_text().splitlines()
# Find index where commands start (first srun occurrence)
cmd_start = 0
for i, ln in enumerate(template_header):
    if ln.strip().startswith('srun'):
        cmd_start = i
        break
header_lines = template_header[:cmd_start]

# Create batches of 10, skipping the first batch (01-10) as it already exists
for batch_start in range(11, 141, 10):
    batch_end = batch_start + 9
    # compute zero-padded indices into basenames list: filenames were created starting at 1
    # example: batch_start=11 corresponds to basenames[10]..basenames[19]
    start_idx = batch_start - 1
    end_idx = batch_end - 1
    batch_files = basenames[start_idx:end_idx+1]
    if not batch_files:
        continue

    job_name = f"{batch_start:02d}-{batch_end:02d}"
    out_name = OUT_DIR / f"{batch_start:02d}-{batch_end:02d}_run.sh"

    with out_name.open('w') as f:
        # write header, but update job-name line
        for ln in header_lines:
            if ln.strip().startswith('#SBATCH --job-name='):
                f.write(f"#SBATCH --job-name=\"{job_name}\"\n")
            else:
                f.write(ln + "\n")
        f.write('\n')
        # write srun lines
        for fname in batch_files:
            f.write(f"srun python ./{fname} > ./{fname.replace('.py', '.log')}\n")

    # make executable
    out_name.chmod(0o755)
    print(f"Wrote {out_name}")

print("All batch run scripts created.")

