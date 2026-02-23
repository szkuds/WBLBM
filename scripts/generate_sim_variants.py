"""Generate simulation variant scripts from 00_base_sim.py template.

This script creates:
 - 5 inclination-only variants (01..05)
 - 15 CA-pair-only variants (06..20)
 - All combinations of the 15 CA pairs with inclinations 10..80 step 10 (120 files), continuing numbering from 21..

Generated files are written next to the template in the same directory.
"""
import re
import os
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "DB_run" / "26_02_14_DB_run_phase_plots" / "00_base_sim.py"
OUT_DIR = TEMPLATE.parent

# Parameters
inclination_only = [10, 20, 30, 50, 70]
ca_pairs = [
    (110, 90), (100, 90), (130, 90), (140, 90), (130, 100),
    (140, 100), (120, 110), (130, 110), (140, 110), (130, 120),
    (140, 120), (140, 130), (108, 102), (115, 95), (125, 85),
]
angles_full = list(range(10, 81, 10))  # 10..80 step 10

# Read template
template_text = TEMPLATE.read_text()

# Helper to replace a top-level assignment like `NAME = ...` using line-by-line safety
def replace_assignment(text, name, value):
    lines = text.splitlines()
    name_eq = f"{name}"
    replaced = False
    for i, ln in enumerate(lines):
        # strip leading/trailing spaces for robust match but preserve indentation level
        stripped = ln.lstrip()
        if stripped.startswith(name_eq) and '=' in stripped:
            indent = ln[: len(ln) - len(stripped)]
            lines[i] = f"{indent}{name} = {value}"
            replaced = True
            break
    if not replaced:
        raise RuntimeError(f"Assignment {name} not found in template")
    return "\n".join(lines) + "\n"

# Helper to rename function and its call and pipeline tag using direct string replacements
def rename_chem_function(text, new_name):
    # Replace the function definition header
    if 'def run_chem_step_base(' not in text:
        raise RuntimeError('run_chem_step_base definition not found')
    text = text.replace('def run_chem_step_base(', f'def {new_name}(')

    # Replace the call in main and the call site that sets chem_dir
    text = text.replace('chem_dir = run_chem_step_base(', f'chem_dir = {new_name}(')

    # Replace the pipeline tag string used in move_results_to_pipeline inside the function
    # Original uses "run_chem_step" as the tag; change any occurrence inside the file to the new name
    text = text.replace('"run_chem_step"', f'"{new_name}"')

    return text

# Create variants list (tuples of filename, generated_text)
variants = []
index = 1

# Group 1: inclination-only (01..05)
for angle in inclination_only:
    idx = f"{index:02d}"
    name = f"{idx}_inc{angle}.py"
    func_name = f"run_chem_step_inc{angle}"
    t = template_text
    t = replace_assignment(t, "INCLINATION_ANGLE", f"{angle}")
    t = rename_chem_function(t, func_name)
    variants.append((name, t))
    index += 1

# Group 2: CA pair-only (06..20)
for adv, rec in ca_pairs:
    idx = f"{index:02d}"
    name = f"{idx}_pre_cah_{adv}_{rec}.py"
    func_name = f"run_chem_step_pre_cah_{adv}_{rec}"
    t = template_text
    t = replace_assignment(t, "CA_ADVANCING_PRE", f"{float(adv)}")
    t = replace_assignment(t, "CA_RECEDING_PRE", f"{float(rec)}")
    t = rename_chem_function(t, func_name)
    variants.append((name, t))
    index += 1

# Group 3: combinations of all CA pairs with angles 10..80 (120 files)
for adv, rec in ca_pairs:
    for angle in angles_full:
        idx = f"{index:02d}" if index < 100 else f"{index:03d}"
        name = f"{idx}_inc{angle}_pre{adv}_{rec}.py"
        func_name = f"run_chem_step_inc{angle}_pre{adv}_{rec}"
        t = template_text
        t = replace_assignment(t, "INCLINATION_ANGLE", f"{angle}")
        t = replace_assignment(t, "CA_ADVANCING_PRE", f"{float(adv)}")
        t = replace_assignment(t, "CA_RECEDING_PRE", f"{float(rec)}")
        t = rename_chem_function(t, func_name)
        variants.append((name, t))
        index += 1

# Write files
written = []
for fname, content in variants:
    out_path = OUT_DIR / fname
    if out_path.exists():
        print(f"Overwriting {out_path}")
    out_path.write_text(content)
    written.append(str(out_path))

print(f"Wrote {len(written)} files to {OUT_DIR}")

# Print a short summary of the first few generated files
for p in written[:10]:
    print(" -", p)

# Save a manifest
manifest = OUT_DIR / "generated_variants_manifest.txt"
manifest.write_text("\n".join(written))
print("Manifest saved to", manifest)

