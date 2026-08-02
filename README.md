# barna-utils

General-purpose infrastructure helpers shared across every project in `Barna/Projects/` —
LSF job submission/monitoring, FlashBlade NVMe staging, and similar cluster/environment
plumbing that has no spatial-biology content of its own. If a piece of logic would make sense
in a completely unrelated project, it belongs here; if it's inherently about spatial
transcriptomics analysis, it belongs in `IDH-Spatial_Transcriptomics/common/` instead.

## Why this exists

Before 2026-08-02, LSF submission and FlashBlade staging code was independently duplicated at
least five times across this project's repos: a standalone `lsf_utils.py` module in MERFISH,
separate inline copies in MERFISH notebooks 01/02/04/05 (some already diverged from the
module, one never migrated to it at all), and a further separate inline implementation in
IHC's stitching notebook. Every copy had the same basic shape with small, accidental
differences in which `bsub` resource flags were supported. This package is the single
consensus implementation each project now imports instead of vendoring its own copy — a fix
here lands everywhere at once.

## Interruption safety (VS Code / Jupyter kernel restarts, connection drops)

Submitted jobs are safe from notebook interruption by construction: `bsub` hands the job off
to the LSF scheduler as its own independent process tree the instant it's submitted, with no
parent-child relationship to the Python kernel that called it — killing/restarting the kernel,
or losing the connection, cannot touch a job that's already running on the cluster.

The one thing that needed fixing (found and fixed 2026-08-02, confirmed against a real
submitted job): re-running a `submit_and_wait()`/`submit_multi_and_wait()` cell after such an
interruption used to only check "do the final output files already exist?" — if the job was
still mid-run (not finished yet), that check doesn't help, and the old behavior would submit a
**second, duplicate** job. Both functions now check `bjobs -J <job_name>` for an already
PEND/RUN job with the same name before submitting anything, and resume monitoring that job
instead. Verified end-to-end: submitted a real test job manually, then called
`submit_and_wait()` for the same job name while it was still running — it found and resumed
monitoring the existing job (never re-submitting), correctly tracked real elapsed time, and
correctly reported completion when the actual job finished.

## Modules

- **`barna_utils.lsf`** — `submit_and_wait()` (single job, poll until done),
  `submit_multi_and_wait()` (several independent jobs, one combined status display), and
  `submit()` (fire-and-forget: submit and return the job ID immediately, no polling -- for the
  "launch N jobs now, check back on them in a separate cell later" pattern, e.g. a parameter
  sweep with many candidates, where blocking the notebook isn't the point). All three support
  every resource flag found across the five prior implementations: `queue`, `cpus`,
  `memory_mb`, `span_hosts`, `gpu_flag`, `gpu_model_select`. Deliberately never sets email
  flags or a walltime limit (this project's queues don't need one, and letting LSF apply its
  own default avoids `RUNLIMIT` submission failures).
- **`barna_utils.staging`** — `stage_to_flashblade()` / `cleanup_flashblade()`. rsyncs a file
  or directory to fast NVMe scratch before a job reads it, idempotent via a `.stage_complete`
  marker, cleaned up only on success so a failed job can retry without re-paying the copy cost.
- **`barna_utils.progress`** — the live-display layer `lsf.py` uses internally. Renders a real
  `ipywidgets` progress bar + status line in Jupyter/VS Code, falling back to plain
  `print` + `clear_output` everywhere else (headless scripts, a terminal Python REPL, or a
  Jupyter frontend without `ipywidgets` installed).

### A note on "progress" — what actually works vs. what looked fancier

IHC's original inline implementation used `ipywidgets` plus a regex parser for `tqdm`-style
progress lines written to a file by the running job. It looked the most sophisticated of all
five prior versions, but **never actually worked**: nothing wrote real progress into that file,
so the bar always jumped straight from 0% to "finished" with no gradual movement in between.

This package's progress bar only ever shows things that are genuinely, verifiably true at each
poll tick: real LSF job status (`bjobs`, not assumed), a real elapsed-time clock, and — for
jobs with multiple expected output files — a real count of how many exist yet. For a
single-output job, pass a `progress_fn` callback to `submit_and_wait()` that checks real,
job-specific state (the same pattern MERFISH's NB02/NB04 already use for per-region progress)
rather than relying on an assumed log format.

## Installation

Installed as an editable pip package into whichever environment needs it, not vendored:

```bash
pip install -e /omics/odcf/analysis/OE0146_projects/idh_astro/Barna/Projects/barna-utils
# for the ipywidgets-based progress display specifically:
pip install -e "/omics/odcf/analysis/OE0146_projects/idh_astro/Barna/Projects/barna-utils[widgets]"
```

## Usage

```python
from barna_utils import submit_and_wait, stage_to_flashblade, cleanup_flashblade

LSF_PROFILE = {
    'train': dict(queue='gpu', gpu_flag='num=1:gmem=16000'),
    'filter': dict(queue='verylong', cpus=8, memory_mb=64000),
}

ok = submit_and_wait(
    job_name='my_job', lsf_profile=LSF_PROFILE, stage='filter',
    cmd_args=[PYTHON_BIN, str(script_path)],
    success_paths=[output_path], logs_dir=LOGS_DIR,
)
```

## Status

Consolidated from MERFISH + IHC's five prior implementations on 2026-08-02. Being adopted by
each consuming project in place of its own local copy — see each project's own README for
whether that migration is done yet.
