"""
Generalized LSF (bsub/bjobs) job submission and monitoring.

This consolidates what were, before 2026-08-02, at least five separately
maintained near-duplicate implementations across this project: a standalone
`lsf_utils.py` module in the MERFISH pipeline, inline copies in MERFISH
notebooks 01/02/04/05 (some diverged from the module, some never migrated to
it at all), and a further separate inline version in IHC's stitching notebook.
Every one of those had the same basic shape (submit one `bsub` job, poll
`bjobs` until it finishes, report whether the expected outputs now exist) with
small, accidental differences in which resource flags were supported. This
module is the single consensus implementation -- the union of every flag/
feature actually found in use anywhere, not just whichever version happened to
be copied last.

Deliberately excluded: a percentage-based progress bar driven by parsing a
job's own log output. An earlier attempt at this (IHC's tqdm-regex parser) was
confirmed never to have worked in practice -- see `progress.py`'s module
docstring. Real, verified progress information (per-output-file completion,
or a caller-supplied status callback) is supported instead.
"""
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from .progress import make_display


def _build_bsub_args(job_name, stage_cfg, log_out, log_err, project, extra_bsub_args):
    """Build the bsub argv from a stage config dict. Recognized keys (all
    optional except `queue`): `cpus`, `memory_mb`, `span_hosts`, `gpu_flag`,
    `gpu_model_select`, `env` (dict of extra environment variables to export
    before the command, e.g. LD_LIBRARY_PATH -- IHC's stitching jobs need
    this; most jobs don't).

    Deliberately never sets email (-B/-N) or a walltime (-W) limit -- this
    project's queues don't need one, and letting LSF apply its own default
    avoids RUNLIMIT submission failures (see MERFISH lsf_utils.py's original
    docstring for the incident this was learned from).
    """
    args = ['bsub', '-P', project, '-q', stage_cfg['queue'], '-J', job_name,
            '-o', str(log_out), '-e', str(log_err)]

    # GPU stages conventionally omit -n/-R entirely: CPU/RAM get assigned
    # automatically alongside the requested GPU. Only set cpus/memory_mb
    # for a GPU stage if there's a specific, documented reason (e.g. a
    # stage that needs real system RAM beyond what comes bundled with the
    # GPU allocation).
    if 'cpus' in stage_cfg:
        args += ['-n', str(stage_cfg['cpus'])]
    if 'memory_mb' in stage_cfg:
        r_spec = f"rusage[mem={stage_cfg['memory_mb']}]"
        if stage_cfg.get('span_hosts'):
            r_spec += f" span[hosts={stage_cfg['span_hosts']}]"
        args += ['-R', r_spec]
    if stage_cfg.get('gpu_flag'):
        args += ['-gpu', stage_cfg['gpu_flag']]
    if stage_cfg.get('gpu_model_select'):
        args += ['-R', f"select[gpumodel=={stage_cfg['gpu_model_select']}]"]
    if extra_bsub_args:
        args += extra_bsub_args
    # Prepend env-var exports to the command so the job inherits them.
    # PYTHONUNBUFFERED=1 is always set so Python flushes stdout immediately
    # (critical for log tailing on NFS where line-buffered writes can stall).
    env_prefix = ['env', 'PYTHONUNBUFFERED=1']
    if stage_cfg.get('env'):
        env_prefix += [f'{k}={v}' for k, v in stage_cfg['env'].items()]
    args += env_prefix
    return args


def _job_status(job_id):
    r = subprocess.run(['bjobs', '-noheader', str(job_id)], capture_output=True, text=True)
    return r.stdout.strip().split()[2] if r.stdout.strip() else None


def _find_running_job(job_name):
    """Return the job ID of an already PEND/RUN job with this exact name, if
    one exists -- used so that re-running a `submit_and_wait()` cell after a
    notebook interruption (kernel restart, connection drop, accidental
    interrupt) resumes monitoring the job that's actually still running on
    the cluster instead of submitting a duplicate. The job itself is never at
    risk from a notebook interruption (`bsub` hands it off to the LSF
    scheduler as an independent process tree the moment it's submitted) --
    this check is purely about not wasting cluster resources on a redundant
    second submission when the *monitoring* side gets interrupted and resumed.
    Returns None if no matching job is found (job names aren't required to be
    unique in LSF, so this returns the first match).
    """
    r = subprocess.run(['bjobs', '-J', job_name, '-noheader'], capture_output=True, text=True)
    line = r.stdout.strip()
    if not line or 'not found' in line:
        return None
    fields = line.split()
    if len(fields) >= 3 and fields[2] in ('PEND', 'RUN'):
        return fields[0]
    return None


def _job_status_final(job_id):
    r = subprocess.run(['bjobs', '-noheader', '-d', str(job_id)], capture_output=True, text=True)
    return r.stdout.strip().split()[2] if r.stdout.strip() else 'DONE'


def submit_and_wait(job_name, lsf_profile, stage, cmd_args, success_paths, logs_dir,
                     project='OE0146', extra_bsub_args=None, poll_interval=30,
                     force=False, progress_fn=None):
    """Submit `cmd_args` as an LSF job using resources from
    `lsf_profile[stage]`, then poll until it finishes.

    Returns True if every path in `success_paths` exists once the job is done.
    Safe to re-run: if they all already exist, submission is skipped entirely.

    `progress_fn`, if given, is called as `progress_fn()` on every poll tick
    and should return a list of strings to display underneath the status line
    -- the mechanism NB02/NB04 used for real per-region/per-file progress
    (e.g. "region_R5: OK", "region_R6: NOT WRITTEN"), which is genuinely
    reliable since it checks real state rather than parsing a log.
    """
    success_paths = [Path(p) for p in success_paths]
    if not force and all(p.exists() for p in success_paths):
        print(f'[{job_name}] All outputs already exist -- skipping submission.')
        for p in success_paths:
            print(f'    {p}')
        return True

    logs_dir = Path(logs_dir)

    existing = _find_running_job(job_name)
    if existing:
        print(f'[{job_name}] Found already-running job {existing} -- resuming monitoring '
              f'(not submitting a duplicate; this is the expected path after a notebook '
              f'interruption while the job was still in flight).', flush=True)
        job_id = existing
    else:
        stage_cfg = lsf_profile[stage]
        log_out = logs_dir / f'{job_name}.%J.out'
        log_err = logs_dir / f'{job_name}.%J.err'

        bsub_args = _build_bsub_args(job_name, stage_cfg, log_out, log_err, project, extra_bsub_args)
        bsub_args += cmd_args

        print(f'[{job_name}] Submitting ...', flush=True)
        res = subprocess.run(bsub_args, capture_output=True, text=True, timeout=30)
        m = re.search(r'Job <(\d+)>', res.stdout)
        if not m:
            print('Submission failed:', res.stdout or res.stderr)
            return False
        job_id = m.group(1)

    disp = make_display(f'Job {job_id} -- {job_name}<br>Log: {logs_dir}/{job_name}.{job_id}.out')
    start = time.monotonic()

    while True:
        st = _job_status(job_id)
        done_now = sum(p.exists() for p in success_paths)
        extra = list(progress_fn()) if progress_fn else [
            f'{"EXISTS" if p.exists() else "pending"}  {p}' for p in success_paths
        ]
        if st is None or st in ('DONE', 'EXIT'):
            final = _job_status_final(job_id)
            all_ok = all(p.exists() for p in success_paths)
            disp.update(f'Finished -- {final}', time.monotonic() - start,
                        done_now, len(success_paths), extra)
            disp.finalize(all_ok)
            print(f'\n[{job_name}] Job {job_id} finished -- {final}', flush=True)
            for line in extra:
                print(f'    {line}', flush=True)
            if not all_ok:
                err_file = logs_dir / f'{job_name}.{job_id}.err'
                if err_file.exists():
                    print('\n--- tail of stderr ---', flush=True)
                    print('\n'.join(err_file.read_text().splitlines()[-30:]), flush=True)
            return all_ok
        disp.update(st, time.monotonic() - start, done_now, len(success_paths), extra)
        time.sleep(poll_interval)


def submit_multi_and_wait(jobs, lsf_profile, logs_dir, project='OE0146', poll_interval=60):
    """Submit several independent jobs at once and poll all of them together
    in one combined status display, rather than submitting-and-fully-waiting
    one at a time. `jobs` is `{label: dict(job_name=..., stage=..., cmd_args=...,
    success_paths=..., extra_bsub_args=...)}`. Returns `{label: bool}`.
    """
    logs_dir = Path(logs_dir)
    submitted = {}
    for lbl, spec in jobs.items():
        success_paths = [Path(p) for p in spec['success_paths']]
        if all(p.exists() for p in success_paths):
            print(f'[{lbl}] All outputs already exist -- skipping submission.')
            continue

        job_name = spec['job_name']
        existing = _find_running_job(job_name)
        if existing:
            print(f'[{lbl}] Found already-running job {existing} -- resuming monitoring '
                  f'(not submitting a duplicate).', flush=True)
            submitted[lbl] = dict(job_id=existing, job_name=job_name, success_paths=success_paths)
            continue

        stage_cfg = lsf_profile[spec['stage']]
        log_out = logs_dir / f'{job_name}.%J.out'
        log_err = logs_dir / f'{job_name}.%J.err'
        bsub_args = _build_bsub_args(job_name, stage_cfg, log_out, log_err, project,
                                      spec.get('extra_bsub_args'))
        bsub_args += spec['cmd_args']

        res = subprocess.run(bsub_args, capture_output=True, text=True, timeout=30)
        m = re.search(r'Job <(\d+)>', res.stdout)
        if not m:
            print(f'[{lbl}] Submission failed:', res.stdout or res.stderr)
            continue
        submitted[lbl] = dict(job_id=m.group(1), job_name=job_name, success_paths=success_paths)
        print(f'[{lbl}] Job {submitted[lbl]["job_id"]} submitted -- {job_name}')

    if not submitted:
        return {lbl: all(Path(p).exists() for p in spec['success_paths'])
                for lbl, spec in jobs.items()}

    disp = make_display(f'Monitoring {len(submitted)} job(s)')
    start = time.monotonic()

    while True:
        lines = []
        all_done = True
        done_count = 0
        for lbl, info in submitted.items():
            st = _job_status(info['job_id'])
            done = st is None or st in ('DONE', 'EXIT')
            all_done &= done
            done_count += done
            n_ok = sum(p.exists() for p in info['success_paths'])
            status_str = 'FINISHED' if done else f'{st}'
            lines.append(f'{lbl:<10} job {info["job_id"]:<10} {status_str:<10} '
                        f'{n_ok}/{len(info["success_paths"])} outputs present')
        disp.update(f'{done_count}/{len(submitted)} jobs finished', time.monotonic() - start,
                    done_count, len(submitted), lines)
        if all_done:
            break
        time.sleep(poll_interval)

    results = {}
    for lbl, spec in jobs.items():
        success_paths = [Path(p) for p in spec['success_paths']]
        results[lbl] = all(p.exists() for p in success_paths)
    disp.finalize(all(results.values()))
    print(f'\n=== Final status ===')
    for lbl, ok in results.items():
        print(f'  {lbl}: {"OK" if ok else "MISSING -- check log"}')
    return results
