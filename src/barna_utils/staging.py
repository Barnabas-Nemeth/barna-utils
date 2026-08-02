"""
FlashBlade NVMe staging: rsync an input (file or directory) to fast scratch
storage before an LSF job reads it, instead of reading directly over NFS.

Same pattern used (with small accidental variations) in MERFISH's
`lsf_utils.py`, MERFISH NB01's inline copy, and IHC's `01_IHC_stitching.ipynb`
Cell 1.5 -- consolidated here as the one implementation. Kept MERFISH's
file-or-directory dual handling (IHC's version only ever handled directories,
but NB05's ResolVI stage needs to stage a single AnnData checkpoint file, not
a whole directory) since it's a strict superset with no downside for the
directory-only callers.
"""
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def stage_to_flashblade(src_path, dst_dir, label):
    """rsync `src_path` (a file or a directory) to `dst_dir` on fast scratch
    storage. Idempotent: if a previous run left a `.stage_complete` marker
    behind (e.g. because the LSF job failed after staging finished), the
    existing copy is reused instead of re-copied.

    A directory's contents land directly under `dst_dir`; a single file lands
    at `dst_dir / src_path.name`.
    """
    src_path = Path(src_path)
    dst_dir = Path(dst_dir)
    marker = dst_dir / '.stage_complete'

    if marker.exists():
        print(f'[{label}] Reusing existing staged copy at {dst_dir}', flush=True)
        return True

    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f'[{label}] Staging {src_path}  ->  {dst_dir} ...', flush=True)

    if src_path.is_file():
        cmd = ['rsync', '-a', '--info=progress2', str(src_path), f'{dst_dir}/']
    else:
        cmd = ['rsync', '-a', '--info=progress2', f'{src_path}/', f'{dst_dir}/']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    last = ''
    for line in proc.stdout:
        line = line.strip()
        if not line or '%' not in line:
            continue
        last = line
        print(f'\r[{label}] {line}', end='', flush=True)
    proc.wait()
    print(flush=True)

    if proc.returncode != 0:
        print(f'[{label}] rsync FAILED (exit code {proc.returncode}).', flush=True)
        return False

    marker.write_text(datetime.now().isoformat())
    print(f'[{label}] Staging complete.  {last}', flush=True)
    return True


def cleanup_flashblade(dst_dir, success, label):
    """Remove the staged copy only on success; on failure the copy is
    deliberately kept in place to enable fast retries without re-paying the
    copy cost."""
    dst_dir = Path(dst_dir)
    if not dst_dir.exists():
        return
    if success:
        shutil.rmtree(dst_dir)
        print(f'[{label}] Staged copy removed (job succeeded).', flush=True)
    else:
        print(f'[{label}] Staged copy kept for retry: {dst_dir}', flush=True)
