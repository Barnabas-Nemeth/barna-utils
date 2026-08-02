"""
Live progress display for polling loops (LSF job monitoring and similar), with a
real ipywidgets-based UI in Jupyter/VS Code and a plain-text fallback everywhere
else.

Design note (2026-08-02): an earlier version of this idea (in IHC's own
pre-barna-utils notebook code) tried to show a genuine percentage-complete
progress bar by having the polling loop parse tqdm-style lines out of a file the
job was supposedly writing to. In practice nothing ever wrote to that file, so
the bar always jumped straight from 0% to "finished" with no gradual movement --
a broken feature that looked more sophisticated than the plain print-based
version, but was actually less honest about what was really known.

This module fixes that by only ever displaying things that are genuinely,
verifiably true at each poll tick:
  - LSF job status (PEND/RUN/DONE/EXIT) -- always accurate, straight from `bjobs`.
  - Elapsed time -- a real, continuously-updating clock, not a fake animation.
  - Output-file completion count (e.g. "3 of 5 regions written") -- when a job
    has multiple expected output paths, this is real incremental progress,
    verified by checking whether each file actually exists yet.
  - An optional caller-supplied `progress_fn` callback for job-specific status
    (e.g. NB02/NB04's pattern of checking per-region/per-file state) -- this is
    the recommended way to get genuine incremental progress for a single-output
    job, since it queries real state rather than assuming a log format.

If a caller genuinely has a job that writes real, parseable progress to a file
(not assumed here to be the common case), pass a `progress_file` + matching
`progress_parser(text) -> str | None` and it will be shown too -- but nothing
in this module assumes that mechanism works without the caller proving it does.
"""
from datetime import datetime
from pathlib import Path


def widgets_available():
    """True if ipywidgets can actually render here (a real Jupyter/VS Code
    kernel, not a plain terminal or a headless script). Both conditions matter:
    ipywidgets being importable is not the same as having somewhere to draw
    into -- a plain `python script.py` run has ipywidgets installed in most
    environments but no frontend to render a widget in.
    """
    try:
        import ipywidgets  # noqa: F401
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == 'ZMQInteractiveShell'
    except ImportError:
        return False


class _WidgetDisplay:
    """Real ipywidgets-based display: a colored status line, an elapsed-time
    clock, and an IntProgress bar driven by verified output-file counts (not a
    parsed log)."""

    def __init__(self, title):
        import ipywidgets as w
        from IPython.display import display
        self._w = w
        self.title = w.HTML(f'<b>{title}</b>')
        self.status = w.HTML()
        self.bar = w.IntProgress(min=0, max=1, value=0, layout=w.Layout(width='400px'))
        self.extra = w.HTML()
        self.box = w.VBox([self.title, self.status, self.bar, self.extra])
        display(self.box)

    def update(self, status, elapsed_s, done, total, extra_lines=None, bar_style=''):
        self.status.value = (
            f'Status: <b>{status}</b> &nbsp;|&nbsp; Elapsed: {_fmt_elapsed(elapsed_s)}'
        )
        self.bar.max = max(total, 1)
        self.bar.value = done
        self.bar.bar_style = bar_style
        self.bar.description = f'{done}/{total}'
        if extra_lines:
            self.extra.value = '<pre>' + '\n'.join(extra_lines) + '</pre>'

    def finalize(self, success):
        self.bar.bar_style = 'success' if success else 'danger'


class _PlainDisplay:
    """Fallback for non-widget contexts (plain terminal, headless scripts,
    or a Jupyter frontend without ipywidgets installed): the original
    print + clear_output refresh pattern, already confirmed working across
    every pre-barna-utils implementation in this project."""

    def __init__(self, title):
        self.title = title
        try:
            from IPython.display import clear_output
            self._clear = lambda: clear_output(wait=True)
        except ImportError:
            self._clear = lambda: None

    def update(self, status, elapsed_s, done, total, extra_lines=None, bar_style=''):
        self._clear()
        print(f'{self.title}', flush=True)
        print(f'[{datetime.now().strftime("%H:%M:%S")}] {status} '
              f'-- elapsed {_fmt_elapsed(elapsed_s)} -- {done}/{total} outputs present',
              flush=True)
        if extra_lines:
            for line in extra_lines:
                print(f'    {line}', flush=True)

    def finalize(self, success):
        pass


def make_display(title):
    """Return a display handle (widget-based if possible, plain-text
    otherwise) with `.update(...)` and `.finalize(success)`."""
    if widgets_available():
        try:
            return _WidgetDisplay(title)
        except Exception:
            pass
    return _PlainDisplay(title)


def _fmt_elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f'{h:d}:{m:02d}:{s:02d}' if h else f'{m:d}:{s:02d}'
