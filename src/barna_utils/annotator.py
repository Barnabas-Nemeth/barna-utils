"""
Interactive multi-region/multi-section polygon annotator for marking TMA
cores (or any other named region-of-interest) on a raster background image,
saved as per-run GeoJSON.

Extracted 2026-08-02 from MERFISH's `03_core_annotation.ipynb` Cell 2.2
("MultiRegionAnnotator v7 -- global canvas edition"), generalized so it works
on any raster source, not just MERSCOPE DAPI mosaics -- specifically so it
can also annotate IHC TMA sections read from raw OME-TIFF/OME-Zarr (see
`image_sources.py`). The interaction logic itself (pan, zoom, draw, select,
edit vertices, rename, delete, undo, confirm, save/reload) was never
MERFISH-specific to begin with -- it only ever touched the background image
through one method (`_get_canvas_display`), so the actual generalization is
narrow: four things that used to be read from MERFISH notebook globals
(`EXPERIMENTS`, `GLOBAL_CANVAS`, `PX_PER_MICRON`, `DOWNSAMPLE`, `ANNOTATIONS`)
are now constructor parameters instead. No interaction/rendering logic was
rewritten.

Built on pure matplotlib (`matplotlib.widgets.Button`/`TextBox` + raw mouse-
event handlers) + IPython.display + shapely/geopandas -- no napari, no
ipywidgets, no GPU/OpenGL dependency, which is why this works fine over a
plain SSH + Jupyter session without the remote-rendering problems napari hit
elsewhere in this project.

New on top of the original MERFISH-only version, added 2026-08-02/03 for
multi-channel IHC use: a per-channel toggle row (click a channel's button to
hide/show it in the composite -- only rendered when the canvas provides
`channel_planes`, i.e. real multi-channel data, not MERFISH's single
pre-rendered DAPI mosaic), a Low%/High% contrast control (adjusts the
percentile cutoff used for display normalization, Apply-triggered rather
than live-dragging since it recomputes CLAHE), and a Maximize/Restore toggle
(resizes the figure to fill most of the screen -- not true OS/browser
fullscreen, which isn't controllable from matplotlib itself, but the closest
deliverable equivalent).

IMPORTANT -- not yet interactively tested: this is a careful, mechanical
refactor of the original interaction logic (global lookups -> constructor-
provided instance state) plus new, additive features built the same way,
verified to import/instantiate correctly and (for the image-reading side)
verified against real production data (see `image_sources.py`'s module
note), but an interactive GUI's actual click/drag/button behavior can't be
verified by an automated agent with no display to click on. Test this for
real (open it, toggle channels, adjust contrast, draw a polygon, save,
reload, maximize) before trusting it in production.
"""
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from shapely.geometry import Point, Polygon, MultiPolygon

_DRAWING = 'drawing'
_NAMING = 'naming'
_RENAMING = 'renaming'

_C_ACTIVE = '#90EE90'
_C_ANNOTATED = '#ADD8E6'
_C_EMPTY = '#F5F5F5'
_C_SELECT_ON = '#FFB6C1'
_C_EDIT_ON = '#FFA500'


def _detect_screen_size():
    """Best-effort screen resolution detection, for sizing the figure and
    the Maximize button. Falls back to a plausible 1080p default if nothing
    else works (e.g. a truly headless environment with no X server and no
    `screeninfo` package)."""
    try:
        import tkinter as _tk
        _r = _tk.Tk()
        _r.withdraw()
        sw, sh = _r.winfo_screenwidth(), _r.winfo_screenheight()
        _r.destroy()
        return sw, sh
    except Exception:
        pass
    try:
        from screeninfo import get_monitors as _gm
        _m = _gm()[0]
        return _m.width, _m.height
    except Exception:
        return 1920, 1080


class MultiRegionAnnotator:
    """Interactive polygon annotator over a per-run/per-section raster canvas.

    Parameters
    ----------
    runs : list[str]
        Identifiers for the regions/sections to annotate (e.g. MERFISH's
        ``['Run1', 'Run2']``, or a list of IHC TMA section names). One
        GeoJSON file is saved per run.
    canvas_provider : callable(run: str) -> dict
        Returns ``{'W': int, 'H': int, 'disp_rgb': np.ndarray}`` for the
        given run -- the display image (already normalized/composited to
        RGB) and its pixel dimensions. See `image_sources.py` for ready-made
        providers (MERSCOPE DAPI mosaic, OME-TIFF, OME-Zarr).
    annotations_dir : str | Path
        Directory GeoJSON files are saved to/loaded from, one per run:
        ``core_shapes_{run}.geojson``.
    px_per_micron : float | dict[str, float]
        Pixels per micron for canvas<->micron coordinate conversion. A
        single float applies to every run; a dict allows per-run values
        (e.g. different sections scanned at different resolutions).
    downsample : float | dict[str, float], default 1
        Downsample factor applied to the canvas relative to full-resolution
        pixels (same shape rules as `px_per_micron`).
    """

    def __init__(self, runs, canvas_provider, annotations_dir, px_per_micron, downsample=1):
        self.runs = list(runs)
        self._canvas_provider = canvas_provider
        self._annotations_dir = Path(annotations_dir)
        self._px_per_micron = px_per_micron
        self._downsample = downsample

        self.current_idx = 0
        self._cores = {run: {} for run in self.runs}
        self._auto_counts = {run: 0 for run in self.runs}

        self._state = _DRAWING
        self._current_verts = []
        self._pending_verts = []
        self._rename_target = None

        self._select_mode = False
        self._selected_core = None
        self._edit_mode = False
        self._dragging = None

        self._confirming = False
        self._redrawing = False
        self._rendering = False

        self._xlim = None
        self._ylim = None
        self._last_scroll_t = 0.0
        self._last_drag_t = 0.0
        self._last_pan_t = 0.0
        self._pan_start = None

        self.fig = None
        self.ax = None
        self._img_artist = None
        self._overlay_artists = []

        self._tb_name = None
        self._status_text = None
        self._btn_select = None
        self._btn_edit = None
        self._btn_rename = None
        self._run_btns = []
        self._all_buttons = []

        self._canvas_cache = {}
        self._maximized = False
        self._normal_figsize = None
        self._btn_maximize = None
        self._active_channels = None   # None == "show every channel"; else a set
        self._channel_btns = []        # [(Button, channel_index), ...]
        self._pct_low = 1.0
        self._pct_high = 99.0
        self._tb_low = None
        self._tb_high = None
        self._load_all_existing()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def _run(self):
        return self.runs[self.current_idx]

    @property
    def _confirmed(self):
        if self._run not in self._cores:
            self._cores[self._run] = {}
        return self._cores[self._run]

    def _px_per_micron_for(self, run):
        return (self._px_per_micron[run] if isinstance(self._px_per_micron, dict)
                else self._px_per_micron)

    def _downsample_for(self, run):
        return (self._downsample[run] if isinstance(self._downsample, dict)
                else self._downsample)

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def _canvas_to_um(self, col, row):
        """Canvas pixel (col, row) -> physical microns (x, y), for the
        currently-displayed run."""
        ppm = self._px_per_micron_for(self._run)
        ds = self._downsample_for(self._run)
        return col * ds / ppm, row * ds / ppm

    def _um_to_canvas(self, x_um, y_um):
        """Physical microns (x, y) -> canvas pixel (col, row), for the
        currently-displayed run."""
        ppm = self._px_per_micron_for(self._run)
        ds = self._downsample_for(self._run)
        return x_um * ppm / ds, y_um * ppm / ds

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _get_canvas(self, run):
        if run not in self._canvas_cache:
            self._canvas_cache[run] = self._canvas_provider(run)
        return self._canvas_cache[run]

    def _bbox_um(self):
        """Shapely Polygon covering the full canvas in micron space."""
        gc = self._get_canvas(self._run)
        if gc is None:
            return None
        ppm = self._px_per_micron_for(self._run)
        ds = self._downsample_for(self._run)
        x_max = gc['W'] * ds / ppm
        y_max = gc['H'] * ds / ppm
        return Polygon([(0, 0), (x_max, 0), (x_max, y_max), (0, y_max)])

    def _img_dims(self):
        """Return (H, W, margin_px) for the current run's canvas."""
        gc = self._get_canvas(self._run)
        if gc is None:
            return 512, 512, 77
        H, W = gc['H'], gc['W']
        return H, W, int(min(H, W) * 0.075)

    # ── Load / save ───────────────────────────────────────────────────────────

    def _load_all_existing(self):
        for run in self.runs:
            path = self._annotations_dir / f'core_shapes_{run}.geojson'
            if not path.exists():
                continue
            gdf = gpd.read_file(path)
            cores = {}
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom.geom_type == 'Polygon':
                    parts = [list(zip(*geom.exterior.coords.xy))[:-1]]
                elif geom.geom_type == 'MultiPolygon':
                    parts = [list(zip(*p.exterior.coords.xy))[:-1] for p in geom.geoms]
                else:
                    continue
                cores[row['name']] = parts
            self._cores[run] = cores
            self._auto_counts[run] = len(cores)
            print(f'  Loaded {len(cores)} cores for {run}.')

    def _is_annotated(self, run):
        return (bool(self._cores.get(run)) or
                (self._annotations_dir / f'core_shapes_{run}.geojson').exists())

    def save(self, run=None):
        """Write one run's annotations to GeoJSON."""
        if run is None:
            run = self._run
        cores = self._cores.get(run, {})
        if not cores:
            return None
        records = []
        for i, (nm, parts) in enumerate(cores.items()):
            geom = (Polygon(parts[0]) if len(parts) == 1
                    else MultiPolygon([Polygon(p) for p in parts]))
            records.append({'geometry': geom, 'name': nm,
                            'core_id': i, 'experiment': run})
        gdf = gpd.GeoDataFrame(records, crs=None)
        self._annotations_dir.mkdir(parents=True, exist_ok=True)
        path = self._annotations_dir / f'core_shapes_{run}.geojson'
        gdf.to_file(path, driver='GeoJSON')
        print(f'  {run}: saved {len(gdf)} cores -> {path.name}')
        return path

    def save_all(self):
        for run in self.runs:
            if self._cores.get(run):
                self.save(run)
        print('All runs saved.')

    def _autosave(self):
        if self._confirmed:
            self.save(self._run)

    # ── Run switching ─────────────────────────────────────────────────────────

    def _switch_to(self, run_name, _=None):
        if run_name == self._run:
            return
        self._autosave()
        self.current_idx = self.runs.index(run_name)
        self._state = _DRAWING
        self._current_verts = []
        self._pending_verts = []
        self._rename_target = None
        self._selected_core = None
        self._dragging = None
        self._select_mode = False
        self._edit_mode = False
        self._xlim = None
        self._ylim = None
        if self._tb_name is not None:
            self._tb_name.set_val(self._default_name())
        for btn_ref in [self._btn_select, self._btn_edit, self._btn_rename]:
            if btn_ref is not None:
                btn_ref.ax.set_facecolor(_C_EMPTY)
                btn_ref.color = _C_EMPTY
        self._update_run_btn_colors()
        self._redraw(reset_zoom=True)
        self._set_status(f'Switched to {run_name}.  {len(self._confirmed)} cores loaded.')

    # ── Button colours ────────────────────────────────────────────────────────

    def _btn_run_color(self, run):
        if run == self._run:
            return _C_ACTIVE
        if self._is_annotated(run):
            return _C_ANNOTATED
        return _C_EMPTY

    def _update_run_btn_colors(self):
        for btn, run in self._run_btns:
            c = self._btn_run_color(run)
            btn.ax.set_facecolor(c)
            btn.color = btn.hovercolor = c
        if self.fig is not None and not self._rendering:
            self.fig.canvas.draw_idle()

    # ── Misc helpers ──────────────────────────────────────────────────────────

    def _default_name(self):
        n = self._auto_counts.get(self._run, 0) + 1
        return f'{self._run}_core_{n:02d}'

    def _set_status(self, msg, color='black'):
        if self._status_text is not None:
            self._status_text.set_text(msg)
            self._status_text.set_color(color)
            if self.fig is not None:
                self.fig.canvas.draw_idle()

    def _get_canvas_display(self, run):
        """Return the RGB display array for a run -- recomposed from cached
        *raw* per-channel planes using the current channel-toggle and
        contrast (percentile cutoff) settings whenever per-channel data is
        available (cheap: no re-read from disk, since the raw planes are
        cached), otherwise the provider's own precomputed image (e.g.
        MERFISH's single already-rendered DAPI mosaic, which has no
        per-channel/contrast controls to apply)."""
        gc = self._get_canvas(run)
        if gc is None:
            raise RuntimeError(f'canvas_provider returned nothing for {run!r}.')
        planes = gc.get('channel_planes')
        if planes:
            from .image_sources import composite_channel_planes
            return composite_channel_planes(
                planes, active=self._active_channels,
                pct_low=self._pct_low, pct_high=self._pct_high)
        return gc['disp_rgb']

    def _find_core_at(self, col_px, row_px):
        x_um, y_um = self._canvas_to_um(col_px, row_px)
        pt = Point(x_um, y_um)
        for name, parts in self._confirmed.items():
            for part in parts:
                if Polygon(part).contains(pt):
                    return name
        return None

    def _find_vertex_at(self, col_px, row_px, threshold=15):
        best, best_dist = None, float(threshold)
        for name, parts in self._confirmed.items():
            for pi, part in enumerate(parts):
                for vi, (x_um, y_um) in enumerate(part):
                    cx, cy = self._um_to_canvas(x_um, y_um)
                    d = ((cx - col_px) ** 2 + (cy - row_px) ** 2) ** 0.5
                    if d < best_dist:
                        best_dist = d
                        best = (name, pi, vi)
        return best

    def _update_title(self):
        if self._state == _NAMING:
            mode = '[NAME PENDING  ->  Confirm or Cancel]'
        elif self._state == _RENAMING:
            mode = '[RENAMING  ->  edit name, then Confirm]'
        elif self._edit_mode:
            mode = '[EDIT VERTICES]'
        elif self._select_mode:
            mode = '[SELECT]'
        else:
            mode = '[DRAW]'
        self.ax.set_title(
            f'{self._run}  |  {len(self._confirmed)} cores  |  {mode}',
            fontsize=8, pad=2)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_overlays(self):
        """Add all non-image artists; append each to self._overlay_artists."""
        colors = plt.cm.Set1.colors

        for i, (name, parts) in enumerate(self._confirmed.items()):
            c = colors[i % len(colors)]
            is_sel = (name == self._selected_core)
            edge_c = 'red' if is_sel else c
            lw = 3 if is_sel else 2

            for pi, part in enumerate(parts):
                pts_c = [self._um_to_canvas(x, y) for x, y in part]
                xs = [p[0] for p in pts_c]
                ys = [p[1] for p in pts_c]

                patch = mpatches.Polygon(list(zip(xs, ys)), closed=True,
                    facecolor=c, edgecolor=edge_c, alpha=0.25, linewidth=lw)
                self._overlay_artists.append(self.ax.add_patch(patch))

                if len(parts) > 1:
                    self._overlay_artists.append(
                        self.ax.scatter(xs, ys, c=[c] * len(xs), s=10, zorder=4, alpha=0.7))

                if self._edit_mode and (self._selected_core is None
                                        or name == self._selected_core):
                    dragging_this = (self._dragging is not None
                                     and self._dragging[0] == name
                                     and self._dragging[1] == pi)
                    self._overlay_artists.append(
                        self.ax.scatter(xs, ys,
                            c='yellow' if dragging_this else 'white',
                            s=90, zorder=7, edgecolors=c, linewidths=2))

            largest = max([Polygon(p) for p in parts], key=lambda p: p.area)
            cx, cy = self._um_to_canvas(largest.centroid.x, largest.centroid.y)
            label = name.split('_', 2)[-1] if name.count('_') >= 2 else name
            self._overlay_artists.append(
                self.ax.text(cx, cy, label, color='white', fontsize=8,
                             ha='center', va='center', fontweight='bold',
                             clip_on=True,
                             bbox=dict(facecolor=c, alpha=0.85, pad=2,
                                       boxstyle='round,pad=0.3')))

        if self._pending_verts:
            xs = [v[0] for v in self._pending_verts]
            ys = [v[1] for v in self._pending_verts]
            self._overlay_artists.append(self.ax.add_patch(mpatches.Polygon(
                list(zip(xs, ys)), closed=True,
                facecolor='gold', edgecolor='orange', alpha=0.35,
                linewidth=2.5, linestyle='--')))
            self._overlay_artists.append(
                self.ax.scatter(xs, ys, c='orange', s=20, zorder=5))

        if self._current_verts:
            xs = [v[0] for v in self._current_verts]
            ys = [v[1] for v in self._current_verts]
            self._overlay_artists.extend(
                self.ax.plot(xs + [xs[0]], ys + [ys[0]], 'r--', lw=1.5, alpha=0.7))
            self._overlay_artists.append(
                self.ax.scatter(xs, ys, c='red', s=25, zorder=5))
            self._overlay_artists.append(
                self.ax.scatter(xs[0:1], ys[0:1], c='yellow', s=70, zorder=6, marker='*'))

        self._update_title()

    def _refresh_overlays(self):
        """Remove old overlays, redraw them. Does NOT touch the AxesImage."""
        self._rendering = True
        try:
            for a in self._overlay_artists:
                try:
                    a.remove()
                except Exception:
                    pass
            self._overlay_artists = []
            self._draw_overlays()
            self.fig.canvas.draw_idle()
        finally:
            self._rendering = False

    def _redraw(self, reset_zoom=False):
        """Full redraw: clear axes, re-add canvas image, then overlays.
        Only called on run switch or first load."""
        if self._redrawing:
            return
        self._redrawing = True
        self._rendering = True
        try:
            canvas_disp = self._get_canvas_display(self._run)
            H, W, mg = self._img_dims()

            self.ax.clear()
            self._overlay_artists = []
            self.ax.set_autoscale_on(False)
            self.ax.set_facecolor('#1e1e1e')
            self._img_artist = self.ax.imshow(canvas_disp, origin='upper')
            self.ax.set_xticks([])
            self.ax.set_yticks([])

            self._draw_overlays()

            if not reset_zoom and self._xlim is not None:
                self.ax.set_xlim(self._xlim)
                self.ax.set_ylim(self._ylim)
            else:
                self.ax.set_xlim(-mg, W + mg)
                self.ax.set_ylim(H + mg, -mg)
                self._xlim = None
                self._ylim = None

            self.fig.canvas.draw_idle()
        finally:
            self._redrawing = False
            self._rendering = False

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        if self._state == _NAMING:
            self._set_status('Polygon pending — Confirm or Cancel first.', 'darkorange')
            return
        if self._state == _RENAMING:
            self._set_status('Rename in progress — edit name and Confirm, or Cancel.',
                             'darkorange')
            return

        if self._edit_mode:
            if event.button == 1:
                hit = self._find_vertex_at(event.xdata, event.ydata)
                if hit is not None:
                    name, pi, vi = hit
                    if self._selected_core is None or name == self._selected_core:
                        self._dragging = hit
                        self._set_status(
                            f'Dragging vertex {vi} of "{name}" part {pi + 1} — release to drop.',
                            'darkorange')
                    else:
                        self._set_status(
                            f'Vertex belongs to "{name}" — select that core first.', 'gray')
                else:
                    self._set_status('No vertex nearby — zoom in or click closer.', 'gray')
            elif event.button == 2:
                self._pan_start = (list(self.ax.get_xlim()), list(self.ax.get_ylim()),
                                   event.x, event.y)
            return

        if self._select_mode:
            if event.button == 1:
                hit = self._find_core_at(event.xdata, event.ydata)
                if hit is not None:
                    self._selected_core = hit
                    if self._tb_name is not None:
                        self._tb_name.set_val(hit)
                        try:
                            self._tb_name.begin_typing(None)
                        except Exception:
                            pass
                    self._set_status(
                        f'Selected "{hit}" — Del sel. to delete, Rename to rename.',
                        'darkred')
                else:
                    self._selected_core = None
                    self._set_status('No core at that position.', 'gray')
                self._refresh_overlays()
            elif event.button == 2:
                self._pan_start = (list(self.ax.get_xlim()), list(self.ax.get_ylim()),
                                   event.x, event.y)
            return

        # Draw mode
        if event.button == 1:
            self._current_verts.append((event.xdata, event.ydata))
            self._refresh_overlays()
        elif event.button == 3:
            n = len(self._current_verts)
            if n < 3:
                self._set_status(f'Need >= 3 vertices (have {n}).', 'red')
                return
            self._pending_verts = list(self._current_verts)
            self._current_verts = []
            self._state = _NAMING
            if self._tb_name is not None:
                self._tb_name.set_val(self._default_name())
            self._refresh_overlays()
            self._set_status(
                f'Polygon closed ({n} verts).  Edit name if needed, then Confirm.',
                'darkgreen')
        elif event.button == 2:
            if self._current_verts:
                self._current_verts.pop()
                self._refresh_overlays()
            else:
                self._pan_start = (list(self.ax.get_xlim()), list(self.ax.get_ylim()),
                                   event.x, event.y)

    def _on_key(self, event):
        if event.key == 'enter' and self._state in (_NAMING, _RENAMING):
            self._on_confirm()

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        xc, yc = event.xdata, event.ydata
        factor = 0.82 if event.button == 'up' else (1.0 / 0.82)
        new_xlim = [xc + (x - xc) * factor for x in self.ax.get_xlim()]
        new_ylim = [yc + (y - yc) * factor for y in self.ax.get_ylim()]
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self._xlim = tuple(new_xlim)
        self._ylim = tuple(new_ylim)
        now = time.monotonic()
        if now - self._last_scroll_t > 0.15:
            self._last_scroll_t = now
            self.fig.canvas.draw_idle()

    def _on_axes_enter(self, event):
        if event.inaxes == self.ax and self.fig is not None:
            self.fig.canvas.capture_scroll = True

    def _on_axes_leave(self, event):
        if event.inaxes == self.ax and self.fig is not None:
            self.fig.canvas.capture_scroll = False

    def _on_motion(self, event):
        if self._pan_start is not None:
            ax_bb = self.ax.get_window_extent()
            if ax_bb.width > 0 and ax_bb.height > 0:
                xl0, yl0, xp0, yp0 = self._pan_start
                xs = (xl0[1] - xl0[0]) / ax_bb.width
                ys = (yl0[1] - yl0[0]) / ax_bb.height
                dx = (event.x - xp0) * xs
                dy = (event.y - yp0) * ys
                new_xl = [xl0[0] - dx, xl0[1] - dx]
                new_yl = [yl0[0] - dy, yl0[1] - dy]
                self.ax.set_xlim(new_xl)
                self.ax.set_ylim(new_yl)
                self._xlim = tuple(new_xl)
                self._ylim = tuple(new_yl)
                now = time.monotonic()
                if now - self._last_pan_t >= 0.20:
                    self._last_pan_t = now
                    self.fig.canvas.draw_idle()
            return

        if self._dragging is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        name, pi, vi = self._dragging
        self._confirmed[name][pi][vi] = self._canvas_to_um(event.xdata, event.ydata)
        now = time.monotonic()
        if now - self._last_drag_t >= 0.05:
            self._last_drag_t = now
            self._refresh_overlays()

    def _on_release(self, event):
        if self._pan_start is not None and event.button == 2:
            self._pan_start = None
            self.fig.canvas.draw_idle()
            return
        if self._dragging is not None:
            self._dragging = None
            self._refresh_overlays()
            self._set_status('Vertex moved — Save GeoJSON to persist.', 'darkgreen')

    # ── Button callbacks ──────────────────────────────────────────────────────

    def _on_confirm(self, _=None):
        if self._confirming:
            return
        if self._state not in (_NAMING, _RENAMING):
            return
        self._confirming = True
        try:
            name = (self._tb_name.text.strip()
                    if self._tb_name is not None else self._default_name())
            if not name:
                self._set_status('Name cannot be empty.', 'red')
                return

            if self._state == _RENAMING:
                old = self._rename_target
                if old and old in self._confirmed:
                    if name != old:
                        if name in self._confirmed:
                            self._set_status(f'Name "{name}" already exists.', 'red')
                            return
                        parts = self._confirmed.pop(old)
                        self._confirmed[name] = parts
                        if self._selected_core == old:
                            self._selected_core = name
                self._state = _DRAWING
                self._rename_target = None
                if self._tb_name is not None:
                    self._tb_name.set_val(self._default_name())
                self._update_run_btn_colors()
                self._refresh_overlays()
                self._set_status(f'Renamed to "{name}".', 'darkgreen')
                return

            if not self._pending_verts:
                return
            pts_um = [self._canvas_to_um(c, r) for c, r in self._pending_verts]

            bbox = self._bbox_um()
            if bbox is not None:
                clipped = Polygon(pts_um).intersection(bbox)
                if clipped.is_empty or clipped.geom_type not in ('Polygon', 'MultiPolygon'):
                    self._set_status('Polygon outside canvas after clipping — discarded.', 'red')
                    self._pending_verts = []
                    self._state = _DRAWING
                    self._refresh_overlays()
                    return
                if clipped.geom_type == 'MultiPolygon':
                    clipped = max(clipped.geoms, key=lambda g: g.area)
                pts_um = list(clipped.exterior.coords)[:-1]

            if name in self._confirmed:
                self._confirmed[name].append(pts_um)
                n_parts = len(self._confirmed[name])
                msg = f'Added fragment #{n_parts} to "{name}".  Total: {len(self._confirmed)}.'
            else:
                self._confirmed[name] = [pts_um]
                self._auto_counts[self._run] += 1
                msg = (f'Saved "{name}" ({len(pts_um)} verts).  '
                       f'Total: {len(self._confirmed)}.')

            self._pending_verts = []
            self._state = _DRAWING
            if self._tb_name is not None:
                self._tb_name.set_val(self._default_name())
            self._update_run_btn_colors()
            self._refresh_overlays()
            self._set_status(msg, 'darkgreen')
        finally:
            self._confirming = False

    def _on_cancel(self, _=None):
        self._current_verts = []
        self._pending_verts = []
        self._state = _DRAWING
        self._rename_target = None
        if self._tb_name is not None:
            self._tb_name.set_val(self._default_name())
        self._refresh_overlays()
        self._set_status('Cancelled.', 'gray')

    def _on_undo(self, _=None):
        if self._state == _DRAWING and self._current_verts:
            self._current_verts.pop()
            self._refresh_overlays()

    def _on_rename_selected(self, _=None):
        if self._selected_core is None:
            self._set_status('Select a core first (Select mode + click inside it).', 'red')
            return
        self._rename_target = self._selected_core
        self._state = _RENAMING
        if self._tb_name is not None:
            self._tb_name.set_val(self._selected_core)
        self._refresh_overlays()
        self._set_status(
            f'Renaming "{self._selected_core}" — edit name above, then Confirm or Cancel.',
            'darkorange')

    def _on_toggle_select(self, _=None):
        self._select_mode = not self._select_mode
        if self._select_mode and self._edit_mode:
            self._edit_mode = False
            self._dragging = None
            if self._btn_edit is not None:
                self._btn_edit.ax.set_facecolor(_C_EMPTY)
                self._btn_edit.color = _C_EMPTY
        if not self._select_mode:
            self._selected_core = None
        c = _C_SELECT_ON if self._select_mode else _C_EMPTY
        if self._btn_select is not None:
            self._btn_select.ax.set_facecolor(c)
            self._btn_select.color = c
        self._set_status(
            'SELECT — click inside a polygon to highlight it.' if self._select_mode
            else 'DRAW mode.', 'gray')
        self._refresh_overlays()

    def _on_toggle_edit(self, _=None):
        self._edit_mode = not self._edit_mode
        if self._edit_mode:
            self._select_mode = False
            if self._btn_select is not None:
                self._btn_select.ax.set_facecolor(_C_EMPTY)
                self._btn_select.color = _C_EMPTY
            msg = (f'EDIT VERTICES of "{self._selected_core}" — drag white dots.  '
                   'Click Edit to exit.'
                   if self._selected_core else
                   'EDIT VERTICES — Select a core first for faster rendering.  '
                   'Click Edit to exit.')
            self._set_status(msg, 'darkorange')
        else:
            self._dragging = None
            self._set_status('DRAW mode.', 'gray')
        c = _C_EDIT_ON if self._edit_mode else _C_EMPTY
        if self._btn_edit is not None:
            self._btn_edit.ax.set_facecolor(c)
            self._btn_edit.color = c
        self._refresh_overlays()

    def _on_delete_selected(self, _=None):
        if self._selected_core is None:
            self._set_status('Nothing selected — enter Select mode and click a polygon.', 'red')
            return
        nm = self._selected_core
        del self._confirmed[nm]
        self._auto_counts[self._run] = max(0, self._auto_counts[self._run] - 1)
        self._selected_core = None
        self._update_run_btn_colors()
        self._refresh_overlays()
        self._set_status(f'Deleted "{nm}".', 'darkorange')

    def _on_delete_last(self, _=None):
        if self._confirmed:
            nm = list(self._confirmed)[-1]
            del self._confirmed[nm]
            self._auto_counts[self._run] = max(0, self._auto_counts[self._run] - 1)
            self._dragging = None
            self._update_run_btn_colors()
            self._refresh_overlays()
            self._set_status(f'Deleted last: "{nm}".', 'darkorange')

    def _on_reset_zoom(self, _=None):
        H, W, mg = self._img_dims()
        self._xlim = None
        self._ylim = None
        self.ax.set_xlim(-mg, W + mg)
        self.ax.set_ylim(H + mg, -mg)
        self.fig.canvas.draw_idle()
        self._set_status('Zoom reset.', 'gray')

    def _on_toggle_maximize(self, _=None):
        """Resize the figure to fill most of the detected screen (or back to
        its normal compact size). Not true OS/browser fullscreen -- that
        isn't controllable from matplotlib itself -- but the closest
        deliverable equivalent: a much larger canvas, one click away.
        Requires an interactive backend that supports resizing an
        already-open figure (`%matplotlib widget` / ipympl does)."""
        if self.fig is None:
            return
        if not self._maximized:
            self._normal_figsize = self.fig.get_size_inches()
            sw, sh = _detect_screen_size()
            dpi = self.fig.dpi
            self.fig.set_size_inches((sw / dpi) * 0.97, (sh / dpi) * 0.90)
            self._maximized = True
            if self._btn_maximize is not None:
                self._btn_maximize.label.set_text('Restore')
            self._set_status('Maximized.', 'navy')
        else:
            if self._normal_figsize is not None:
                self.fig.set_size_inches(*self._normal_figsize)
            self._maximized = False
            if self._btn_maximize is not None:
                self._btn_maximize.label.set_text('Maximize')
            self._set_status('Restored to normal size.', 'gray')
        self.fig.canvas.draw_idle()

    def _on_toggle_channel(self, channel_idx):
        if self._active_channels is None:
            # First toggle: start from "all on" (every known channel), then
            # remove the one just clicked.
            gc = self._get_canvas(self._run)
            self._active_channels = set(gc['channel_planes'].keys())
        if channel_idx in self._active_channels:
            self._active_channels.discard(channel_idx)
        else:
            self._active_channels.add(channel_idx)
        for btn, ci in self._channel_btns:
            on = self._active_channels is None or ci in self._active_channels
            btn.ax.set_alpha(1.0 if on else 0.25)
        self._redraw()
        n_on = len(self._active_channels)
        self._set_status(f'{n_on} channel(s) shown.', 'navy')

    def _on_apply_contrast(self, _=None):
        """Read the Low%/High% textboxes and re-render with the new
        percentile cutoffs. Explicit Apply (not a live-dragging slider) --
        recomposing calls CLAHE again per channel, which is cheap compared
        to a disk re-read but not cheap enough for smooth continuous drag
        on a large canvas."""
        if self._tb_low is None or self._tb_high is None:
            return
        try:
            lo = float(self._tb_low.text)
            hi = float(self._tb_high.text)
        except ValueError:
            self._set_status('Low/High must be numbers (0-100).', 'red')
            return
        if not (0 <= lo < hi <= 100):
            self._set_status('Need 0 <= Low < High <= 100.', 'red')
            return
        self._pct_low, self._pct_high = lo, hi
        self._redraw()
        self._set_status(f'Contrast updated: {lo:g}-{hi:g} percentile.', 'navy')

    def _on_save(self, _=None):
        path = self.save()
        if path:
            self._set_status(f'Saved {len(self._confirmed)} cores -> {path.name}', 'navy')
        self._update_run_btn_colors()

    def _on_reload(self, _=None):
        """Reset all drawing/rendering state and force a clean redraw."""
        self._state = _DRAWING
        self._current_verts = []
        self._pending_verts = []
        self._rename_target = None
        self._selected_core = None
        self._dragging = None
        self._select_mode = False
        self._edit_mode = False
        self._redrawing = False
        self._rendering = False
        self._confirming = False
        self._xlim = None
        self._ylim = None
        for btn_ref in [self._btn_select, self._btn_edit, self._btn_rename]:
            if btn_ref is not None:
                btn_ref.ax.set_facecolor(_C_EMPTY)
                btn_ref.color = _C_EMPTY
        if self._tb_name is not None:
            self._tb_name.set_val(self._default_name())
        try:
            self._redraw(reset_zoom=True)
            if self.fig is not None:
                self.fig.canvas.flush_events()
        except Exception as e:
            print(f'[Reload] redraw error: {e}')
            if self.fig is not None:
                self.fig.canvas.draw_idle()
        self._set_status('Reloaded.  All drawing state and zoom reset.', 'navy')

    # ── display() ─────────────────────────────────────────────────────────────

    def display(self):
        """Open the interactive figure. Calling cell must begin with
        `%matplotlib widget` (or an equivalent interactive backend)."""
        _sw, _sh = _detect_screen_size()
        _DPI = 96
        _fig_w = min((_sw / _DPI) * 0.58, 12.0)
        _fig_h = min((_sh / _DPI) * 0.72, 8.5)

        self.fig, self.ax = plt.subplots(figsize=(_fig_w, _fig_h), dpi=_DPI)
        # Extra bottom margin vs. the original layout to fit a 3rd control
        # row (contrast cutoffs); extra top margin for the channel-toggle
        # row when the canvas has per-channel data.
        self.fig.subplots_adjust(left=0.06, right=0.98, bottom=0.22, top=0.85)

        self._status_text = self.fig.text(
            0.5, 0.195, 'Ready.',
            ha='center', va='center', fontsize=9, color='black',
            transform=self.fig.transFigure,
            bbox=dict(facecolor='#f4f4f4', edgecolor='#cccccc',
                      pad=0.8, boxstyle='square'))

        row_y = 0.925
        btn_w, btn_h = 0.15, 0.033
        for col_i, run in enumerate(self.runs):
            x = 0.07 + col_i * (btn_w + 0.01)
            c = self._btn_run_color(run)
            ax_b = self.fig.add_axes([x, row_y, btn_w, btn_h])
            btn = Button(ax_b, run, color=c, hovercolor=c)
            btn.on_clicked(lambda _, r=run: self._switch_to(r))
            self._run_btns.append((btn, run))

        # Channel-toggle row -- only built if the current run's canvas has
        # per-channel data (MERFISH's single-image DAPI mosaic doesn't, and
        # simply won't show this row at all).
        gc0 = self._get_canvas(self._run)
        channel_planes = gc0.get('channel_planes') if gc0 else None
        if channel_planes:
            ch_row_y = 0.878
            n_ch = len(channel_planes)
            ch_w = min(0.12, 0.90 / n_ch)
            for i, (c_idx, (color, _rgb, _raw)) in enumerate(channel_planes.items()):
                x = 0.07 + i * (ch_w + 0.008)
                ax_c = self.fig.add_axes([x, ch_row_y, ch_w, 0.030])
                btn = Button(ax_c, f'Ch{c_idx}', color=color, hovercolor=color)
                btn.on_clicked(lambda _, ci=c_idx: self._on_toggle_channel(ci))
                self._channel_btns.append((btn, c_idx))

        bh = 0.048
        r1y = 0.140
        row1 = [
            ('undo', 0.060, 0.085),
            ('cancel', 0.155, 0.085),
            ('name', 0.250, 0.220),
            ('confirm', 0.480, 0.100),
            ('resetz', 0.592, 0.085),
            ('reload', 0.688, 0.085),
            ('maximize', 0.783, 0.100),
        ]
        axs1 = {k: self.fig.add_axes([x, r1y, w, bh]) for k, x, w in row1}

        r2y = 0.078
        row2 = [
            ('select', 0.060, 0.085),
            ('delsel', 0.155, 0.085),
            ('rename', 0.250, 0.085),
            ('dellast', 0.345, 0.085),
            ('edit', 0.440, 0.085),
            ('save', 0.755, 0.215),
        ]
        axs2 = {k: self.fig.add_axes([x, r2y, w, bh]) for k, x, w in row2}

        # Row 3 -- contrast (percentile cutoff) controls. Only meaningful
        # when the canvas has per-channel raw data to recompute from; shown
        # regardless (harmless no-op via _on_apply_contrast's guard) since
        # every provider that supports it uses the same channel_planes shape.
        r3y = 0.016
        row3 = [
            ('lowlbl', 0.060, 0.070),
            ('low', 0.135, 0.090),
            ('highlbl', 0.250, 0.075),
            ('high', 0.330, 0.090),
            ('apply', 0.440, 0.100),
        ]
        axs3 = {k: self.fig.add_axes([x, r3y, w, bh]) for k, x, w in row3}

        btn_undo = Button(axs1['undo'], 'Undo vtx', color='lightyellow')
        btn_cancel = Button(axs1['cancel'], 'Cancel', color='lightyellow')
        self._tb_name = TextBox(axs1['name'], '',
                                initial=self._default_name(), textalignment='left')
        btn_confirm = Button(axs1['confirm'], 'Confirm ✓', color='lightgreen')
        btn_resetz = Button(axs1['resetz'], 'Reset Z', color='lightcyan')
        btn_reload = Button(axs1['reload'], 'Reload', color='#ffe4b5')
        btn_maximize = Button(axs1['maximize'], 'Maximize', color='#e0e0ff')

        btn_select = Button(axs2['select'], 'Select', color=_C_EMPTY)
        btn_delsel = Button(axs2['delsel'], 'Del sel.', color='lightsalmon')
        btn_rename = Button(axs2['rename'], 'Rename', color=_C_EMPTY)
        btn_dellast = Button(axs2['dellast'], 'Del last', color='lightsalmon')
        btn_edit = Button(axs2['edit'], 'Edit', color=_C_EMPTY)
        btn_save = Button(axs2['save'], 'Save GeoJSON', color='lightblue')

        axs3['lowlbl'].axis('off')
        axs3['lowlbl'].text(0.5, 0.5, 'Low %', ha='center', va='center', fontsize=8,
                            transform=axs3['lowlbl'].transAxes)
        axs3['highlbl'].axis('off')
        axs3['highlbl'].text(0.5, 0.5, 'High %', ha='center', va='center', fontsize=8,
                             transform=axs3['highlbl'].transAxes)
        self._tb_low = TextBox(axs3['low'], '', initial=str(self._pct_low), textalignment='center')
        self._tb_high = TextBox(axs3['high'], '', initial=str(self._pct_high), textalignment='center')
        btn_apply = Button(axs3['apply'], 'Apply', color='lightcyan')

        btn_undo.on_clicked(self._on_undo)
        btn_cancel.on_clicked(self._on_cancel)
        btn_confirm.on_clicked(self._on_confirm)
        btn_resetz.on_clicked(self._on_reset_zoom)
        btn_reload.on_clicked(self._on_reload)
        btn_maximize.on_clicked(self._on_toggle_maximize)
        btn_select.on_clicked(self._on_toggle_select)
        btn_delsel.on_clicked(self._on_delete_selected)
        btn_rename.on_clicked(self._on_rename_selected)
        btn_dellast.on_clicked(self._on_delete_last)
        btn_edit.on_clicked(self._on_toggle_edit)
        btn_save.on_clicked(self._on_save)
        btn_apply.on_clicked(self._on_apply_contrast)

        self._btn_select = btn_select
        self._btn_edit = btn_edit
        self._btn_rename = btn_rename
        self._btn_maximize = btn_maximize
        self._all_buttons = (
            [btn_undo, btn_cancel, btn_confirm, btn_resetz, btn_reload, btn_maximize,
             btn_select, btn_delsel, btn_rename, btn_dellast, btn_edit, btn_save,
             btn_apply] +
            [b for b, _ in self._run_btns] + [b for b, _ in self._channel_btns])

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('axes_enter_event', self._on_axes_enter)
        self.fig.canvas.mpl_connect('axes_leave_event', self._on_axes_leave)

        self._redraw()
        plt.show()
        n_ann = sum(1 for run in self.runs if self._is_annotated(run))
        print(f'MultiRegionAnnotator ready — {len(self.runs)} runs, {n_ann} already annotated.')
