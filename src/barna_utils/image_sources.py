"""
Canvas providers for `MultiRegionAnnotator` (see `annotator.py`) -- functions
that read a raster image source and return the
``{'W': int, 'H': int, 'disp_rgb': np.ndarray}`` dict the annotator needs to
display a background image for a given run/section.

Two new, generic readers, built 2026-08-02 specifically so the annotator
(originally MERFISH-only, reading MERSCOPE DAPI mosaics) can also be used for
IHC TMA annotation, which stores images as pyramidal OME-TIFF and OME-Zarr
(OME-NGFF v0.4) rather than MERSCOPE's own format:

- `ome_tiff_canvas_provider` -- reads via `tifffile`'s zarr bridge
  (`zarr.open(tifffile.TiffFile(path).aszarr(), mode='r')`), the same
  mechanism IHC's own stitching pipeline already uses
  (`01_IHC_stitching.ipynb` Cell 3.1's `open_tiff_zarr()`) to read its input
  OME-TIFFs, so this reads the exact format IHC already produces/consumes.
- `ome_zarr_canvas_provider` -- reads a standard OME-NGFF v0.4 multiscale
  zarr store (the format IHC's stitching job writes,
  `root.attrs['multiscales']` with numbered pyramid-level datasets), again
  matching IHC's own actual output structure rather than a guessed one.

Canvas dicts returned here also include `channel_planes` (each channel's
CLAHE-equalized plane + assigned color, precomputed once) whenever more than
a single channel is involved, so `MultiRegionAnnotator`'s live per-channel
toggle buttons can cheaply re-blend a subset of channels without re-reading
or re-equalizing anything.

The MERSCOPE DAPI-mosaic provider is NOT here -- it stays in MERFISH's own
notebook/scripts, since building it requires MERSCOPE-specific mosaic-tile
assembly logic (`dapi_utils.py`) that has no equivalent generic meaning.
Any canvas provider is just a plain callable `(run) -> dict`, so MERFISH's
existing `GLOBAL_CANVAS`-based approach can be wrapped in a one-line lambda
and passed to the generalized `MultiRegionAnnotator` unchanged.

NOT YET VALIDATED against a real IHC file end-to-end (no display available to
an automated agent to actually confirm the rendered image looks correct) --
the reading/reshaping logic was written directly against IHC's own
documented zarr structure, not guessed, but treat this as needing a real
smoke test (open the annotator against an actual TMA file, confirm the image
looks right) before relying on it.
"""
import numpy as np
import zarr
from skimage import exposure as _sk_exp


def _clahe_equalize(img, pct_low=1, pct_high=99, clip_limit=0.03, nbins=256, use_clahe=True):
    """Percentile-rescale (the "top/bottom cutoff" a viewer's contrast
    control usually adjusts -- `pct_low`/`pct_high` decide which percentile
    of pixel intensities maps to black/white) then, by default, also
    adaptive-histogram-equalize -- same recipe as MERFISH's
    `dapi_utils.clahe_equalize()`, reused here so both platforms' annotator
    canvases get the same visual treatment (dim and bright regions equally
    visible). `use_clahe=False` skips the (slower) equalization step and
    returns the plain percentile-rescaled image, for callers that want fast
    interactive cutoff adjustment without paying for CLAHE on every change.
    """
    valid = img[img > 0]
    if valid.size == 0:
        return img.astype(np.float32)
    vmin, vmax = np.percentile(valid, [pct_low, pct_high])
    if vmax <= vmin:
        return np.clip(img.astype(np.float32) / max(vmax, 1e-6), 0.0, 1.0)
    resc = _sk_exp.rescale_intensity(img.astype(np.float32), in_range=(vmin, vmax), out_range=(0.0, 1.0))
    resc = np.clip(resc, 0.0, 1.0)
    if not use_clahe:
        return resc
    return _sk_exp.equalize_adapthist(resc, clip_limit=clip_limit, nbins=nbins)


_DEFAULT_CHANNEL_COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow',
                           'white', 'orange', 'purple']


def _normalize_channel_spec(channels):
    """Convert any of the three accepted `channels` forms into the one
    canonical shape everything downstream (compositing, live toggling)
    actually works with: `{channel_index: color}`.

    - a single int -> `{idx: 'white'}` (renders as grayscale).
    - a list of ints -> each assigned a distinct default color (falls back
      to the same R/G/B-first ordering as before for the first 3, so
      existing 3-channel calls look the same as before this refactor, but
      any number of channels is now supported here too, not just 3).
    - a dict -> returned as-is (already canonical).
    """
    if isinstance(channels, dict):
        return dict(channels)
    if isinstance(channels, (int, np.integer)):
        return {int(channels): 'white'}
    return {int(c): _DEFAULT_CHANNEL_COLORS[i % len(_DEFAULT_CHANNEL_COLORS)]
            for i, c in enumerate(channels)}


def _compute_channel_planes(arr, channel_spec):
    """`arr` is (C, H, W); `channel_spec` is `{channel_index: color}`
    (see `_normalize_channel_spec`). Returns `{channel_index: (color,
    rgb_array, raw_plane)}` -- the *raw* (unprocessed) per-channel pixel
    data, read from disk exactly once here and cached by the caller. Kept
    raw (not pre-CLAHE'd) specifically so contrast/cutoff adjustments and
    channel toggling (see `composite_channel_planes`) are cheap in-memory
    recomputations, never a re-read.
    """
    import matplotlib.colors as mcolors
    planes = {}
    for c, color in channel_spec.items():
        raw = np.asarray(arr[c])
        rgb = np.array(mcolors.to_rgb(color), dtype=np.float32)
        planes[c] = (color, rgb, raw)
    return planes


def composite_channel_planes(channel_planes, active=None, pct_low=1, pct_high=99,
                             use_clahe=True):
    """Blend cached raw per-channel planes (see `_compute_channel_planes`)
    into a displayable RGB array, applying the percentile cutoff (and,
    unless disabled, CLAHE) at blend time.

    `active`, if given, restricts the blend to a subset of channel indices
    (the annotator's live channel-toggle buttons). `pct_low`/`pct_high` are
    the "top/bottom cutoff" contrast control (which percentile of each
    channel's pixel intensities maps to black/white) -- adjustable live via
    the annotator's contrast controls, recomputed here each time rather than
    cached, since it's a per-view display choice, not a property of the data.
    """
    any_raw = next(iter(channel_planes.values()))[2]
    disp = np.zeros((*any_raw.shape, 3), dtype=np.float32)
    for c, (_color, rgb, raw) in channel_planes.items():
        if active is not None and c not in active:
            continue
        plane = _clahe_equalize(raw, pct_low=pct_low, pct_high=pct_high, use_clahe=use_clahe)
        disp += plane[..., np.newaxis] * rgb[np.newaxis, np.newaxis, :]
    return np.clip(disp, 0.0, 1.0)


def _resolve_channels(channels, run, paths_by_run):
    """`channels` can be a single spec applied to every run, or a
    `{run: spec}` dict for per-run overrides -- but a *spec itself* can also
    be a dict (`{channel_index: color}`, the pseudocolor composite form), so
    a bare `isinstance(channels, dict)` check can't tell the two apart. A
    dict is only treated as "per-run" if every one of its keys is an actual
    run name; otherwise it's treated as one composite spec shared by every
    run.
    """
    if isinstance(channels, dict) and set(channels).issubset(paths_by_run):
        return channels[run]
    return channels


def _pick_level(shapes_by_level, target_max_dim=2048):
    """Given `{level_key: (C, H, W)}`, pick the lowest-resolution level whose
    largest spatial dimension is still >= target_max_dim (so annotation has
    enough detail), falling back to the smallest level available if every
    level is already smaller than that. Mirrors why MERFISH's own canvas
    uses a DOWNSAMPLE factor -- interactive pan/zoom needs a canvas that
    comfortably fits in memory and redraws fast, not full native resolution.
    """
    best_key, best_max = None, None
    for key, shape in shapes_by_level.items():
        max_dim = max(shape[-2], shape[-1])
        if max_dim >= target_max_dim and (best_max is None or max_dim < best_max):
            best_key, best_max = key, max_dim
    if best_key is not None:
        return best_key
    # every level is smaller than target -- use the largest one available
    return max(shapes_by_level, key=lambda k: max(shapes_by_level[k][-2], shapes_by_level[k][-1]))


def ome_tiff_canvas_provider(paths_by_run, channels=0, target_max_dim=2048):
    """Build a `canvas_provider` callable for `MultiRegionAnnotator` that
    reads pyramidal OME-TIFF files via `tifffile`'s zarr bridge.

    `paths_by_run`: `{run: path_to_ome_tiff}`.
    `channels`: what to display -- an int (single channel, grayscale), a
    list of up to 3 ints (direct R/G/B mapping), or a `{channel_index:
    color}` dict for a real multi-channel pseudocolor composite (any number
    of channels, each an arbitrary color -- see `_channels_to_rgb`). Any of
    these can also be wrapped in a `{run: spec}` dict for per-run overrides
    (disambiguated from the composite-dict form by checking whether the
    dict's keys are actual run names).
    `target_max_dim`: see `_pick_level`.
    """
    import tifffile

    def _provider(run):
        path = paths_by_run[run]
        ch = _resolve_channels(channels, run, paths_by_run)

        tif = tifffile.TiffFile(str(path))
        store = tif.aszarr()
        z = zarr.open(store, mode='r')

        # A pyramidal OME-TIFF's zarr bridge exposes one array per series/level;
        # tifffile names singleton/multi-level arrays "0", "1", ... in
        # decreasing resolution, matching what IHC's own reader does
        # (`zarr.open(tif.aszarr(), mode="r")["0"]`).
        if hasattr(z, 'array_keys'):
            shapes = {k: z[k].shape for k in z.array_keys()}
            level = _pick_level(shapes, target_max_dim)
            arr = z[level]
        else:
            arr = z  # a plain (non-pyramidal) array

        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]  # treat as single-channel (1, H, W)

        spec = _normalize_channel_spec(ch)
        channel_planes = _compute_channel_planes(arr, spec)
        disp_rgb = composite_channel_planes(channel_planes)
        H, W = disp_rgb.shape[0], disp_rgb.shape[1]
        return {'W': W, 'H': H, 'disp_rgb': disp_rgb, 'channel_planes': channel_planes}

    return _provider


def ome_zarr_canvas_provider(paths_by_run, channels=0, target_max_dim=2048):
    """Build a `canvas_provider` callable for `MultiRegionAnnotator` that
    reads OME-NGFF v0.4 multiscale zarr stores (the format IHC's own
    stitching pipeline writes -- `root.attrs['multiscales']` with numbered
    pyramid-level datasets).

    `paths_by_run`: `{run: path_to_ome_zarr}`.
    `channels`, `target_max_dim`: see `ome_tiff_canvas_provider`.
    """

    def _provider(run):
        path = paths_by_run[run]
        ch = _resolve_channels(channels, run, paths_by_run)

        root = zarr.open_group(str(path), mode='r')
        multiscales = root.attrs['multiscales'][0]
        levels = [d['path'] for d in multiscales['datasets']]
        shapes = {lvl: root[lvl].shape for lvl in levels}
        level = _pick_level(shapes, target_max_dim)
        arr = np.asarray(root[level])

        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        spec = _normalize_channel_spec(ch)
        channel_planes = _compute_channel_planes(arr, spec)
        disp_rgb = composite_channel_planes(channel_planes)
        H, W = disp_rgb.shape[0], disp_rgb.shape[1]
        return {'W': W, 'H': H, 'disp_rgb': disp_rgb, 'channel_planes': channel_planes}

    return _provider
