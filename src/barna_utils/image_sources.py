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

Canvas dicts returned here also include:

- `channel_planes` -- each channel's *raw* pixel plane + assigned color
  (see `_compute_channel_planes`), so `MultiRegionAnnotator`'s live
  per-channel toggle buttons and contrast/CLAHE controls can cheaply
  re-blend/re-normalize a subset of channels via `composite_channel_planes`
  without ever re-reading from disk. Normalized results are memoized per
  `(pct_low, pct_high, use_clahe)` combination actually used, so repeated
  redraws at unchanged settings are free.
- `channel_names` -- `{channel_index: display_name}`, using each channel's
  *real* embedded name from the file's own metadata whenever present
  (falling back to excitation wavelength, then a plain positional label --
  see `_channel_display_name`). Confirmed empirically that IHC's raw TMA
  scan has no real channel names (OME-XML `Channel Name=""` throughout,
  only `ExcitationWavelength` populated), but IHC's *stitched/merged*
  outputs (both the OME-Zarr and the derived pyramid OME-TIFF) do carry
  real marker names end-to-end -- the case that actually matters for
  day-to-day annotation.
- `hires_fetcher` -- a callable `(x0, y0, x1, y1, target_max_dim) ->
  {'channel_planes', 'extent_overview_px', 'level'}` (see
  `_make_hires_fetcher`) that fetches a small on-demand crop from a much
  finer pyramid level for just the currently-visible region, in the
  overview canvas's own pixel coordinate space. This backs
  `MultiRegionAnnotator`'s debounced "always sharp while zoomed in"
  behavior -- the low-res overview stays the default (fast to pan/zoom),
  and only the actual visible crop gets replaced with a higher-resolution
  fetch once panning/zooming settles, rather than ever loading a whole
  gigapixel-scale level into memory at once.

CLAHE (`use_clahe` in `composite_channel_planes`/`_clahe_equalize`) defaults
to **off**: it was only ever needed for MERFISH's own poor-quality DAPI
mosaics, not real fluorescence imaging, and it's the expensive step in the
normalization pipeline -- so it's opt-in via the annotator's CLAHE toggle,
not paid for by default.

The MERSCOPE DAPI-mosaic provider is NOT here -- it stays in MERFISH's own
notebook/scripts, since building it requires MERSCOPE-specific mosaic-tile
assembly logic (`dapi_utils.py`) that has no equivalent generic meaning.
Any canvas provider is just a plain callable `(run) -> dict`, so MERFISH's
existing `GLOBAL_CANVAS`-based approach can be wrapped in a one-line lambda
and passed to the generalized `MultiRegionAnnotator` unchanged.
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


def _normalize_channel_spec(channels, channel_info=None):
    """Convert any of the three accepted `channels` forms into the one
    canonical shape everything downstream (compositing, live toggling)
    actually works with: `{channel_index: color}`.

    - a single int -> that channel's real color from `channel_info` if
      known, else 'white' (grayscale).
    - a list of ints -> each channel's real color from `channel_info` if
      known, else a distinct default color per position.
    - a dict -> returned as-is (already canonical; explicit colors always
      win over metadata-derived ones).

    `channel_info`, if given, is `{channel_index: {'name':..., 'color':...}}`
    (see `_extract_ome_tiff_channel_info`/`_extract_ome_zarr_channel_info`)
    -- real per-channel colors read from the file's own metadata (e.g.
    IHC's stitched OME-Zarr embeds a genuine display color per marker),
    used as smarter defaults than an arbitrary fixed color list whenever a
    channel's color isn't explicitly specified.
    """
    channel_info = channel_info or {}

    def _default_color(idx, position):
        info_color = channel_info.get(idx, {}).get('color')
        if info_color:
            return info_color
        return _DEFAULT_CHANNEL_COLORS[position % len(_DEFAULT_CHANNEL_COLORS)]

    if isinstance(channels, dict):
        return dict(channels)
    if isinstance(channels, (int, np.integer)):
        idx = int(channels)
        return {idx: _default_color(idx, 0) if channel_info else 'white'}
    return {int(c): _default_color(int(c), i) for i, c in enumerate(channels)}


def _channel_display_name(idx, channel_info):
    """Best available label for channel `idx`: its real embedded name if
    the file has one, else its excitation wavelength (still real metadata,
    just less specific), else a plain positional fallback. Never invents a
    marker name that isn't actually in the file -- if neither is present,
    say so plainly rather than guess."""
    info = (channel_info or {}).get(idx, {})
    if info.get('name'):
        return info['name']
    if info.get('wavelength_nm') is not None:
        return f'{info["wavelength_nm"]:g}nm'
    return f'Ch{idx}'


def _extract_ome_tiff_channel_info(tif):
    """Parse the real per-channel `Name`/`Color`/`ExcitationWavelength` out
    of an OME-TIFF's embedded OME-XML (proper XML parsing, not regex).
    Returns `{channel_index: {'name': str|None, 'color': str|None,
    'wavelength_nm': float|None}}`. A file with no OME-XML at all, or with
    every `Name` attribute empty (confirmed to happen in practice -- IHC's
    own raw TMA scan has this: SizeC=6 channels, every `Name=""`, only
    `ExcitationWavelength` populated), returns an empty-name/color dict
    per channel rather than failing -- callers fall back via
    `_channel_display_name`/`_normalize_channel_spec`."""
    xml = getattr(tif, 'ome_metadata', None)
    if not xml:
        return {}
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    info = {}
    for i, ch in enumerate(root.findall('.//ome:Channel', ns)):
        name = ch.get('Name') or None
        color_int = ch.get('Color')
        color = None
        if color_int is not None:
            # OME stores Color as a signed 32-bit RGBA integer.
            v = int(color_int) & 0xFFFFFFFF
            r, g, b = (v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF
            if (r, g, b) != (0, 0, 0):  # (0,0,0) is not a usable display color
                color = f'#{r:02x}{g:02x}{b:02x}'
        exc = ch.get('ExcitationWavelength')
        info[i] = {'name': name, 'color': color,
                   'wavelength_nm': float(exc) if exc is not None else None}
    return info


def _extract_ome_zarr_channel_info(root):
    """Read the real per-channel `label`/`color` out of an OME-NGFF zarr
    store's `omero.channels[]` metadata (confirmed present and populated in
    IHC's own stitched output -- real marker names like "Nuclei (DAPI)",
    "Ki-67 (Alexa Fluor 488)", each with a real hex display color). Returns
    the same shape as `_extract_ome_tiff_channel_info`. A store with no
    `omero` block (not every OME-Zarr writer includes one) returns `{}`."""
    channels = root.attrs.get('omero', {}).get('channels', [])
    info = {}
    for i, ch in enumerate(channels):
        color = ch.get('color')
        info[i] = {
            'name': ch.get('label') or None,
            'color': f'#{color}' if color else None,
            'wavelength_nm': None,
        }
    return info


def _compute_channel_planes(arr, channel_spec):
    """`arr` is (C, H, W); `channel_spec` is `{channel_index: color}`
    (see `_normalize_channel_spec`). Returns `{channel_index: (color,
    rgb_array, raw_plane, cache_dict)}` -- the *raw* (unprocessed) per-channel
    pixel data, read from disk exactly once here. Kept raw (not
    pre-normalized) specifically so contrast/cutoff adjustments and channel
    toggling (see `composite_channel_planes`) are cheap in-memory
    recomputations, never a re-read. `cache_dict` (initially empty, mutated
    in place by `composite_channel_planes`) memoizes the normalized result
    per `(pct_low, pct_high, use_clahe)` combination actually requested, so
    e.g. toggling a channel on/off costs nothing once it's been shown once
    at the current settings, and only a genuinely new contrast/CLAHE setting
    pays the real (CLAHE especially) computation cost.
    """
    import matplotlib.colors as mcolors
    planes = {}
    for c, color in channel_spec.items():
        raw = np.asarray(arr[c])
        rgb = np.array(mcolors.to_rgb(color), dtype=np.float32)
        planes[c] = (color, rgb, raw, {})
    return planes


def composite_channel_planes(channel_planes, active=None, pct_low=1, pct_high=99,
                             use_clahe=False):
    """Blend cached raw per-channel planes (see `_compute_channel_planes`)
    into a displayable RGB array, applying the percentile cutoff (and,
    only if requested, CLAHE) at blend time -- memoized per channel per
    `(pct_low, pct_high, use_clahe)` combination (see `_compute_channel_planes`'s
    `cache_dict`), so repeated calls with unchanged settings (e.g. toggling
    a *different* channel, or redrawing after a pan/zoom) never recompute
    normalization for channels whose settings didn't change.

    `active`, if given, restricts the blend to a subset of channel indices
    (the annotator's live channel-toggle buttons). `pct_low`/`pct_high` are
    the "top/bottom cutoff" contrast control. `use_clahe` defaults to False
    -- adaptive histogram equalization is the expensive step and was only
    ever needed for MERFISH's poor-quality DAPI; real fluorescence imaging
    (IHC's own use case) generally looks correct with plain percentile
    rescaling alone, so it's opt-in via the annotator's CLAHE toggle, not
    always paid for.
    """
    any_raw = next(iter(channel_planes.values()))[2]
    disp = np.zeros((*any_raw.shape, 3), dtype=np.float32)
    key = (pct_low, pct_high, use_clahe)
    for c, (_color, rgb, raw, cache) in channel_planes.items():
        if active is not None and c not in active:
            continue
        plane = cache.get(key)
        if plane is None:
            plane = _clahe_equalize(raw, pct_low=pct_low, pct_high=pct_high, use_clahe=use_clahe)
            cache[key] = plane
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


def _make_hires_fetcher(get_level_array, shapes_by_level, overview_level, spec):
    """Build the `hires_fetcher(x0, y0, x1, y1, target_max_dim)` closure
    stored in a canvas dict, used by `MultiRegionAnnotator`'s debounced
    "always sharp while zoomed in" behavior. `get_level_array(level_key)`
    returns the (C, H, W) zarr-backed array for a given pyramid level
    (opened lazily -- indexing it only reads the chunks actually touched,
    which is what makes fetching a small high-res crop out of a
    gigapixel-scale image fast). Coordinates in and out are always in the
    *overview* level's canvas-pixel space, so the caller never needs to
    know about the underlying pyramid's level scale factors.
    """
    overview_shape = shapes_by_level[overview_level]

    def _fetcher(x0, y0, x1, y1, target_max_dim=1024):
        x0, x1 = sorted((max(0, x0), min(overview_shape[-1], x1)))
        y0, y1 = sorted((max(0, y0), min(overview_shape[-2], y1)))
        if x1 <= x0 or y1 <= y0:
            return None

        # Pick the finest level whose crop of this same physical region
        # wouldn't exceed target_max_dim in either dimension -- the same
        # "just enough resolution, not more than needed" logic as
        # _pick_level, just evaluated for a sub-region instead of the whole
        # image so it can go much finer without the crop becoming huge.
        best_level, best_scale = overview_level, 1.0
        for level, shape in shapes_by_level.items():
            scale = shape[-1] / overview_shape[-1]  # this level's pixels per overview pixel
            crop_w, crop_h = (x1 - x0) * scale, (y1 - y0) * scale
            cur_scale = shapes_by_level[best_level][-1] / overview_shape[-1]
            if scale > cur_scale and max(crop_w, crop_h) <= target_max_dim:
                best_level, best_scale = level, scale

        arr = get_level_array(best_level)
        fy0, fy1 = int(y0 * best_scale), int(y1 * best_scale)
        fx0, fx1 = int(x0 * best_scale), int(x1 * best_scale)
        if arr.ndim == 2:
            crop = np.asarray(arr[fy0:fy1, fx0:fx1])[np.newaxis, ...]
        else:
            crop = np.asarray(arr[:, fy0:fy1, fx0:fx1])

        channel_planes = _compute_channel_planes(crop, spec)
        return {
            'channel_planes': channel_planes,
            'extent_overview_px': (x0, x1, y1, y0),  # matplotlib imshow extent order
            'level': best_level,
        }

    return _fetcher


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
        # (`zarr.open(tif.aszarr(), mode="r")["0"]`). `get_level_array` keeps
        # the lazy zarr handle (not a fully-read numpy array) per level, so
        # `hires_fetcher` can later index just a small region of a much
        # finer level without reading the whole thing.
        if hasattr(z, 'array_keys'):
            shapes = {k: z[k].shape for k in z.array_keys()}
            level = _pick_level(shapes, target_max_dim)
            get_level_array = lambda lvl: z[lvl]  # noqa: E731
        else:
            shapes = {'0': z.shape}
            level = '0'
            get_level_array = lambda lvl: z  # noqa: E731  (no finer level exists)

        arr = np.asarray(get_level_array(level))
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]  # treat as single-channel (1, H, W)

        channel_info = _extract_ome_tiff_channel_info(tif)
        spec = _normalize_channel_spec(ch, channel_info)
        channel_planes = _compute_channel_planes(arr, spec)
        channel_names = {c: _channel_display_name(c, channel_info) for c in spec}
        disp_rgb = composite_channel_planes(channel_planes)
        H, W = disp_rgb.shape[0], disp_rgb.shape[1]
        hires_fetcher = _make_hires_fetcher(get_level_array, shapes, level, spec)
        return {'W': W, 'H': H, 'disp_rgb': disp_rgb, 'channel_planes': channel_planes,
                'channel_names': channel_names, 'hires_fetcher': hires_fetcher}

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
        get_level_array = lambda lvl: root[lvl]  # noqa: E731  (lazy zarr handle, see ome_tiff_canvas_provider)
        arr = np.asarray(get_level_array(level))

        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        channel_info = _extract_ome_zarr_channel_info(root)
        spec = _normalize_channel_spec(ch, channel_info)
        channel_planes = _compute_channel_planes(arr, spec)
        channel_names = {c: _channel_display_name(c, channel_info) for c in spec}
        disp_rgb = composite_channel_planes(channel_planes)
        H, W = disp_rgb.shape[0], disp_rgb.shape[1]
        hires_fetcher = _make_hires_fetcher(get_level_array, shapes, level, spec)
        return {'W': W, 'H': H, 'disp_rgb': disp_rgb, 'channel_planes': channel_planes,
                'channel_names': channel_names, 'hires_fetcher': hires_fetcher}

    return _provider
