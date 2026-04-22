"""Project file format: .gpj (= zip file)

Contents:
- manifest.json       : metadata (version, app_version, created, modified, title, authors)
- data.csv            : the raw data table (Unicode UTF-8 with BOM for Excel compatibility)
- settings.json       : chart config dict + global export settings
- analyses.json       : list of AnalysisResult.to_dict() outputs
- fits.json           : list of FitResult dicts (curve_models) if any
- preview.png         : thumbnail (optional, for future file explorer)

Also provides PDF report generation and a simple unsaved-changes tracker.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_FORMAT_VERSION = 1
APP_VERSION = "0.5.0"


# ------------------------------------------------------------------
# JSON serialization helpers
# ------------------------------------------------------------------

def _json_default(obj):
    """Handle numpy types, np.nan, pd.Timestamp, dataclasses.

    Used as default= in json.dumps. Anything returnable as a JSON-compatible
    value should be returned here; the top-level sanitize step below handles
    float NaN/Inf inside containers before json sees them.
    """
    # numpy scalar → python scalar
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        fv = float(obj)
        if math.isnan(fv):
            return {'__special__': 'nan'}
        if math.isinf(fv):
            return {'__special__': 'inf' if fv > 0 else '-inf'}
        return fv
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return {
            '__ndarray__': True,
            'dtype': str(obj.dtype),
            'shape': list(obj.shape),
            'data': _sanitize(obj.tolist()),
        }
    # pandas timestamps / datetimes
    if isinstance(obj, (pd.Timestamp, datetime)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, pd.Timedelta):
        return obj.isoformat()
    # dataclass (not already handled)
    if is_dataclass(obj) and not isinstance(obj, type):
        return _sanitize(asdict(obj))
    # fall back to str so we never explode
    try:
        return str(obj)
    except Exception:
        return None


def _sanitize(obj: Any) -> Any:
    """Recursively walk a value and convert numpy/NaN/Inf/arrays into
    JSON-safe equivalents, marking NaN/Inf with a sentinel dict so the
    reverse path can round-trip them back.
    """
    # primitives
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int,)) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return {'__special__': 'nan'}
        if math.isinf(obj):
            return {'__special__': 'inf' if obj > 0 else '-inf'}
        return obj
    # numpy scalars
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _sanitize(float(obj))
    if isinstance(obj, np.bool_):
        return bool(obj)
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return {
            '__ndarray__': True,
            'dtype': str(obj.dtype),
            'shape': list(obj.shape),
            'data': _sanitize(obj.tolist()),
        }
    # pandas structures
    if isinstance(obj, pd.DataFrame):
        return {
            '__dataframe__': True,
            'payload': json.loads(obj.to_json(orient='split', default_handler=str)),
        }
    if isinstance(obj, pd.Series):
        return {
            '__series__': True,
            'name': obj.name,
            'index': _sanitize(list(obj.index)),
            'data': _sanitize(list(obj.values)),
        }
    if isinstance(obj, (pd.Timestamp, datetime)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    # mappings / iterables
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(v) for v in obj]
    # dataclass
    if is_dataclass(obj) and not isinstance(obj, type):
        return _sanitize(asdict(obj))
    # unknown — fall through to default handler via str
    try:
        return str(obj)
    except Exception:
        return None


def _restore(obj: Any) -> Any:
    """Reverse of _sanitize: turn sentinel dicts back into NaN/Inf/ndarray."""
    if isinstance(obj, dict):
        if obj.get('__special__') == 'nan':
            return float('nan')
        if obj.get('__special__') == 'inf':
            return float('inf')
        if obj.get('__special__') == '-inf':
            return float('-inf')
        if obj.get('__ndarray__'):
            data = _restore(obj.get('data', []))
            try:
                arr = np.asarray(data, dtype=obj.get('dtype'))
            except (TypeError, ValueError):
                arr = np.asarray(data)
            shape = obj.get('shape')
            if shape:
                try:
                    arr = arr.reshape(shape)
                except ValueError:
                    pass
            return arr
        if obj.get('__dataframe__'):
            payload = obj.get('payload', {})
            try:
                return pd.DataFrame(
                    data=payload.get('data', []),
                    index=payload.get('index'),
                    columns=payload.get('columns'),
                )
            except Exception:
                return pd.DataFrame()
        if obj.get('__series__'):
            return pd.Series(
                data=_restore(obj.get('data', [])),
                index=_restore(obj.get('index', [])),
                name=obj.get('name'),
            )
        return {k: _restore(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore(v) for v in obj]
    return obj


def _json_dumps(obj: Any) -> str:
    """Dump with sanitize first then default fallback; always valid JSON."""
    return json.dumps(_sanitize(obj), ensure_ascii=False, indent=2,
                      default=_json_default)


def _json_loads(text: str) -> Any:
    return _restore(json.loads(text))


# ------------------------------------------------------------------
# FitResult (de)serialization
# ------------------------------------------------------------------

_FIT_FIELDS_KEEP = (
    'model_name', 'equation',
    'params', 'param_se', 'param_ci_95',
    'r_squared', 'adjusted_r2', 'rmse', 'aicc', 'bic',
    'n', 'converged', 'message', 'derived',
)


def _serialize_fit_result(fit) -> dict:
    """curve_models.FitResult → JSON-safe dict.

    Skips residuals/predicted_y/covariance (large and not needed for restore).
    Accepts either a FitResult instance or a plain dict that already contains
    the same keys (for forward-compatibility with code that pre-serializes).
    """
    if isinstance(fit, dict):
        src = fit
        getter = src.get
    else:
        src = fit
        def getter(key, default=None):
            return getattr(src, key, default)

    out: dict = {}
    for key in _FIT_FIELDS_KEEP:
        val = getter(key, None)
        if val is None and key in ('params', 'param_se', 'param_ci_95', 'derived'):
            val = {}
        out[key] = val
    return _sanitize(out)


def _deserialize_fit_result(d: dict) -> dict:
    """Reverse. Returns a plain dict — actual FitResult instantiation is the
    caller's responsibility (curve_models is not imported here).
    """
    restored = _restore(d) if isinstance(d, dict) else {}
    # ensure all expected fields exist
    for key in _FIT_FIELDS_KEEP:
        restored.setdefault(key, {} if key in ('params', 'param_se', 'param_ci_95', 'derived') else None)
    return restored


# ------------------------------------------------------------------
# DataFrame <-> CSV (UTF-8 BOM)
# ------------------------------------------------------------------

def _df_to_csv_bytes(df: pd.DataFrame | None) -> bytes:
    """Serialize a DataFrame to UTF-8-BOM CSV bytes. Empty df → empty CSV."""
    if df is None or (hasattr(df, 'empty') and df.empty and len(df.columns) == 0):
        return '﻿'.encode('utf-8')
    buf = io.StringIO()
    buf.write('﻿')  # BOM for Excel compatibility
    try:
        df.to_csv(buf, index=True, encoding='utf-8')
    except Exception:
        df.to_csv(buf, index=True)
    return buf.getvalue().encode('utf-8')


def _csv_bytes_to_df(raw: bytes) -> pd.DataFrame | None:
    if raw is None or len(raw) == 0:
        return None
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = raw.decode('utf-8', errors='replace')
    if not text.strip():
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(text), index_col=0)
    except Exception:
        try:
            return pd.read_csv(io.StringIO(text))
        except Exception:
            return pd.DataFrame()


# ------------------------------------------------------------------
# ProjectFile
# ------------------------------------------------------------------

@dataclass
class ProjectFile:
    title: str = "Untitled Project"
    authors: str = ""
    notes: str = ""
    table_type: str = "column"        # data_model.DataTableType.value
    data: pd.DataFrame | None = None
    chart_type: str = "heatmap"       # which chart tab was active
    chart_config: dict = field(default_factory=dict)
    global_config: dict = field(default_factory=dict)  # journal preset, font, DPI, etc.
    analyses: list = field(default_factory=list)       # list of dicts (AnalysisResult.to_dict)
    fits: list = field(default_factory=list)           # list of FitResult dicts
    preview_png_bytes: bytes | None = None

    # ---------- save ----------
    def save(self, path: str | Path) -> None:
        """Write .gpj file (zip archive)."""
        path = Path(path)
        now_iso = datetime.now().isoformat(timespec='seconds')
        manifest = {
            'format_version': PROJECT_FORMAT_VERSION,
            'app_version': APP_VERSION,
            'created': now_iso,
            'modified': now_iso,
            'title': self.title,
            'authors': self.authors,
            'notes': self.notes,
            'table_type': self.table_type,
            'chart_type': self.chart_type,
        }

        settings = {
            'chart_type': self.chart_type,
            'chart_config': self.chart_config,
            'global_config': self.global_config,
        }

        # analyses: each may be an AnalysisResult (with to_dict) or already a dict
        analyses_payload = []
        for a in self.analyses:
            if hasattr(a, 'to_dict') and callable(a.to_dict):
                try:
                    analyses_payload.append(a.to_dict())
                    continue
                except Exception:
                    pass
            if isinstance(a, dict):
                analyses_payload.append(a)
            else:
                analyses_payload.append({'analysis_name': str(a)})

        fits_payload = [_serialize_fit_result(f) for f in self.fits]

        data_bytes = _df_to_csv_bytes(self.data)

        # write the archive
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('manifest.json', _json_dumps(manifest))
            zf.writestr('data.csv', data_bytes)
            zf.writestr('settings.json', _json_dumps(settings))
            zf.writestr('analyses.json', _json_dumps(analyses_payload))
            zf.writestr('fits.json', _json_dumps(fits_payload))
            if self.preview_png_bytes:
                zf.writestr('preview.png', self.preview_png_bytes)

    # ---------- load ----------
    @classmethod
    def load(cls, path: str | Path) -> 'ProjectFile':
        """Read .gpj file. Raises ValueError (Korean message) on format mismatch."""
        path = Path(path)
        if not path.exists():
            raise ValueError(f"프로젝트 파일을 찾을 수 없습니다: {path}")

        try:
            zf = zipfile.ZipFile(path, 'r')
        except zipfile.BadZipFile:
            raise ValueError("유효한 프로젝트 파일(.gpj)이 아닙니다. 파일이 손상되었거나 형식이 잘못되었습니다.")

        with zf:
            names = set(zf.namelist())
            if 'manifest.json' not in names:
                raise ValueError("manifest.json 이 없는 프로젝트 파일입니다. .gpj 형식이 아닙니다.")

            try:
                manifest = _json_loads(zf.read('manifest.json').decode('utf-8'))
            except Exception as e:
                raise ValueError(f"manifest.json 을 읽을 수 없습니다: {e}")

            fmt = manifest.get('format_version', 0)
            if not isinstance(fmt, int):
                raise ValueError(f"format_version 값이 올바르지 않습니다: {fmt!r}")
            if fmt > PROJECT_FORMAT_VERSION:
                raise ValueError(
                    f"더 새 버전의 프로젝트 파일입니다 (format_version={fmt}). "
                    f"현재 앱 버전({APP_VERSION})에서는 열 수 없습니다. 앱을 업데이트해 주세요."
                )
            if fmt < 1:
                raise ValueError(
                    f"지원하지 않는 프로젝트 파일 버전입니다 (format_version={fmt})."
                )

            settings = {}
            if 'settings.json' in names:
                try:
                    settings = _json_loads(zf.read('settings.json').decode('utf-8')) or {}
                except Exception:
                    settings = {}

            analyses: list = []
            if 'analyses.json' in names:
                try:
                    analyses = _json_loads(zf.read('analyses.json').decode('utf-8')) or []
                except Exception:
                    analyses = []

            fits: list = []
            if 'fits.json' in names:
                try:
                    raw_fits = _json_loads(zf.read('fits.json').decode('utf-8')) or []
                    fits = [_deserialize_fit_result(f) for f in raw_fits]
                except Exception:
                    fits = []

            data_df: pd.DataFrame | None = None
            if 'data.csv' in names:
                try:
                    data_df = _csv_bytes_to_df(zf.read('data.csv'))
                except Exception:
                    data_df = None

            preview_bytes: bytes | None = None
            if 'preview.png' in names:
                try:
                    preview_bytes = zf.read('preview.png')
                except Exception:
                    preview_bytes = None

        return cls(
            title=manifest.get('title', 'Untitled Project'),
            authors=manifest.get('authors', ''),
            notes=manifest.get('notes', ''),
            table_type=manifest.get('table_type', 'column'),
            data=data_df,
            chart_type=settings.get('chart_type', manifest.get('chart_type', 'heatmap')),
            chart_config=settings.get('chart_config', {}) or {},
            global_config=settings.get('global_config', {}) or {},
            analyses=analyses if isinstance(analyses, list) else [],
            fits=fits if isinstance(fits, list) else [],
            preview_png_bytes=preview_bytes,
        )


# ------------------------------------------------------------------
# PDF Report generator
# ------------------------------------------------------------------

_A4_PORTRAIT = (8.27, 11.69)   # inches
_KOREAN_FONT_PREF = ['Malgun Gothic', 'Noto Sans CJK KR', 'NanumGothic',
                     'AppleGothic', 'DejaVu Sans']


def _wrap_text_lines(text: str, max_chars: int = 95) -> list[str]:
    """Wrap long lines (preserving pre-existing newlines) to fit the page."""
    out: list[str] = []
    for raw in text.split('\n'):
        if len(raw) <= max_chars:
            out.append(raw)
            continue
        # break long lines at whitespace when possible, else hard-cut
        while len(raw) > max_chars:
            cut = raw.rfind(' ', 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(raw[:cut])
            raw = raw[cut:].lstrip()
        if raw:
            out.append(raw)
    return out


def _paginate(lines: list[str], lines_per_page: int) -> list[list[str]]:
    if not lines:
        return []
    return [lines[i:i + lines_per_page] for i in range(0, len(lines), lines_per_page)]


_KOREAN_MONO_PREF = ['D2Coding', 'NanumGothicCoding',
                     'Malgun Gothic', 'Nanum Gothic', 'NanumGothic',
                     'AppleGothic', 'Noto Sans CJK KR']


def _set_korean_font():
    """Apply Korean-capable font for both sans-serif and monospace.

    Returns (prev_family, prev_mono) to restore later.
    """
    import matplotlib
    prev_family = list(matplotlib.rcParams.get('font.family', ['sans-serif']))
    prev_mono = list(matplotlib.rcParams.get('font.monospace', []))
    try:
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        sans_pick = next((n for n in _KOREAN_FONT_PREF if n in available), None)
        mono_pick = next((n for n in _KOREAN_MONO_PREF if n in available), None)
        if sans_pick:
            matplotlib.rcParams['font.family'] = sans_pick
        if mono_pick:
            # monospace 별칭이 한글 폰트로 먼저 해석되도록 체인 맨 앞에 삽입
            matplotlib.rcParams['font.monospace'] = [mono_pick] + [
                m for m in prev_mono if m != mono_pick]
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass
    return prev_family, prev_mono


def _restore_font(prev):
    import matplotlib
    try:
        prev_family, prev_mono = prev if isinstance(prev, tuple) else (prev, None)
        matplotlib.rcParams['font.family'] = prev_family
        if prev_mono:
            matplotlib.rcParams['font.monospace'] = prev_mono
    except Exception:
        pass


def _cover_page(pdf, title: str, authors: str, date_str: str, notes: str = ''):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=_A4_PORTRAIT)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # three aesthetic horizontal rules
    for y, lw in [(0.82, 1.2), (0.52, 0.8), (0.18, 1.2)]:
        ax.plot([0.1, 0.9], [y, y], color='#333333', linewidth=lw)

    ax.text(0.5, 0.68, title, fontsize=28, fontweight='bold',
            ha='center', va='center', wrap=True)
    if authors:
        ax.text(0.5, 0.58, authors, fontsize=14, ha='center', va='center',
                color='#222222')
    ax.text(0.5, 0.46, date_str, fontsize=11, ha='center', va='center',
            color='#666666')

    if notes:
        ax.text(0.5, 0.32, notes, fontsize=10, ha='center', va='top',
                color='#333333', wrap=True)

    pdf.savefig(fig)
    plt.close(fig)


def _figure_page(pdf, src_fig_input, caption: str):
    """Render a single figure on its own A4 page with caption below.

    Strategy: capture the source figure to PNG in memory, then place it
    on a fresh A4-sized host figure so we don't mutate the caller's fig.
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import matplotlib.image as mpimg

    # Grab pixels from source
    if isinstance(src_fig_input, Figure):
        buf = io.BytesIO()
        src_fig_input.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        buf.seek(0)
        img = mpimg.imread(buf, format='png')
    elif isinstance(src_fig_input, (bytes, bytearray)):
        img = mpimg.imread(io.BytesIO(bytes(src_fig_input)), format='png')
    elif isinstance(src_fig_input, (str, Path)) and Path(str(src_fig_input)).is_file():
        img = mpimg.imread(str(src_fig_input))
    elif hasattr(src_fig_input, 'read'):
        img = mpimg.imread(src_fig_input, format='png')
    else:
        raise TypeError(f"Unsupported figure input: {type(src_fig_input).__name__}")

    fig = plt.figure(figsize=_A4_PORTRAIT)
    # image axis occupies top ~80% with 0.5 in margin
    a4_w, a4_h = _A4_PORTRAIT
    margin_x = 0.5 / a4_w
    top_margin = 0.5 / a4_h
    caption_h = 1.2 / a4_h
    img_ax = fig.add_axes([margin_x, caption_h + 0.05,
                           1 - 2 * margin_x, 1 - caption_h - top_margin - 0.05])
    img_ax.imshow(img)
    img_ax.axis('off')

    cap_ax = fig.add_axes([margin_x, 0.02, 1 - 2 * margin_x, caption_h])
    cap_ax.axis('off')
    cap_ax.text(0.5, 0.5, caption, ha='center', va='center',
                fontsize=11, wrap=True)

    pdf.savefig(fig)
    plt.close(fig)


def _stats_pages(pdf, analyses: list):
    """Render each AnalysisResult.to_text() block, paginating as needed."""
    import matplotlib.pyplot as plt

    lines_per_page = 62   # conservative for 9pt mono on A4 with 0.5in margin

    for idx, res in enumerate(analyses, start=1):
        # obtain text
        if hasattr(res, 'to_text') and callable(res.to_text):
            try:
                text = res.to_text()
            except Exception as e:
                text = f"[분석 결과 렌더 실패] {e}"
        elif isinstance(res, dict):
            parts = [f"분석: {res.get('analysis_name', '?')}"]
            if 'summary' in res and isinstance(res['summary'], dict):
                parts.append('')
                parts.append('[ 요약 ]')
                for k, v in res['summary'].items():
                    parts.append(f"  {k}: {v}")
            if res.get('notes'):
                parts.append('')
                parts.append('[ 비고 ]')
                for n in res['notes']:
                    parts.append(f"  - {n}")
            text = '\n'.join(parts)
        else:
            text = str(res)

        lines = _wrap_text_lines(text, max_chars=95)
        pages = _paginate(lines, lines_per_page) or [['(비어 있음)']]
        total = len(pages)
        for page_i, page_lines in enumerate(pages, start=1):
            fig = plt.figure(figsize=_A4_PORTRAIT)
            ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
            ax.axis('off')
            header = f"통계 분석 #{idx}"
            if total > 1:
                header += f"  (페이지 {page_i}/{total})"
            ax.text(0.0, 1.0, header, fontsize=12, fontweight='bold',
                    va='top', ha='left')
            body = '\n'.join(page_lines)
            ax.text(0.0, 0.96, body, family='monospace', fontsize=9,
                    verticalalignment='top', ha='left')
            pdf.savefig(fig)
            plt.close(fig)


def _fits_pages(pdf, fits: list):
    """Compact summary page(s) for curve fits."""
    import matplotlib.pyplot as plt

    lines: list[str] = []
    for i, fit in enumerate(fits, start=1):
        if isinstance(fit, dict):
            get = fit.get
        else:
            def get(k, default=None, _f=fit):
                return getattr(_f, k, default)
        lines.append('=' * 72)
        lines.append(f"  곡선 피팅 #{i}: {get('model_name', '?')}")
        lines.append('=' * 72)
        eq = get('equation', '')
        if eq:
            lines.append(f"  식     : {eq}")
        lines.append(f"  R^2    : {get('r_squared', float('nan'))}")
        lines.append(f"  AICc   : {get('aicc', float('nan'))}")
        lines.append(f"  BIC    : {get('bic', float('nan'))}")
        lines.append(f"  n      : {get('n', 0)}")
        lines.append(f"  수렴   : {get('converged', False)}")
        params = get('params', {}) or {}
        if params:
            lines.append('')
            lines.append('  [ 파라미터 ]')
            for k, v in params.items():
                lines.append(f"    {k} = {v}")
        derived = get('derived', {}) or {}
        if derived:
            lines.append('')
            lines.append('  [ 파생 지표 ]')
            for k, v in derived.items():
                lines.append(f"    {k} = {v}")
        lines.append('')

    if not lines:
        return
    wrapped = _wrap_text_lines('\n'.join(lines), max_chars=95)
    pages = _paginate(wrapped, 62)
    total = len(pages)
    for page_i, page_lines in enumerate(pages, start=1):
        fig = plt.figure(figsize=_A4_PORTRAIT)
        ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
        ax.axis('off')
        header = '곡선 피팅 결과'
        if total > 1:
            header += f"  (페이지 {page_i}/{total})"
        ax.text(0.0, 1.0, header, fontsize=12, fontweight='bold',
                va='top', ha='left')
        ax.text(0.0, 0.96, '\n'.join(page_lines), family='monospace',
                fontsize=9, verticalalignment='top', ha='left')
        pdf.savefig(fig)
        plt.close(fig)


def _data_page(pdf, df: pd.DataFrame):
    """Render a data table. Truncates to first 50 rows if large."""
    import matplotlib.pyplot as plt

    if df is None:
        return
    try:
        n_rows = len(df)
    except TypeError:
        return
    if n_rows == 0 and len(df.columns) == 0:
        return

    truncated = False
    display_df = df
    if n_rows > 50:
        display_df = df.head(50)
        truncated = True

    # if table is extremely wide, too many columns to render, skip with note
    MAX_COLS = 12
    skipped_cols = max(0, len(display_df.columns) - MAX_COLS)
    if skipped_cols > 0:
        display_df = display_df.iloc[:, :MAX_COLS]

    fig = plt.figure(figsize=_A4_PORTRAIT)
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.86])
    ax.axis('off')
    ax.text(0.0, 1.0, '원본 데이터', fontsize=12, fontweight='bold',
            va='top', ha='left')

    # stringify for display
    try:
        cell_text = display_df.astype(object).where(pd.notna(display_df), '').values.tolist()
    except Exception:
        cell_text = display_df.values.tolist()
    col_labels = [str(c) for c in display_df.columns]
    row_labels = [str(i) for i in display_df.index]

    if not cell_text:
        ax.text(0.0, 0.9, '(비어 있음)', fontsize=10, va='top', ha='left')
    else:
        tbl = ax.table(cellText=cell_text,
                       colLabels=col_labels,
                       rowLabels=row_labels,
                       loc='upper left',
                       cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.0, 1.2)

    note_bits = []
    if truncated:
        note_bits.append(f"... 외 {n_rows - 50}개 행 생략")
    if skipped_cols > 0:
        note_bits.append(f"... 외 {skipped_cols}개 열 생략")
    if note_bits:
        ax.text(0.0, 0.02, ' / '.join(note_bits), fontsize=9,
                color='#666666', ha='left', va='bottom')

    pdf.savefig(fig)
    plt.close(fig)


def generate_report_pdf(
    fig_paths_or_buffers: list,
    analyses: list,
    fits: list,
    title: str,
    authors: str = "",
    data_df: pd.DataFrame | None = None,
    notes: str = "",
    output_path: str | Path = "report.pdf",
) -> None:
    """Multi-page PDF report.

    Page 1        : Cover (title, authors, date)
    Page 2+       : Each figure at publication size with caption
    Following     : Statistical analysis blocks (monospace)
    Curve fits    : Combined summary
    Last page     : Raw data table (first 50 rows if large)
    """
    from matplotlib.backends.backend_pdf import PdfPages

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prev_font = _set_korean_font()
    try:
        with PdfPages(str(output_path)) as pdf:
            # cover
            _cover_page(
                pdf,
                title=title or 'Untitled Report',
                authors=authors or '',
                date_str=datetime.now().strftime('%Y-%m-%d'),
                notes=notes or '',
            )

            # figures
            for i, item in enumerate(fig_paths_or_buffers or [], start=1):
                caption = f"Figure {i}"
                try:
                    _figure_page(pdf, item, caption)
                except Exception as e:
                    # render error-placeholder page so report still completes
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=_A4_PORTRAIT)
                    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                    ax.axis('off')
                    ax.text(0.5, 0.5,
                            f"{caption}\n\n[figure render failed]\n{e}",
                            ha='center', va='center', fontsize=12, color='red')
                    pdf.savefig(fig)
                    plt.close(fig)

            # stats
            if analyses:
                _stats_pages(pdf, analyses)

            # fits
            if fits:
                _fits_pages(pdf, fits)

            # data table
            if data_df is not None:
                try:
                    _data_page(pdf, data_df)
                except Exception:
                    # if rendering a table fails (too large, weird types), just skip with note
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=_A4_PORTRAIT)
                    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                    ax.axis('off')
                    ax.text(0.5, 0.5,
                            '원본 데이터가 너무 커서 PDF에 포함되지 않았습니다.',
                            ha='center', va='center', fontsize=12, color='#666666')
                    pdf.savefig(fig)
                    plt.close(fig)

            # PDF metadata
            d = pdf.infodict()
            d['Title'] = title or 'Report'
            if authors:
                d['Author'] = authors
            d['CreationDate'] = datetime.now()
            d['ModDate'] = datetime.now()
    finally:
        _restore_font(prev_font)


# ------------------------------------------------------------------
# Auto-save tracker
# ------------------------------------------------------------------

class DirtyTracker:
    """Simple observable for 'unsaved changes' state.

    Optional callback receives the new dirty state whenever it changes.
    """

    def __init__(self, on_change=None):
        self._dirty = False
        self._on_change = on_change

    def mark_dirty(self):
        if not self._dirty:
            self._dirty = True
            self._fire()

    def mark_clean(self):
        if self._dirty:
            self._dirty = False
            self._fire()

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def set_callback(self, fn):
        self._on_change = fn

    def _fire(self):
        if self._on_change is not None:
            try:
                self._on_change(self._dirty)
            except Exception:
                pass


__all__ = [
    'PROJECT_FORMAT_VERSION',
    'APP_VERSION',
    'ProjectFile',
    'generate_report_pdf',
    'DirtyTracker',
    '_json_default',
    '_serialize_fit_result',
    '_deserialize_fit_result',
]
