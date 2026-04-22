"""차트 렌더링: Heatmap, Boxplot, Bar, Violin, Scatter"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from stats_utils import (
    describe, pairwise_tests, linear_regression,
    build_bracket_positions, draw_brackets_xy,
)


def font_weight(is_bold):
    return 'bold' if is_bold else 'normal'


def font_style(is_italic):
    return 'italic' if is_italic else 'normal'


def apply_italic_to_label(label, italic_texts, use_bold=False):
    """라벨 내 특정 텍스트를 mathtext 이탤릭으로 감쌈"""
    result = label
    for t in italic_texts:
        if t in result:
            escaped = t.replace(' ', r'\ ')
            if use_bold:
                result = result.replace(t, f'$\\bf{{\\it{{{escaped}}}}}$')
            else:
                result = result.replace(t, f'$\\it{{{escaped}}}$')
    return result


# --- 공통: 축 스타일 적용 ---
def _style_axes(ax, cfg):
    """축 라벨·눈금 공통 스타일"""
    aw = font_weight(cfg['axis_bold'])
    as_ = font_style(cfg['axis_italic'])
    tw = font_weight(cfg['tick_bold'])
    ts = font_style(cfg['tick_italic'])

    ax.set_xlabel(ax.get_xlabel(), fontsize=cfg['axis_size'], fontweight=aw, fontstyle=as_)
    ax.set_ylabel(ax.get_ylabel(), fontsize=cfg['axis_size'], fontweight=aw, fontstyle=as_)

    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(cfg['tick_size'])
        lbl.set_fontweight(tw)
        lbl.set_fontstyle(ts)
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(cfg['tick_size'])
        lbl.set_fontweight(tw)
        lbl.set_fontstyle(ts)


def _apply_title(ax, cfg, italic_texts=None):
    title = cfg.get('title', '').strip()
    if not title:
        return
    if italic_texts:
        title = apply_italic_to_label(title, italic_texts, use_bold=cfg['title_bold'])
    subtitle = cfg.get('subtitle', '').strip()
    if subtitle:
        if italic_texts:
            subtitle = apply_italic_to_label(subtitle, italic_texts, use_bold=cfg['title_bold'])
        title = f"{title}\n({subtitle})"
    ax.set_title(
        title,
        fontsize=cfg['title_size'],
        fontweight=font_weight(cfg['title_bold']),
        fontstyle=font_style(cfg['title_italic']),
        pad=cfg.get('title_pad', 15),
    )


def _apply_grid(ax, cfg):
    if cfg.get('grid', True):
        ax.yaxis.grid(True, linestyle='--', alpha=0.6, linewidth=0.7)
        ax.set_axisbelow(True)
    else:
        ax.yaxis.grid(False)


def _apply_spines(ax, cfg):
    """축 선 (Prism 스타일: 위/오른쪽 제거)"""
    if cfg.get('despine', True):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    lw = cfg.get('spine_linewidth', 1.0)
    for s in ['left', 'bottom']:
        ax.spines[s].set_linewidth(lw)
    ax.tick_params(width=lw, length=4)


# -------------------------------------------------------------------
# Heatmap
# -------------------------------------------------------------------
def render_heatmap(df, cfg, fig_size_inches):
    fig, ax = plt.subplots(figsize=fig_size_inches)

    italic_texts = cfg.get('italic_texts', [])
    axis_bold = cfg['axis_bold']

    if italic_texts:
        new_columns = [apply_italic_to_label(c, italic_texts, axis_bold) for c in df.columns]
        new_index = [apply_italic_to_label(i, italic_texts, axis_bold) for i in df.index]
        df = df.copy()
        df.columns = new_columns
        df.index = new_index

    annot_kws = {
        'size': cfg['annot_size'],
        'weight': font_weight(cfg['annot_bold']),
        'style': font_style(cfg['annot_italic']),
    }

    sns.heatmap(
        df,
        annot=cfg.get('annot', True),
        fmt=cfg.get('fmt', '.0f'),
        cmap=cfg['cmap'],
        vmin=cfg['vmin'],
        vmax=cfg['vmax'],
        linewidths=cfg.get('linewidths', 0.5),
        linecolor=cfg.get('linecolor', 'white'),
        cbar_kws={'label': cfg['cbar_label'], 'shrink': 0.8},
        annot_kws=annot_kws,
        square=cfg.get('square', False),
        ax=ax,
    )

    plt.setp(ax.get_xticklabels(), rotation=cfg.get('xtick_rotation', 45), ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    _style_axes(ax, cfg)

    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel(
        cfg['cbar_label'],
        fontsize=cfg['cbar_size'],
        fontweight=font_weight(cfg.get('cbar_bold', False)),
    )
    cbar.ax.tick_params(labelsize=max(6, cfg['cbar_size'] - 2))

    _apply_title(ax, cfg, italic_texts)
    plt.tight_layout()
    return fig


# -------------------------------------------------------------------
# Boxplot
# -------------------------------------------------------------------
def render_boxplot(df, cfg, fig_size_inches):
    fig, ax = plt.subplots(figsize=fig_size_inches)

    order = _sort_columns(df, cfg.get('sort', '원본 순서'))
    italic_texts = cfg.get('italic_texts', [])

    palette_name = cfg.get('palette', 'Set2')
    colors = sns.color_palette(palette_name, len(order))

    data_arrays = [df[c].dropna().values for c in order]

    bp = ax.boxplot(
        data_arrays,
        patch_artist=True,
        widths=cfg.get('box_width', 0.6),
        notch=cfg.get('notch', False),
        showfliers=cfg.get('showfliers', True),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='gray', linewidth=1.0),
        capprops=dict(color='gray', linewidth=1.0),
        flierprops=dict(marker='o', markerfacecolor='white',
                        markeredgecolor='gray', markersize=5),
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(cfg.get('alpha', 0.85))
        patch.set_edgecolor('darkgray')
        patch.set_linewidth(1.2)

    # 개별 데이터 점
    if cfg.get('show_points', False):
        rng = np.random.default_rng(42)
        for i, arr in enumerate(data_arrays):
            x_jitter = rng.normal(loc=i + 1, scale=0.05, size=len(arr))
            ax.scatter(x_jitter, arr, s=cfg.get('point_size', 20),
                       color='black', alpha=0.5, zorder=3, edgecolor='none')

    # 평균 마커
    means = [float(np.nanmean(df[c])) for c in order]
    if cfg.get('show_mean', True):
        ax.scatter(range(1, len(order) + 1), means,
                   color='red', marker='D', s=40, zorder=5, label='Mean')

        if cfg.get('show_mean_label', True):
            for i, m in enumerate(means):
                ax.annotate(
                    f'{m:.1f}', (i + 1, m),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center',
                    fontsize=cfg['tick_size'],
                    fontweight=font_weight(cfg['tick_bold']),
                    fontstyle=font_style(cfg['tick_italic']),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.8),
                )

    x_labels = [apply_italic_to_label(l, italic_texts, cfg['tick_bold']) if italic_texts else l
                for l in order]
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(x_labels, rotation=cfg.get('xtick_rotation', 45), ha='right')

    if cfg.get('ymin') is not None and cfg.get('ymax') is not None:
        ax.set_ylim(cfg['ymin'], cfg['ymax'])

    ax.set_xlabel(cfg.get('xlabel', ''))
    ax.set_ylabel(cfg.get('ylabel', ''))
    _style_axes(ax, cfg)

    # 유의성 브래킷
    _draw_comparison_brackets(ax, df[order], cfg, means_tops_mode='box')

    if cfg.get('show_mean', True):
        ax.legend(loc='upper right', fontsize=max(6, cfg['tick_size'] - 1))

    _apply_grid(ax, cfg)
    _apply_spines(ax, cfg)
    _apply_title(ax, cfg, italic_texts)
    plt.tight_layout()
    return fig


# -------------------------------------------------------------------
# Bar chart
# -------------------------------------------------------------------
def render_barplot(df, cfg, fig_size_inches):
    fig, ax = plt.subplots(figsize=fig_size_inches)

    order = _sort_columns(df, cfg.get('sort', '원본 순서'))
    italic_texts = cfg.get('italic_texts', [])
    palette_name = cfg.get('palette', 'Set2')
    colors = sns.color_palette(palette_name, len(order))

    data_arrays = [df[c].dropna().values for c in order]
    means = [float(np.nanmean(arr)) if len(arr) else 0.0 for arr in data_arrays]

    err_mode = cfg.get('err_mode', 'SEM')  # SEM / SD / 95%CI / none
    errs = []
    for arr in data_arrays:
        d = describe(arr)
        if err_mode == 'SD':
            errs.append(d['std'])
        elif err_mode == '95%CI':
            errs.append(d['ci95'])
        elif err_mode == 'none':
            errs.append(0)
        else:
            errs.append(d['sem'])

    x_pos = np.arange(len(order))
    bars = ax.bar(
        x_pos, means,
        yerr=errs if err_mode != 'none' else None,
        capsize=cfg.get('capsize', 4),
        width=cfg.get('bar_width', 0.6),
        color=colors,
        edgecolor='black',
        linewidth=1.0,
        alpha=cfg.get('alpha', 0.9),
        error_kw=dict(elinewidth=1.0, ecolor='black'),
    )

    if cfg.get('show_points', True):
        rng = np.random.default_rng(42)
        for i, arr in enumerate(data_arrays):
            x_jitter = rng.normal(loc=i, scale=0.06, size=len(arr))
            ax.scatter(x_jitter, arr, s=cfg.get('point_size', 22),
                       color='black', alpha=0.55, zorder=4, edgecolor='none')

    if cfg.get('show_mean_label', False):
        for i, m in enumerate(means):
            ax.annotate(
                f'{m:.1f}', (i, m + errs[i]),
                textcoords='offset points', xytext=(0, 8),
                ha='center',
                fontsize=cfg['tick_size'],
                fontweight=font_weight(cfg['tick_bold']),
            )

    x_labels = [apply_italic_to_label(l, italic_texts, cfg['tick_bold']) if italic_texts else l
                for l in order]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=cfg.get('xtick_rotation', 45), ha='right')

    if cfg.get('ymin') is not None and cfg.get('ymax') is not None:
        ax.set_ylim(cfg['ymin'], cfg['ymax'])

    ax.set_xlabel(cfg.get('xlabel', ''))
    ax.set_ylabel(cfg.get('ylabel', ''))
    _style_axes(ax, cfg)

    _draw_comparison_brackets(ax, df[order], cfg, means_tops_mode='bar',
                              x_positions=x_pos.tolist(),
                              bar_tops=[m + e for m, e in zip(means, errs)])

    _apply_grid(ax, cfg)
    _apply_spines(ax, cfg)
    _apply_title(ax, cfg, italic_texts)
    plt.tight_layout()
    return fig


# -------------------------------------------------------------------
# Violin plot
# -------------------------------------------------------------------
def render_violin(df, cfg, fig_size_inches):
    fig, ax = plt.subplots(figsize=fig_size_inches)

    order = _sort_columns(df, cfg.get('sort', '원본 순서'))
    italic_texts = cfg.get('italic_texts', [])
    palette_name = cfg.get('palette', 'Set2')
    colors = sns.color_palette(palette_name, len(order))

    data_arrays = [df[c].dropna().values for c in order]

    parts = ax.violinplot(
        data_arrays,
        positions=range(1, len(order) + 1),
        widths=cfg.get('box_width', 0.75),
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[i])
        body.set_alpha(cfg.get('alpha', 0.6))
        body.set_edgecolor('black')
        body.set_linewidth(0.8)

    # 내부 boxplot (inner='box')
    if cfg.get('inner_box', True):
        for i, arr in enumerate(data_arrays):
            if len(arr) == 0:
                continue
            q1, med, q3 = np.percentile(arr, [25, 50, 75])
            iqr = q3 - q1
            whisker_lo = max(np.min(arr), q1 - 1.5 * iqr)
            whisker_hi = min(np.max(arr), q3 + 1.5 * iqr)
            x = i + 1
            ax.vlines(x, whisker_lo, whisker_hi, color='black', linewidth=1.0)
            ax.vlines(x, q1, q3, color='black', linewidth=6.0)
            ax.scatter(x, med, color='white', s=18, zorder=5, edgecolor='black', linewidth=0.8)

    # 개별 점
    if cfg.get('show_points', False):
        rng = np.random.default_rng(42)
        for i, arr in enumerate(data_arrays):
            x_jitter = rng.normal(loc=i + 1, scale=0.05, size=len(arr))
            ax.scatter(x_jitter, arr, s=cfg.get('point_size', 18),
                       color='black', alpha=0.45, zorder=3, edgecolor='none')

    x_labels = [apply_italic_to_label(l, italic_texts, cfg['tick_bold']) if italic_texts else l
                for l in order]
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(x_labels, rotation=cfg.get('xtick_rotation', 45), ha='right')

    if cfg.get('ymin') is not None and cfg.get('ymax') is not None:
        ax.set_ylim(cfg['ymin'], cfg['ymax'])

    ax.set_xlabel(cfg.get('xlabel', ''))
    ax.set_ylabel(cfg.get('ylabel', ''))
    _style_axes(ax, cfg)

    _draw_comparison_brackets(ax, df[order], cfg, means_tops_mode='violin')

    _apply_grid(ax, cfg)
    _apply_spines(ax, cfg)
    _apply_title(ax, cfg, italic_texts)
    plt.tight_layout()
    return fig


# -------------------------------------------------------------------
# Scatter plot with regression
# -------------------------------------------------------------------
def render_scatter(df, cfg, fig_size_inches):
    """
    df: 첫 열=X, 나머지 열=Y series. 각 Y 시리즈마다 점+회귀선.
    """
    fig, ax = plt.subplots(figsize=fig_size_inches)

    italic_texts = cfg.get('italic_texts', [])
    palette_name = cfg.get('palette', 'tab10')
    x_col = df.columns[0]
    y_cols = list(df.columns[1:])
    colors = sns.color_palette(palette_name, max(len(y_cols), 1))

    x = df[x_col].values.astype(float)

    markers = cfg.get('markers', ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*'])

    for i, y_col in enumerate(y_cols):
        y = df[y_col].values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(y))
        xv, yv = x[mask], y[mask]
        if len(xv) == 0:
            continue
        label_base = apply_italic_to_label(y_col, italic_texts, cfg['tick_bold']) if italic_texts else y_col

        ax.scatter(
            xv, yv,
            s=cfg.get('point_size', 45),
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            alpha=cfg.get('alpha', 0.8),
            edgecolor='black',
            linewidth=0.6,
            label=label_base,
        )

        if cfg.get('show_regression', True) and len(xv) >= 2:
            reg = linear_regression(xv, yv)
            if reg:
                x_line = np.linspace(np.min(xv), np.max(xv), 100)
                y_line = reg['slope'] * x_line + reg['intercept']
                ax.plot(x_line, y_line, color=colors[i % len(colors)],
                        linewidth=1.2, linestyle='-', alpha=0.9)
                if cfg.get('show_reg_equation', True):
                    txt = f"y={reg['slope']:.3g}x+{reg['intercept']:.3g}  R²={reg['r2']:.3f}"
                    ax.text(
                        0.02, 0.98 - 0.06 * i, txt,
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=max(6, cfg['tick_size'] - 1),
                        color=colors[i % len(colors)],
                        fontweight=font_weight(cfg['tick_bold']),
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                                  edgecolor='none', alpha=0.6),
                    )

    if cfg.get('ymin') is not None and cfg.get('ymax') is not None:
        ax.set_ylim(cfg['ymin'], cfg['ymax'])
    if cfg.get('xmin') is not None and cfg.get('xmax') is not None:
        ax.set_xlim(cfg['xmin'], cfg['xmax'])

    ax.set_xlabel(cfg.get('xlabel', x_col))
    ax.set_ylabel(cfg.get('ylabel', ''))
    _style_axes(ax, cfg)

    if y_cols:
        ax.legend(loc=cfg.get('legend_loc', 'best'),
                  fontsize=max(6, cfg['tick_size'] - 1),
                  frameon=cfg.get('legend_frame', True))

    _apply_grid(ax, cfg)
    _apply_spines(ax, cfg)
    _apply_title(ax, cfg, italic_texts)
    plt.tight_layout()
    return fig


# -------------------------------------------------------------------
# Grouped bar (2-factor design)
# -------------------------------------------------------------------
def render_grouped_bar(parsed_data, cfg, fig_size_inches):
    """Grouped bar chart for data_model.GROUPED parsed data.

    Each factor1 level is a group on the X axis; bars within a group are
    color-coded by factor2 level, placed side by side.
    """
    fig, ax = plt.subplots(figsize=fig_size_inches)

    italic_texts = cfg.get('italic_texts', [])
    palette_name = cfg.get('palette', 'Set2')

    factor1_name = parsed_data.get('factor1_name', 'factor1')
    factor2_names = list(parsed_data.get('factor2_names', []))
    levels = parsed_data.get('levels', {})

    factor1_levels = list(levels.keys())
    n1 = len(factor1_levels)
    n2 = len(factor2_names)

    if n1 == 0 or n2 == 0:
        ax.text(0.5, 0.5, '데이터가 비어 있습니다.', ha='center', va='center',
                transform=ax.transAxes)
        _apply_title(ax, cfg, italic_texts)
        plt.tight_layout()
        return fig

    colors = sns.color_palette(palette_name, max(n2, 1))

    x = np.arange(n1)
    width = 0.8 / max(n2, 1)

    err_mode = cfg.get('err_mode', 'SEM')
    show_points = cfg.get('show_points', False)

    # means[lvl2] -> list of means per factor1 level, same ordering as factor1_levels.
    means_by_f2 = {name: [] for name in factor2_names}
    errs_by_f2 = {name: [] for name in factor2_names}
    raw_by_f2 = {name: [] for name in factor2_names}

    for lvl1 in factor1_levels:
        bucket = levels.get(lvl1, {})
        for name in factor2_names:
            arr = np.asarray(bucket.get(name, []), dtype=float)
            clean = arr[np.isfinite(arr)]
            raw_by_f2[name].append(clean)
            if clean.size == 0:
                means_by_f2[name].append(np.nan)
                errs_by_f2[name].append(0.0)
                continue
            means_by_f2[name].append(float(np.nanmean(clean)))
            d = describe(clean)
            if err_mode == 'SD':
                errs_by_f2[name].append(float(d['std']))
            elif err_mode == '95%CI':
                errs_by_f2[name].append(float(d['ci95']))
            elif err_mode == 'none':
                errs_by_f2[name].append(0.0)
            else:  # SEM default
                errs_by_f2[name].append(float(d['sem']))

    rng = np.random.default_rng(42)
    bar_tops_for_group = np.full(n1, -np.inf)

    for i, lvl2 in enumerate(factor2_names):
        offset = (i - (n2 - 1) / 2.0) * width
        x_pos = x + offset
        means = np.array(means_by_f2[lvl2], dtype=float)
        errs = np.array(errs_by_f2[lvl2], dtype=float)
        color = colors[i % len(colors)]

        label = (apply_italic_to_label(lvl2, italic_texts, cfg['tick_bold'])
                 if italic_texts else lvl2)

        ax.bar(
            x_pos, means,
            yerr=errs if err_mode != 'none' else None,
            capsize=cfg.get('capsize', 4),
            width=width * 0.9,
            color=color,
            edgecolor='black',
            linewidth=1.0,
            alpha=cfg.get('alpha', 0.9),
            label=label,
            error_kw=dict(elinewidth=1.0, ecolor=color),
        )

        if show_points:
            for j, clean in enumerate(raw_by_f2[lvl2]):
                if clean.size == 0:
                    continue
                jitter = rng.normal(loc=x_pos[j], scale=width * 0.08, size=clean.size)
                ax.scatter(jitter, clean, s=cfg.get('point_size', 18),
                           color='black', alpha=0.55, zorder=4, edgecolor='none')

        # Track tops for bracket placement
        tops = means + errs
        for j in range(n1):
            if np.isfinite(tops[j]) and tops[j] > bar_tops_for_group[j]:
                bar_tops_for_group[j] = tops[j]

    # X tick labels = factor1 levels
    x_labels = [apply_italic_to_label(l, italic_texts, cfg['tick_bold']) if italic_texts else l
                for l in factor1_levels]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=cfg.get('xtick_rotation', 0), ha='center'
                       if cfg.get('xtick_rotation', 0) == 0 else 'right')

    if cfg.get('ymin') is not None and cfg.get('ymax') is not None:
        ax.set_ylim(cfg['ymin'], cfg['ymax'])

    ax.set_xlabel(cfg.get('xlabel', factor1_name))
    ax.set_ylabel(cfg.get('ylabel', ''))
    _style_axes(ax, cfg)

    # Significance brackets: within each factor1 group, compare across factor2 levels.
    if cfg.get('show_brackets', False):
        for j, lvl1 in enumerate(factor1_levels):
            sub = {name: raw_by_f2[name][j] for name in factor2_names
                   if raw_by_f2[name][j].size > 0}
            if len(sub) < 2:
                continue
            max_len = max(len(v) for v in sub.values())
            sub_padded = {k: np.concatenate([v, np.full(max_len - len(v), np.nan)])
                          for k, v in sub.items()}
            import pandas as _pd
            df_sub = _pd.DataFrame(sub_padded)
            # x positions for this group's bars
            sub_names = list(sub.keys())
            sub_positions = []
            sub_tops = []
            for name in sub_names:
                i = factor2_names.index(name)
                offset = (i - (n2 - 1) / 2.0) * width
                sub_positions.append(x[j] + offset)
                mean_val = means_by_f2[name][j]
                err_val = errs_by_f2[name][j]
                sub_tops.append((mean_val if np.isfinite(mean_val) else 0.0)
                                + (err_val if np.isfinite(err_val) else 0.0))
            _draw_comparison_brackets(
                ax, df_sub[sub_names], cfg, means_tops_mode='bar',
                x_positions=sub_positions, bar_tops=sub_tops,
            )

    # Legend for factor2 levels
    ax.legend(title=parsed_data.get('factor2_name', 'factor2')
              if parsed_data.get('factor2_name') else None,
              loc=cfg.get('legend_loc', 'best'),
              fontsize=max(6, cfg['tick_size'] - 1),
              frameon=cfg.get('legend_frame', True))

    _apply_grid(ax, cfg)
    _apply_spines(ax, cfg)
    _apply_title(ax, cfg, italic_texts)
    plt.tight_layout()
    return fig


# -------------------------------------------------------------------
# Dose-Response curve
# -------------------------------------------------------------------
def render_dose_response(parsed_data, fit_result, cfg, fig_size_inches):
    """Dose-response curve with replicate points, fitted curve, CI band, EC50 marker."""
    fig, ax = plt.subplots(figsize=fig_size_inches)

    italic_texts = cfg.get('italic_texts', [])

    dose = np.asarray(parsed_data.get('dose', []), dtype=float)
    responses = parsed_data.get('responses', []) or []
    dose_log = bool(parsed_data.get('dose_log', False))

    color = cfg.get('palette_single', '#c0392b')
    point_size = cfg.get('point_size', 36)
    show_ci_band = cfg.get('show_ci_band', True)
    show_ec50_marker = cfg.get('show_ec50_marker', True)
    dose_unit = cfg.get('dose_unit', '')

    # Replicate points (semi-transparent).
    rep_arrays = [np.asarray(r, dtype=float) for r in responses]
    # Plot each replicate vs dose.
    for rep in rep_arrays:
        if rep.size == 0:
            continue
        n = min(len(dose), len(rep))
        xv = dose[:n]
        yv = rep[:n]
        mask = np.isfinite(xv) & np.isfinite(yv)
        if dose_log:
            mask &= (xv > 0)
        ax.scatter(xv[mask], yv[mask], s=point_size, color=color,
                   alpha=0.35, edgecolor='none', zorder=3)

    # Mean + error bars across replicates at each dose.
    if rep_arrays:
        stacked = np.vstack([
            np.concatenate([r, np.full(len(dose) - len(r), np.nan)])
            if len(r) < len(dose) else r[:len(dose)]
            for r in rep_arrays
        ])
        y_mean = np.nanmean(stacked, axis=0)
        y_std = np.nanstd(stacked, axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(y_mean)
        n_per = np.sum(np.isfinite(stacked), axis=0)
        y_sem = np.where(n_per > 1, y_std / np.sqrt(np.maximum(n_per, 1)), 0.0)
        err_mode = cfg.get('err_mode', 'SEM')
        if err_mode == 'SD':
            y_err = y_std
        elif err_mode == 'none':
            y_err = np.zeros_like(y_mean)
        else:
            y_err = y_sem

        mask = np.isfinite(dose) & np.isfinite(y_mean)
        if dose_log:
            mask &= (dose > 0)
        if np.any(mask):
            ax.errorbar(
                dose[mask], y_mean[mask],
                yerr=(y_err[mask] if err_mode != 'none' else None),
                fmt='o', color=color, markerfacecolor=color,
                markeredgecolor='black', markeredgewidth=0.6,
                markersize=max(4, np.sqrt(point_size)),
                elinewidth=1.0, capsize=4, ecolor=color, zorder=5,
            )

    # Fitted curve.
    if fit_result is not None and getattr(fit_result, 'converged', False):
        finite_dose = dose[np.isfinite(dose) & (dose > 0)] if dose_log else dose[np.isfinite(dose)]
        if finite_dose.size > 0:
            if dose_log:
                lo = np.log10(finite_dose.min()) - 0.3
                hi = np.log10(finite_dose.max()) + 0.3
                x_smooth_log = np.linspace(lo, hi, 200)
                x_smooth = 10.0 ** x_smooth_log
                # fit was done in log-dose domain (Hill 4PL etc.), so feed log(x)
                try:
                    y_smooth = fit_result.predict(x_smooth_log)
                except Exception:
                    y_smooth = fit_result.predict(x_smooth)
                x_plot = x_smooth
                x_for_ci = x_smooth_log
            else:
                lo = finite_dose.min()
                hi = finite_dose.max()
                pad = (hi - lo) * 0.05 if hi > lo else 1.0
                x_smooth = np.linspace(lo - pad, hi + pad, 200)
                y_smooth = fit_result.predict(x_smooth)
                x_plot = x_smooth
                x_for_ci = x_smooth

            ax.plot(x_plot, y_smooth, color=color, linewidth=1.8,
                    linestyle='-', alpha=0.95, zorder=6)

            if show_ci_band:
                try:
                    ci_lo, ci_hi = fit_result.predict_ci(x_for_ci, alpha=0.05)
                    if np.all(np.isfinite(ci_lo)) and np.all(np.isfinite(ci_hi)):
                        ax.fill_between(x_plot, ci_lo, ci_hi,
                                        color=color, alpha=0.2, zorder=4,
                                        edgecolor='none')
                except Exception:
                    pass

        # EC50 annotation.
        if show_ec50_marker:
            ec50 = None
            ec50_ci = (None, None)
            derived = getattr(fit_result, 'derived', {}) or {}
            for key in ('EC50', 'LC50', 'EC50_effective', 'ED50'):
                if key in derived:
                    try:
                        ec50 = float(derived[key])
                    except (TypeError, ValueError):
                        ec50 = None
                    ci_key = f'{key}_ci95'
                    if ci_key in derived:
                        ci = derived[ci_key]
                        if isinstance(ci, tuple) and len(ci) == 2:
                            ec50_ci = (float(ci[0]), float(ci[1]))
                    break

            if ec50 is not None and np.isfinite(ec50):
                if (not dose_log) or ec50 > 0:
                    ax.axvline(ec50, linestyle='--', color='red',
                               alpha=0.6, linewidth=1.0, zorder=5)
                    ymin, ymax = ax.get_ylim()
                    y_annot = ymin + 0.5 * (ymax - ymin)
                    if (ec50_ci[0] is not None and ec50_ci[1] is not None
                            and np.isfinite(ec50_ci[0]) and np.isfinite(ec50_ci[1])):
                        txt = (f"EC50 = {ec50:.3g} "
                               f"(95% CI: {ec50_ci[0]:.3g}–{ec50_ci[1]:.3g})")
                    else:
                        txt = f"EC50 = {ec50:.3g}"
                    if dose_unit:
                        txt = txt + f" {dose_unit}"
                    ax.annotate(
                        txt,
                        xy=(ec50, y_annot),
                        xytext=(8, 0), textcoords='offset points',
                        ha='left', va='center',
                        fontsize=max(7, cfg['tick_size'] - 1),
                        fontweight=font_weight(cfg['tick_bold']),
                        color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='red', alpha=0.85),
                    )

        # R² + model name corner annotation.
        try:
            r2 = float(getattr(fit_result, 'r_squared', float('nan')))
            model_name = getattr(fit_result, 'model_name', '')
            if np.isfinite(r2):
                ax.text(
                    0.02, 0.98,
                    f"{model_name}\nR² = {r2:.3f}",
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=max(7, cfg['tick_size'] - 1),
                    fontweight=font_weight(cfg['tick_bold']),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.85),
                )
        except Exception:
            pass

    if dose_log:
        ax.set_xscale('log')

    # Default axis labels.
    if cfg.get('xlabel'):
        ax.set_xlabel(cfg['xlabel'])
    else:
        if dose_log:
            xl = "Log₁₀(Dose)"
        else:
            xl = "Dose"
        if dose_unit:
            xl = f"{xl} ({dose_unit})"
        ax.set_xlabel(xl)

    ax.set_ylabel(cfg.get('ylabel', 'Mortality (%)'))

    if cfg.get('ymin') is not None and cfg.get('ymax') is not None:
        ax.set_ylim(cfg['ymin'], cfg['ymax'])

    _style_axes(ax, cfg)
    _apply_grid(ax, cfg)
    _apply_spines(ax, cfg)
    _apply_title(ax, cfg, italic_texts)
    plt.tight_layout()
    return fig


# -------------------------------------------------------------------
# XY with error bars (multi-series)
# -------------------------------------------------------------------
def render_xy_errorbars(parsed_data, cfg, fig_size_inches):
    """XY error-bar plot: multiple Y series vs shared X, with error bars.

    parsed_data['y_series'][name] may be 1D (single measurement per x) or
    2D (replicates per x, axis 0 = replicates).
    """
    fig, ax = plt.subplots(figsize=fig_size_inches)

    italic_texts = cfg.get('italic_texts', [])
    palette_name = cfg.get('palette', 'tab10')

    x = np.asarray(parsed_data.get('x', []), dtype=float)
    y_series = parsed_data.get('y_series', {}) or {}
    series_names = list(y_series.keys())
    n_series = len(series_names)

    if n_series == 0 or x.size == 0:
        ax.text(0.5, 0.5, '데이터가 비어 있습니다.', ha='center', va='center',
                transform=ax.transAxes)
        _apply_title(ax, cfg, italic_texts)
        plt.tight_layout()
        return fig

    colors = sns.color_palette(palette_name, max(n_series, 1))
    markers = cfg.get('markers', ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*'])
    linestyles = cfg.get('linestyles', ['-', '--', '-.', ':'])

    err_mode = cfg.get('err_mode', 'SEM')
    connect = cfg.get('connect_lines', True)
    line_width = cfg.get('line_width', 1.4)
    marker_size = cfg.get('marker_size', 6)

    for i, name in enumerate(series_names):
        arr = np.asarray(y_series[name], dtype=float)
        if arr.ndim == 1:
            y_mean = arr
            y_err = None
        else:
            # 2D: replicates along axis 0, x along axis 1
            y_mean = np.nanmean(arr, axis=0)
            if err_mode == 'none':
                y_err = None
            else:
                y_std = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
                n_per = np.sum(np.isfinite(arr), axis=0)
                y_sem = np.where(n_per > 1, y_std / np.sqrt(np.maximum(n_per, 1)), 0.0)
                if err_mode == 'SD':
                    y_err = y_std
                elif err_mode == '95%CI':
                    from scipy import stats as _sp
                    dof = np.maximum(n_per - 1, 1)
                    tval = _sp.t.ppf(0.975, dof)
                    y_err = tval * y_sem
                else:
                    y_err = y_sem

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ls = linestyles[i % len(linestyles)] if connect else 'None'

        label = (apply_italic_to_label(name, italic_texts, cfg['tick_bold'])
                 if italic_texts else name)

        n = min(len(x), len(y_mean))
        xv = x[:n]
        yv = y_mean[:n]
        ev = y_err[:n] if y_err is not None else None

        mask = np.isfinite(xv) & np.isfinite(yv)
        if not np.any(mask):
            continue

        ax.errorbar(
            xv[mask], yv[mask],
            yerr=(ev[mask] if ev is not None else None),
            fmt=marker,
            linestyle=ls if connect else 'None',
            color=color,
            markerfacecolor=color,
            markeredgecolor='black',
            markeredgewidth=0.6,
            markersize=marker_size,
            linewidth=line_width,
            elinewidth=1.0,
            capsize=4,
            ecolor=color,
            label=label,
        )

    if cfg.get('log_x', False):
        ax.set_xscale('log')
    if cfg.get('log_y', False):
        ax.set_yscale('log')

    if cfg.get('xmin') is not None and cfg.get('xmax') is not None:
        ax.set_xlim(cfg['xmin'], cfg['xmax'])
    if cfg.get('ymin') is not None and cfg.get('ymax') is not None:
        ax.set_ylim(cfg['ymin'], cfg['ymax'])

    ax.set_xlabel(cfg.get('xlabel', parsed_data.get('x_name', 'X')))
    ax.set_ylabel(cfg.get('ylabel', ''))
    _style_axes(ax, cfg)

    ax.legend(loc=cfg.get('legend_loc', 'best'),
              fontsize=max(6, cfg['tick_size'] - 1),
              frameon=cfg.get('legend_frame', True))

    _apply_grid(ax, cfg)
    _apply_spines(ax, cfg)
    _apply_title(ax, cfg, italic_texts)
    plt.tight_layout()
    return fig


# -------------------------------------------------------------------
# Interaction plot (two-way ANOVA interpretation)
# -------------------------------------------------------------------
def render_interaction_plot(parsed_data, cfg, fig_size_inches):
    """Interaction plot: factor1 on X, separate lines per factor2 level.

    Y is mean of each (factor1, factor2) combination. Non-parallel lines
    suggest an interaction effect.
    """
    fig, ax = plt.subplots(figsize=fig_size_inches)

    italic_texts = cfg.get('italic_texts', [])
    palette_name = cfg.get('palette', 'Set1')

    factor1_name = parsed_data.get('factor1_name', 'factor1')
    factor2_names = list(parsed_data.get('factor2_names', []))
    levels = parsed_data.get('levels', {})

    factor1_levels = list(levels.keys())
    n1 = len(factor1_levels)
    n2 = len(factor2_names)

    if n1 == 0 or n2 == 0:
        ax.text(0.5, 0.5, '데이터가 비어 있습니다.', ha='center', va='center',
                transform=ax.transAxes)
        _apply_title(ax, cfg, italic_texts)
        plt.tight_layout()
        return fig

    colors = sns.color_palette(palette_name, max(n2, 1))
    markers = cfg.get('markers', ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*'])
    line_width = cfg.get('line_width', 1.6)
    show_errors = cfg.get('show_errors', True)
    err_mode = cfg.get('err_mode', 'SEM')

    x = np.arange(n1, dtype=float)

    for i, lvl2 in enumerate(factor2_names):
        means = []
        errs = []
        for lvl1 in factor1_levels:
            arr = np.asarray(levels.get(lvl1, {}).get(lvl2, []), dtype=float)
            clean = arr[np.isfinite(arr)]
            if clean.size == 0:
                means.append(np.nan)
                errs.append(0.0)
                continue
            means.append(float(np.nanmean(clean)))
            d = describe(clean)
            if err_mode == 'SD':
                errs.append(float(d['std']))
            elif err_mode == '95%CI':
                errs.append(float(d['ci95']))
            elif err_mode == 'none':
                errs.append(0.0)
            else:
                errs.append(float(d['sem']))

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        label = (apply_italic_to_label(lvl2, italic_texts, cfg['tick_bold'])
                 if italic_texts else lvl2)

        means_arr = np.array(means, dtype=float)
        errs_arr = np.array(errs, dtype=float)

        if show_errors and err_mode != 'none':
            ax.errorbar(
                x, means_arr, yerr=errs_arr,
                fmt=marker, linestyle='-', linewidth=line_width,
                color=color, markerfacecolor=color,
                markeredgecolor='black', markeredgewidth=0.6,
                markersize=cfg.get('marker_size', 7),
                elinewidth=1.0, capsize=4, ecolor=color,
                label=label,
            )
        else:
            ax.plot(
                x, means_arr,
                marker=marker, linestyle='-', linewidth=line_width,
                color=color, markerfacecolor=color,
                markeredgecolor='black', markeredgewidth=0.6,
                markersize=cfg.get('marker_size', 7),
                label=label,
            )

    x_labels = [apply_italic_to_label(l, italic_texts, cfg['tick_bold']) if italic_texts else l
                for l in factor1_levels]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=cfg.get('xtick_rotation', 0),
                       ha='center' if cfg.get('xtick_rotation', 0) == 0 else 'right')

    if cfg.get('ymin') is not None and cfg.get('ymax') is not None:
        ax.set_ylim(cfg['ymin'], cfg['ymax'])

    ax.set_xlabel(cfg.get('xlabel', factor1_name))
    ax.set_ylabel(cfg.get('ylabel', ''))
    _style_axes(ax, cfg)

    ax.legend(title=parsed_data.get('factor2_name', None),
              loc=cfg.get('legend_loc', 'best'),
              fontsize=max(6, cfg['tick_size'] - 1),
              frameon=cfg.get('legend_frame', True))

    _apply_grid(ax, cfg)
    _apply_spines(ax, cfg)
    _apply_title(ax, cfg, italic_texts)
    plt.tight_layout()
    return fig


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _sort_columns(df, sort_option):
    if sort_option == '평균값 내림차순':
        return df.mean(numeric_only=True).sort_values(ascending=False).index.tolist()
    if sort_option == '평균값 오름차순':
        return df.mean(numeric_only=True).sort_values(ascending=True).index.tolist()
    if sort_option == '중앙값 내림차순':
        return df.median(numeric_only=True).sort_values(ascending=False).index.tolist()
    if sort_option == '중앙값 오름차순':
        return df.median(numeric_only=True).sort_values(ascending=True).index.tolist()
    return df.columns.tolist()


def _draw_comparison_brackets(ax, df_ordered, cfg, means_tops_mode='box',
                              x_positions=None, bar_tops=None):
    """유의성 브래킷을 그린다. df_ordered는 표시 순서와 동일한 열 순서."""
    if not cfg.get('show_brackets', False):
        return
    test = cfg.get('test', 't-test')
    correction = cfg.get('correction', 'bonferroni')
    only_significant = True

    results = pairwise_tests(df_ordered, test=test, correction=correction)
    if not results:
        return

    cols = df_ordered.columns.tolist()

    if x_positions is None:
        if means_tops_mode in ('box', 'violin'):
            x_positions = list(range(1, len(cols) + 1))
        else:
            x_positions = list(range(len(cols)))

    name_to_x = {c: x for c, x in zip(cols, x_positions)}

    if bar_tops is None:
        group_tops = {c: float(np.nanmax(df_ordered[c])) if df_ordered[c].notna().any() else 0
                      for c in cols}
    else:
        group_tops = {c: t for c, t in zip(cols, bar_tops)}

    ymin, ymax = ax.get_ylim()
    brackets = build_bracket_positions(
        results, name_to_x, group_tops,
        offset_frac=cfg.get('bracket_offset', 0.04),
        tier_frac=cfg.get('bracket_tier', 0.06),
        y_range=(ymax - ymin),
    )
    if not brackets:
        return

    # ymax가 브래킷을 덮도록 확장
    top_needed = max(y for _, _, y, _ in brackets)
    span = ymax - ymin
    if top_needed + span * 0.05 > ymax:
        ax.set_ylim(ymin, top_needed + span * 0.08)

    draw_brackets_xy(
        ax, brackets,
        linewidth=cfg.get('bracket_linewidth', 1.0),
        fontsize=cfg.get('bracket_fontsize', max(cfg['tick_size'] - 1, 7)),
        color='black',
    )
