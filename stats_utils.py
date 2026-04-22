"""통계 분석: 기술통계, t-test, ANOVA, 유의성 브래킷 + GraphPad Prism 수준 확장"""

import numpy as np
import pandas as pd
from itertools import combinations
from dataclasses import dataclass, field
from typing import Optional

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import scikit_posthocs as sp_ph
    HAS_SKPH = True
except ImportError:
    HAS_SKPH = False


_NO_STATSMODELS_MSG = "statsmodels가 필요합니다: pip install statsmodels"
_NO_SCIPY_MSG = "scipy가 필요합니다: pip install scipy"


def _require_statsmodels():
    if not HAS_STATSMODELS:
        raise RuntimeError(_NO_STATSMODELS_MSG)


def _require_scipy():
    if not HAS_SCIPY:
        raise RuntimeError(_NO_SCIPY_MSG)


def _fmt_num(v, digits=4):
    """숫자 포맷팅 (nan/None 안전)"""
    try:
        if v is None:
            return 'N/A'
        fv = float(v)
        if np.isnan(fv):
            return 'N/A'
        if abs(fv) < 1e-4 and fv != 0:
            return f'{fv:.3e}'
        return f'{fv:.{digits}f}'
    except (TypeError, ValueError):
        return str(v)


def _fmt_p(p):
    """p-value 전용 포맷팅"""
    try:
        if p is None or np.isnan(p):
            return 'N/A'
        if p < 0.0001:
            return '< 0.0001'
        return f'{p:.4f}'
    except (TypeError, ValueError):
        return str(p)


def describe(values):
    """기술통계 반환: n, mean, std, sem, ci95, median, q1, q3, min, max"""
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return {'n': 0, 'mean': np.nan, 'std': np.nan, 'sem': np.nan,
                'ci95': np.nan, 'median': np.nan, 'q1': np.nan, 'q3': np.nan,
                'min': np.nan, 'max': np.nan}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    if HAS_SCIPY and n > 1:
        ci95 = float(sp_stats.t.ppf(0.975, n - 1)) * sem
    else:
        ci95 = 1.96 * sem
    return {
        'n': n,
        'mean': mean,
        'std': std,
        'sem': sem,
        'ci95': ci95,
        'median': float(np.median(arr)),
        'q1': float(np.percentile(arr, 25)),
        'q3': float(np.percentile(arr, 75)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }


def describe_frame(df):
    """DataFrame의 각 열에 대해 describe() 반환"""
    return {col: describe(df[col].values) for col in df.columns}


def format_describe_text(desc_dict):
    """기술통계 dict를 보기 좋은 텍스트로"""
    lines = ['Group'.ljust(24) + 'n'.rjust(5) + 'Mean'.rjust(10) +
             'SD'.rjust(10) + 'SEM'.rjust(10) + '95%CI'.rjust(10) +
             'Median'.rjust(10) + 'Min'.rjust(9) + 'Max'.rjust(9)]
    for name, d in desc_dict.items():
        lines.append(
            f"{str(name)[:23].ljust(24)}"
            f"{d['n']:>5d}"
            f"{d['mean']:>10.3f}"
            f"{d['std']:>10.3f}"
            f"{d['sem']:>10.3f}"
            f"{d['ci95']:>10.3f}"
            f"{d['median']:>10.3f}"
            f"{d['min']:>9.3f}"
            f"{d['max']:>9.3f}"
        )
    return '\n'.join(lines)


def p_to_stars(p):
    """p-value를 별표로 변환"""
    if np.isnan(p):
        return 'ns'
    if p < 0.0001:
        return '****'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def pairwise_tests(df, test='t-test', correction='bonferroni'):
    """모든 쌍에 대해 검정. [(col_a, col_b, p, p_adj, stars)] 반환"""
    if not HAS_SCIPY:
        return []
    cols = list(df.columns)
    results = []
    for a, b in combinations(cols, 2):
        va = df[a].dropna().values
        vb = df[b].dropna().values
        if len(va) < 2 or len(vb) < 2:
            results.append((a, b, np.nan, np.nan, 'ns'))
            continue
        try:
            if test == 't-test':
                _, p = sp_stats.ttest_ind(va, vb, equal_var=False)
            elif test == 'welch':
                _, p = sp_stats.ttest_ind(va, vb, equal_var=False)
            elif test == 'mannwhitney':
                _, p = sp_stats.mannwhitneyu(va, vb, alternative='two-sided')
            else:
                _, p = sp_stats.ttest_ind(va, vb, equal_var=False)
        except Exception:
            p = np.nan
        results.append((a, b, float(p), np.nan, p_to_stars(p)))

    if correction == 'bonferroni' and results:
        m = sum(1 for r in results if not np.isnan(r[2]))
        adjusted = []
        for a, b, p, _, _ in results:
            if np.isnan(p):
                adjusted.append((a, b, p, np.nan, 'ns'))
            else:
                p_adj = min(p * m, 1.0)
                adjusted.append((a, b, p, p_adj, p_to_stars(p_adj)))
        return adjusted
    return results


def linear_regression(x, y):
    """선형회귀. dict(slope, intercept, r2, p, stderr) 반환"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return None
    if HAS_SCIPY:
        res = sp_stats.linregress(x, y)
        return {
            'slope': float(res.slope),
            'intercept': float(res.intercept),
            'r2': float(res.rvalue ** 2),
            'p': float(res.pvalue),
            'stderr': float(res.stderr),
            'n': int(len(x)),
        }
    # scipy 없을 때 fallback
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {'slope': float(slope), 'intercept': float(intercept),
            'r2': float(r2), 'p': np.nan, 'stderr': np.nan, 'n': int(len(x))}


def draw_brackets_xy(ax, brackets, linewidth=1.0, fontsize=10, color='black'):
    """
    brackets: [(x1, x2, y, label), ...]
    위에서 아래로 티어별로 배치하려면 호출자가 y를 조정해서 넘김
    """
    for x1, x2, y, label in brackets:
        bar_height = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + bar_height, y + bar_height, y],
            linewidth=linewidth, color=color, clip_on=False,
        )
        ax.text((x1 + x2) / 2, y + bar_height, label,
                ha='center', va='bottom', fontsize=fontsize, color=color)


def build_bracket_positions(pair_results, name_to_x, group_tops, offset_frac=0.05, tier_frac=0.07, y_range=1.0):
    """
    pair_results: [(a, b, p, p_adj, stars), ...]
    name_to_x: {group_name: x_coord}
    group_tops: {group_name: 그룹의 최대 y값}
    y_range: ymax - ymin (간격 계산용)

    반환: [(x1, x2, y, label), ...] — ns는 제외, 티어 겹침 방지
    """
    offset = y_range * offset_frac
    tier_step = y_range * tier_frac

    sig = []
    for a, b, p, p_adj, stars in pair_results:
        if stars == 'ns' or a not in name_to_x or b not in name_to_x:
            continue
        sig.append((a, b, stars, p if np.isnan(p_adj) else p_adj))

    # 길이 짧은 쌍 먼저
    sig.sort(key=lambda t: abs(name_to_x[t[1]] - name_to_x[t[0]]))

    brackets = []
    occupied = []  # [(xL, xR, y), ...]
    for a, b, stars, _ in sig:
        x1, x2 = sorted([name_to_x[a], name_to_x[b]])
        base = max(group_tops.get(a, 0), group_tops.get(b, 0)) + offset
        # 구간이 겹치는 기존 브래킷들의 y 중 최댓값 찾기
        conflict_y = 0
        for oL, oR, oy in occupied:
            if not (x2 < oL or x1 > oR):
                conflict_y = max(conflict_y, oy)
        y = max(base, conflict_y + tier_step) if conflict_y else base
        brackets.append((x1, x2, y, stars))
        occupied.append((x1, x2, y))
    return brackets


# ============================================================
# Phase 2: AnalysisResult + 확장 분석 함수
# ============================================================

@dataclass
class AnalysisResult:
    """모든 새로운 분석 함수가 반환하는 표준 결과 타입"""
    analysis_name: str
    summary: dict = field(default_factory=dict)
    table: Optional[pd.DataFrame] = None
    assumptions: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    # ---------- 변환 ----------
    def to_dict(self) -> dict:
        return {
            'analysis_name': self.analysis_name,
            'summary': dict(self.summary),
            'table': None if self.table is None else self.table.to_dict(orient='list'),
            'assumptions': dict(self.assumptions),
            'notes': list(self.notes),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Excel 내보내기용 평탄화 DataFrame (섹션별 rows)"""
        rows = []
        rows.append({'section': 'analysis', 'key': 'name', 'value': self.analysis_name})
        for k, v in self.summary.items():
            rows.append({'section': 'summary', 'key': str(k), 'value': _scalar(v)})
        for k, v in self.assumptions.items():
            rows.append({'section': 'assumptions', 'key': str(k), 'value': _scalar(v)})
        for i, note in enumerate(self.notes):
            rows.append({'section': 'notes', 'key': f'note_{i+1}', 'value': str(note)})
        head = pd.DataFrame(rows)
        if self.table is not None and len(self.table) > 0:
            tbl = self.table.copy()
            tbl.insert(0, 'section', 'table')
            # head와 열이 다르므로 concat 전에 맞춰줌
            return pd.concat([head, tbl], axis=0, ignore_index=True, sort=False)
        return head

    # ---------- 텍스트 리포트 ----------
    def to_text(self) -> str:
        lines = []
        lines.append('=' * 72)
        lines.append(f'  {self.analysis_name}')
        lines.append('=' * 72)

        if self.summary:
            lines.append('')
            lines.append('[ 요약 ]')
            key_w = max((len(str(k)) for k in self.summary.keys()), default=10)
            key_w = max(key_w, 14)
            for k, v in self.summary.items():
                lines.append(f'  {str(k).ljust(key_w)} : {_pretty_scalar(v, k)}')

        if self.assumptions:
            lines.append('')
            lines.append('[ 가정 검정 ]')
            key_w = max(len(str(k)) for k in self.assumptions.keys())
            key_w = max(key_w, 14)
            for k, v in self.assumptions.items():
                lines.append(f'  {str(k).ljust(key_w)} : {_pretty_scalar(v, k)}')

        if self.table is not None and len(self.table) > 0:
            lines.append('')
            lines.append('[ 세부 표 ]')
            try:
                tbl_str = self.table.to_string(index=False, float_format=lambda x: f'{x:.4f}')
            except Exception:
                tbl_str = str(self.table)
            for ln in tbl_str.split('\n'):
                lines.append('  ' + ln)

        if self.notes:
            lines.append('')
            lines.append('[ 비고 / 권고 ]')
            for note in self.notes:
                lines.append(f'  - {note}')

        lines.append('=' * 72)
        return '\n'.join(lines)


def _scalar(v):
    """Excel 셀에 적합한 스칼라로 변환"""
    if v is None:
        return ''
    if isinstance(v, (tuple, list, np.ndarray)):
        return ', '.join(_scalar(x) for x in v)
    try:
        if isinstance(v, (int, np.integer)):
            return int(v)
        fv = float(v)
        if np.isnan(fv):
            return ''
        return fv
    except (TypeError, ValueError):
        return str(v)


def _pretty_scalar(v, key=''):
    """to_text용 값 포맷"""
    key_l = str(key).lower()
    if isinstance(v, (tuple, list)):
        return '(' + ', '.join(_pretty_scalar(x) for x in v) + ')'
    if v is None:
        return 'N/A'
    try:
        fv = float(v)
        if np.isnan(fv):
            return 'N/A'
        if 'p' == key_l or key_l.startswith('p_') or key_l.endswith('_p') or 'p-value' in key_l or 'p value' in key_l:
            return _fmt_p(fv)
        if isinstance(v, (int, np.integer)) or (fv.is_integer() and abs(fv) < 1e9):
            return str(int(fv))
        return _fmt_num(fv)
    except (TypeError, ValueError):
        return str(v)


# ------------------------------------------------------------
# 2. Two-way / Three-way ANOVA
# ------------------------------------------------------------

def _build_anova(df_long, value_col, factors_in_order, interaction, analysis_name):
    """안전한 이름으로 리네이밍 후 statsmodels OLS + ANOVA 수행."""
    _require_statsmodels()
    data = df_long[[value_col] + list(factors_in_order)].dropna().copy()
    # patsy 안전한 이름으로 리네이밍 (C, Q 등 예약어 회피)
    safe_value = '__val__'
    safe_factors = [f'__f{i}__' for i in range(len(factors_in_order))]
    rename_map = {value_col: safe_value}
    for orig, safe in zip(factors_in_order, safe_factors):
        rename_map[orig] = safe
    data2 = data.rename(columns=rename_map)
    joiner = ' * ' if interaction else ' + '
    rhs = joiner.join(f'C({sf})' for sf in safe_factors)
    formula = f'{safe_value} ~ {rhs}'
    model = smf.ols(formula, data=data2).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    # 결과 index를 원래 factor 이름으로 복원
    reverse = {safe: orig for orig, safe in zip(factors_in_order, safe_factors)}
    def _restore(s):
        s = str(s)
        # 긴 이름부터 치환 (겹침 방지)
        for safe in sorted(reverse.keys(), key=len, reverse=True):
            s = s.replace(f'C({safe})', reverse[safe]).replace(safe, reverse[safe])
        return s
    aov.index = [_restore(idx) for idx in aov.index]
    # 효과크기 (η², partial η², ω²) 계산
    ss_total = aov['sum_sq'].sum()
    eta_sq = (aov['sum_sq'] / ss_total).rename('eta_sq')
    # partial η² = SS_effect / (SS_effect + SS_residual)
    ss_resid = aov['sum_sq'].get('Residual', np.nan)
    partial_eta = (aov['sum_sq'] / (aov['sum_sq'] + ss_resid)).rename('partial_eta_sq')
    # ω² = (SS - df * MS_error) / (SS_total + MS_error)
    try:
        ms_err = aov['sum_sq']['Residual'] / aov['df']['Residual']
        omega_sq = ((aov['sum_sq'] - aov['df'] * ms_err) /
                    (ss_total + ms_err)).rename('omega_sq')
    except Exception:
        omega_sq = pd.Series(np.nan, index=aov.index, name='omega_sq')

    tbl = pd.concat([aov, eta_sq, partial_eta, omega_sq], axis=1).reset_index()
    tbl = tbl.rename(columns={'index': 'source', 'sum_sq': 'SS',
                              'df': 'df', 'F': 'F', 'PR(>F)': 'p'})

    summary = {}
    for idx, row in tbl.iterrows():
        src = row['source']
        if src == 'Residual':
            continue
        summary[f'{src}_F'] = _safe_float(row.get('F'))
        summary[f'{src}_p'] = _safe_float(row.get('p'))
        summary[f'{src}_df'] = _safe_float(row.get('df'))
        summary[f'{src}_partial_eta_sq'] = _safe_float(row.get('partial_eta_sq'))

    summary['R_squared'] = float(model.rsquared)
    summary['R_squared_adj'] = float(model.rsquared_adj)

    # 잔차 가정 검정
    assumptions = {}
    try:
        resid = model.resid
        if len(resid) <= 5000:
            sw_stat, sw_p = sp_stats.shapiro(resid)
            assumptions['residual_shapiro_W'] = float(sw_stat)
            assumptions['residual_shapiro_p'] = float(sw_p)
    except Exception:
        pass

    notes = []
    for idx, row in tbl.iterrows():
        src = row['source']
        if src == 'Residual':
            continue
        p = row.get('p')
        if p is not None and not np.isnan(p) and p < 0.05:
            notes.append(f'{src}: 유의한 효과 (p={_fmt_p(p)}).')
    if assumptions.get('residual_shapiro_p', 1.0) < 0.05:
        notes.append('잔차가 정규성을 벗어남 (Shapiro-Wilk p<0.05) → 비모수 검정 고려.')

    return AnalysisResult(
        analysis_name=analysis_name,
        summary=summary,
        table=tbl,
        assumptions=assumptions,
        notes=notes,
    )


def two_way_anova(df_long, value_col, factor_a, factor_b, interaction=True):
    """이원분산분석 (Type II SS)"""
    if interaction:
        name = f'Two-way ANOVA ({factor_a} × {factor_b})'
    else:
        name = f'Two-way ANOVA ({factor_a} + {factor_b}, 주효과만)'
    return _build_anova(df_long, value_col, [factor_a, factor_b],
                        interaction, name)


def _safe_float(v):
    try:
        if v is None:
            return np.nan
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# ------------------------------------------------------------
# 3. Post-hoc tests
# ------------------------------------------------------------

def _get_groups(df_long, value_col, group_col):
    d = df_long[[value_col, group_col]].dropna()
    groups = {}
    for name, sub in d.groupby(group_col, sort=False):
        v = sub[value_col].values.astype(float)
        if len(v) >= 1:
            groups[name] = v
    return groups


def tukey_hsd(df_long, value_col, group_col):
    """Tukey HSD 다중비교 (statsmodels pairwise_tukeyhsd 사용)"""
    _require_statsmodels()
    d = df_long[[value_col, group_col]].dropna()
    if d[group_col].nunique() < 2:
        return None
    res = pairwise_tukeyhsd(d[value_col].values, d[group_col].astype(str).values)
    # statsmodels 결과를 DataFrame으로
    tbl = pd.DataFrame(data=res._results_table.data[1:],
                       columns=res._results_table.data[0])
    tbl = tbl.rename(columns={'group1': 'group1', 'group2': 'group2',
                              'meandiff': 'meandiff', 'p-adj': 'p_adj',
                              'lower': 'ci_lower', 'upper': 'ci_upper',
                              'reject': 'reject'})
    tbl['stars'] = tbl['p_adj'].apply(lambda p: p_to_stars(float(p)))

    summary = {
        'n_comparisons': int(len(tbl)),
        'n_significant': int(tbl['reject'].sum()),
        'alpha': 0.05,
        'method': 'Tukey HSD',
    }
    notes = []
    sig_tbl = tbl[tbl['reject'] == True]  # noqa: E712
    for _, row in sig_tbl.iterrows():
        notes.append(f'{row["group1"]} vs {row["group2"]}: 유의 (p_adj={_fmt_p(row["p_adj"])})')

    return AnalysisResult(
        analysis_name='Tukey HSD',
        summary=summary,
        table=tbl,
        notes=notes,
    )


def _pairwise_ttest_base(df_long, value_col, group_col, equal_var=True):
    """모든 쌍에 대해 독립표본 t-test → DataFrame(group1, group2, meandiff, t, p_raw, df)"""
    _require_scipy()
    groups = _get_groups(df_long, value_col, group_col)
    names = list(groups.keys())
    rows = []
    for a, b in combinations(names, 2):
        va, vb = groups[a], groups[b]
        if len(va) < 2 or len(vb) < 2:
            rows.append({'group1': a, 'group2': b, 'meandiff': np.nan,
                         't': np.nan, 'df': np.nan, 'p_raw': np.nan})
            continue
        t, p = sp_stats.ttest_ind(va, vb, equal_var=equal_var)
        if equal_var:
            df_v = len(va) + len(vb) - 2
        else:
            # Welch-Satterthwaite
            s1, s2 = np.var(va, ddof=1), np.var(vb, ddof=1)
            n1, n2 = len(va), len(vb)
            num = (s1 / n1 + s2 / n2) ** 2
            den = (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
            df_v = num / den if den > 0 else np.nan
        rows.append({
            'group1': a, 'group2': b,
            'meandiff': float(np.mean(vb) - np.mean(va)),
            't': float(t), 'df': float(df_v),
            'p_raw': float(p),
        })
    return pd.DataFrame(rows)


def dunnett(df_long, value_col, group_col, control_group):
    """Dunnett 다중비교 (vs. 단일 대조군). scipy 1.11+ 사용, 없으면 Bonferroni로 대체."""
    _require_scipy()
    groups = _get_groups(df_long, value_col, group_col)
    if control_group not in groups:
        raise ValueError(f"대조군 '{control_group}'을 찾을 수 없습니다.")
    control = groups[control_group]
    others = {k: v for k, v in groups.items() if k != control_group}
    if not others:
        return None

    rows = []
    method = 'Dunnett'
    # scipy 1.11+
    dunnett_fn = getattr(sp_stats, 'dunnett', None)
    if dunnett_fn is not None:
        try:
            sample_list = list(others.values())
            name_list = list(others.keys())
            res = dunnett_fn(*sample_list, control=control)
            for name, stat, p in zip(name_list, res.statistic, res.pvalue):
                v_t = others[name]
                rows.append({
                    'control': control_group, 'group': name,
                    'meandiff': float(np.mean(v_t) - np.mean(control)),
                    'statistic': float(stat),
                    'p_adj': float(p),
                })
        except Exception:
            rows = []  # fallback으로
    if not rows:
        # fallback: t-test + Bonferroni
        method = 'Dunnett (Bonferroni fallback)'
        m = len(others)
        for name, v_t in others.items():
            if len(v_t) < 2 or len(control) < 2:
                rows.append({'control': control_group, 'group': name,
                             'meandiff': np.nan, 'statistic': np.nan, 'p_adj': np.nan})
                continue
            t, p = sp_stats.ttest_ind(control, v_t, equal_var=True)
            rows.append({
                'control': control_group, 'group': name,
                'meandiff': float(np.mean(v_t) - np.mean(control)),
                'statistic': float(t),
                'p_adj': min(float(p) * m, 1.0),
            })

    tbl = pd.DataFrame(rows)
    tbl['stars'] = tbl['p_adj'].apply(lambda p: p_to_stars(float(p)) if not pd.isna(p) else 'ns')
    summary = {
        'method': method,
        'control_group': control_group,
        'n_comparisons': int(len(tbl)),
        'n_significant': int((tbl['p_adj'] < 0.05).sum()),
    }
    notes = []
    for _, row in tbl[tbl['p_adj'] < 0.05].iterrows():
        notes.append(f'{row["group"]} vs {control_group}: 유의 (p_adj={_fmt_p(row["p_adj"])})')
    return AnalysisResult(
        analysis_name='Dunnett',
        summary=summary,
        table=tbl,
        notes=notes,
    )


def _apply_correction(p_raw, method):
    """p-values 배열에 보정 적용. method: 'sidak', 'holm-sidak', 'bonferroni', 'holm'"""
    p = np.asarray(p_raw, dtype=float)
    n = len(p)
    if n == 0:
        return p
    out = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    if method == 'sidak':
        out[mask] = np.minimum(1.0, 1.0 - (1.0 - p[mask]) ** mask.sum())
    elif method == 'bonferroni':
        out[mask] = np.minimum(1.0, p[mask] * mask.sum())
    elif method in ('holm', 'holm-sidak'):
        order = np.argsort(p[mask])
        p_sorted = p[mask][order]
        m = len(p_sorted)
        adj = np.empty(m)
        for i in range(m):
            if method == 'holm':
                adj[i] = min(1.0, p_sorted[i] * (m - i))
            else:  # holm-sidak: step-down
                adj[i] = min(1.0, 1.0 - (1.0 - p_sorted[i]) ** (m - i))
        # step-down 단조증가 보장
        for i in range(1, m):
            adj[i] = max(adj[i], adj[i-1])
        unsorted = np.empty(m)
        unsorted[order] = adj
        tmp = out[mask].copy()
        tmp[:] = unsorted
        out[mask] = tmp
    else:
        out = p
    return out


def _pairwise_with_correction(df_long, value_col, group_col, method_name,
                              correction_key, analysis_name, equal_var=True):
    tbl = _pairwise_ttest_base(df_long, value_col, group_col, equal_var=equal_var)
    tbl['p_adj'] = _apply_correction(tbl['p_raw'].values, correction_key)
    tbl['stars'] = tbl['p_adj'].apply(lambda p: p_to_stars(float(p)) if not pd.isna(p) else 'ns')
    summary = {
        'method': method_name,
        'n_comparisons': int(len(tbl)),
        'n_significant': int((tbl['p_adj'] < 0.05).sum()),
        'alpha': 0.05,
    }
    notes = [
        f'{row["group1"]} vs {row["group2"]}: 유의 (p_adj={_fmt_p(row["p_adj"])})'
        for _, row in tbl[tbl['p_adj'] < 0.05].iterrows()
    ]
    return AnalysisResult(
        analysis_name=analysis_name,
        summary=summary,
        table=tbl,
        notes=notes,
    )


def sidak(df_long, value_col, group_col):
    """Šidák 보정 (all pairs)"""
    return _pairwise_with_correction(df_long, value_col, group_col,
                                     'Sidak', 'sidak', 'Sidak')


def holm_sidak(df_long, value_col, group_col):
    """Holm-Šidák step-down 보정"""
    return _pairwise_with_correction(df_long, value_col, group_col,
                                     'Holm-Sidak', 'holm-sidak', 'Holm-Sidak')


def scheffe(df_long, value_col, group_col):
    """Scheffé 보수적 다중비교 (all pairs). F 분포 기반."""
    _require_scipy()
    groups = _get_groups(df_long, value_col, group_col)
    names = list(groups.keys())
    k = len(names)
    if k < 2:
        return None
    all_vals = np.concatenate(list(groups.values()))
    n_total = len(all_vals)
    grand_mean = np.mean(all_vals)
    ss_within = sum(((v - np.mean(v)) ** 2).sum() for v in groups.values())
    df_within = n_total - k
    ms_within = ss_within / df_within if df_within > 0 else np.nan

    rows = []
    for a, b in combinations(names, 2):
        va, vb = groups[a], groups[b]
        na, nb = len(va), len(vb)
        if na < 2 or nb < 2 or ms_within <= 0:
            rows.append({'group1': a, 'group2': b, 'meandiff': np.nan,
                         'F_scheffe': np.nan, 'p_adj': np.nan})
            continue
        diff = np.mean(vb) - np.mean(va)
        se = np.sqrt(ms_within * (1.0 / na + 1.0 / nb))
        t = diff / se
        # Scheffé: F* = t^2 / (k-1), 비교 기준은 F(k-1, N-k)
        F_val = (t ** 2) / (k - 1)
        p = 1 - sp_stats.f.cdf(F_val, k - 1, df_within)
        rows.append({'group1': a, 'group2': b, 'meandiff': float(diff),
                     'F_scheffe': float(F_val), 'p_adj': float(p)})

    tbl = pd.DataFrame(rows)
    tbl['stars'] = tbl['p_adj'].apply(lambda p: p_to_stars(float(p)) if not pd.isna(p) else 'ns')
    summary = {
        'method': 'Scheffe',
        'n_comparisons': int(len(tbl)),
        'n_significant': int((tbl['p_adj'] < 0.05).sum()),
        'df_within': int(df_within),
        'MS_within': float(ms_within) if not np.isnan(ms_within) else np.nan,
    }
    notes = [
        f'{row["group1"]} vs {row["group2"]}: 유의 (p_adj={_fmt_p(row["p_adj"])})'
        for _, row in tbl[tbl['p_adj'] < 0.05].iterrows()
    ]
    notes.append('Scheffé는 가장 보수적인 다중비교로, 탐색적 대조에도 안전합니다.')
    return AnalysisResult(
        analysis_name='Scheffé',
        summary=summary,
        table=tbl,
        notes=notes,
    )


# ------------------------------------------------------------
# 4. Non-parametric tests
# ------------------------------------------------------------

def kruskal_wallis(df_long, value_col, group_col):
    """Kruskal-Wallis H + eta²_H 효과크기"""
    _require_scipy()
    groups = _get_groups(df_long, value_col, group_col)
    group_list = list(groups.values())
    group_list = [g for g in group_list if len(g) >= 1]
    if len(group_list) < 2:
        return None
    H, p = sp_stats.kruskal(*group_list)
    k = len(group_list)
    n = sum(len(g) for g in group_list)
    # η²_H = (H - k + 1) / (n - k)
    eta_sq_H = (H - k + 1) / (n - k) if n > k else np.nan
    eta_sq_H = max(0.0, float(eta_sq_H)) if not np.isnan(eta_sq_H) else np.nan

    summary = {
        'H': float(H),
        'df': int(k - 1),
        'p': float(p),
        'n': int(n),
        'k_groups': int(k),
        'eta_sq_H': float(eta_sq_H) if not np.isnan(eta_sq_H) else np.nan,
        'stars': p_to_stars(float(p)),
    }
    notes = []
    if p < 0.05:
        notes.append(f'군간 중앙값에 유의한 차이가 있습니다 (H={H:.3f}, p={_fmt_p(p)}).')
        notes.append('사후검정으로 Dunn 검정을 권장합니다.')
    else:
        notes.append('군간 중앙값에 유의한 차이가 없습니다.')
    # 효과크기 해석
    if not np.isnan(eta_sq_H):
        if eta_sq_H < 0.01:
            notes.append(f'효과크기 η²_H={eta_sq_H:.3f} (매우 작음).')
        elif eta_sq_H < 0.06:
            notes.append(f'효과크기 η²_H={eta_sq_H:.3f} (작음).')
        elif eta_sq_H < 0.14:
            notes.append(f'효과크기 η²_H={eta_sq_H:.3f} (중간).')
        else:
            notes.append(f'효과크기 η²_H={eta_sq_H:.3f} (큼).')

    return AnalysisResult(
        analysis_name='Kruskal-Wallis',
        summary=summary,
        notes=notes,
    )


def dunn_test(df_long, value_col, group_col, p_adjust='bonferroni'):
    """Dunn 사후검정. scikit_posthocs 있으면 사용, 없으면 수동 구현."""
    _require_scipy()
    d = df_long[[value_col, group_col]].dropna().copy()
    names = list(pd.unique(d[group_col]))

    rows = []
    method_name = f'Dunn ({p_adjust})'
    if HAS_SKPH:
        try:
            wide_matrix = sp_ph.posthoc_dunn(d, val_col=value_col,
                                             group_col=group_col,
                                             p_adjust=p_adjust)
            for a, b in combinations(wide_matrix.columns, 2):
                rows.append({
                    'group1': a, 'group2': b,
                    'p_adj': float(wide_matrix.loc[a, b]),
                })
        except Exception:
            rows = []

    if not rows:
        # 수동 구현: rank-sum 기반 Dunn + Bonferroni(혹은 p_adjust)
        all_vals = d[value_col].values.astype(float)
        ranks = sp_stats.rankdata(all_vals)
        d_rank = d.copy()
        d_rank['__rank'] = ranks
        mean_ranks = d_rank.groupby(group_col)['__rank'].mean()
        n_per = d_rank.groupby(group_col).size()
        N = len(d_rank)
        # tie correction
        _, counts = np.unique(all_vals, return_counts=True)
        tie_term = (counts ** 3 - counts).sum()
        sigma_factor = (N * (N + 1) / 12.0) - tie_term / (12.0 * (N - 1))
        raw_ps = []
        pairs = []
        for a, b in combinations(names, 2):
            se = np.sqrt(sigma_factor * (1.0 / n_per[a] + 1.0 / n_per[b]))
            z = (mean_ranks[a] - mean_ranks[b]) / se if se > 0 else 0.0
            p_raw = 2 * (1 - sp_stats.norm.cdf(abs(z)))
            raw_ps.append(p_raw)
            pairs.append((a, b))
        if p_adjust == 'bonferroni':
            adj = np.minimum(1.0, np.array(raw_ps) * len(raw_ps))
        elif p_adjust in ('holm',):
            adj = _apply_correction(np.array(raw_ps), 'holm')
        elif p_adjust == 'sidak':
            adj = _apply_correction(np.array(raw_ps), 'sidak')
        else:
            adj = np.minimum(1.0, np.array(raw_ps) * len(raw_ps))
        for (a, b), p_val in zip(pairs, adj):
            rows.append({'group1': a, 'group2': b, 'p_adj': float(p_val)})

    tbl = pd.DataFrame(rows)
    tbl['stars'] = tbl['p_adj'].apply(lambda p: p_to_stars(float(p)) if not pd.isna(p) else 'ns')
    summary = {
        'method': method_name,
        'p_adjust': p_adjust,
        'n_comparisons': int(len(tbl)),
        'n_significant': int((tbl['p_adj'] < 0.05).sum()),
    }
    notes = [
        f'{row["group1"]} vs {row["group2"]}: 유의 (p_adj={_fmt_p(row["p_adj"])})'
        for _, row in tbl[tbl['p_adj'] < 0.05].iterrows()
    ]
    return AnalysisResult(
        analysis_name='Dunn test',
        summary=summary,
        table=tbl,
        notes=notes,
    )


def mann_whitney(x, y, alternative='two-sided'):
    """Mann-Whitney U (비모수 독립표본)"""
    _require_scipy()
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    y = np.asarray(y, dtype=float); y = y[~np.isnan(y)]
    if len(x) < 1 or len(y) < 1:
        return None
    U, p = sp_stats.mannwhitneyu(x, y, alternative=alternative)
    n1, n2 = len(x), len(y)
    # rank-biserial effect size
    r_rb = 1 - (2 * U) / (n1 * n2)
    summary = {
        'U': float(U),
        'p': float(p),
        'n1': int(n1),
        'n2': int(n2),
        'alternative': alternative,
        'r_rank_biserial': float(r_rb),
        'stars': p_to_stars(float(p)),
    }
    notes = []
    if p < 0.05:
        notes.append(f'두 군의 분포에 유의한 차이 (p={_fmt_p(p)}).')
    else:
        notes.append('두 군의 분포에 유의한 차이 없음.')
    return AnalysisResult(analysis_name='Mann-Whitney U', summary=summary, notes=notes)


def wilcoxon_signed_rank(x, y):
    """Wilcoxon signed-rank (대응표본 비모수)"""
    _require_scipy()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 1:
        return None
    W, p = sp_stats.wilcoxon(x, y)
    summary = {
        'W': float(W),
        'p': float(p),
        'n_pairs': int(len(x)),
        'stars': p_to_stars(float(p)),
    }
    notes = []
    if p < 0.05:
        notes.append(f'대응쌍에서 유의한 차이 (p={_fmt_p(p)}).')
    else:
        notes.append('대응쌍에서 유의한 차이 없음.')
    return AnalysisResult(analysis_name='Wilcoxon signed-rank', summary=summary, notes=notes)


def friedman(df_wide):
    """Friedman 검정 (반복측정 비모수). df_wide: 각 열이 조건/시점."""
    _require_scipy()
    cols = list(df_wide.columns)
    arrs = [df_wide[c].dropna().values for c in cols]
    n_min = min(len(a) for a in arrs) if arrs else 0
    if n_min < 2 or len(arrs) < 3:
        return None
    # 동일 길이로 맞춤 (행 단위 dropna)
    clean = df_wide.dropna()
    if len(clean) < 2:
        return None
    samples = [clean[c].values for c in cols]
    chi2, p = sp_stats.friedmanchisquare(*samples)
    k = len(samples)
    n = len(clean)
    # Kendall's W = chi2 / (n * (k - 1))
    W = chi2 / (n * (k - 1)) if n > 0 and k > 1 else np.nan
    summary = {
        'chi2': float(chi2),
        'df': int(k - 1),
        'p': float(p),
        'n': int(n),
        'k_conditions': int(k),
        'kendall_W': float(W) if not np.isnan(W) else np.nan,
        'stars': p_to_stars(float(p)),
    }
    notes = []
    if p < 0.05:
        notes.append(f'조건 간 유의한 차이 (χ²={chi2:.3f}, p={_fmt_p(p)}).')
        notes.append('사후검정으로 Wilcoxon signed-rank (쌍별) + Bonferroni 권장.')
    else:
        notes.append('조건 간 유의한 차이 없음.')
    return AnalysisResult(analysis_name='Friedman', summary=summary, notes=notes)


# ------------------------------------------------------------
# 5. Normality tests
# ------------------------------------------------------------

def _clean_1d(data):
    arr = np.asarray(data, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def shapiro_wilk(data):
    """Shapiro-Wilk 정규성 검정"""
    _require_scipy()
    arr = _clean_1d(data)
    if len(arr) < 3:
        return None
    W, p = sp_stats.shapiro(arr)
    summary = {'W': float(W), 'p': float(p), 'n': int(len(arr)), 'stars': p_to_stars(float(p))}
    notes = []
    if p < 0.05:
        notes.append('정규성 가정 기각 (p<0.05) → 비모수 검정 고려.')
    else:
        notes.append('정규성 가정을 벗어나지 않음 (p≥0.05).')
    if len(arr) > 5000:
        notes.append('표본이 매우 큼 (n>5000): 사소한 편차도 유의하게 나올 수 있음.')
    return AnalysisResult(analysis_name='Shapiro-Wilk', summary=summary,
                          assumptions={'normality_p': float(p)}, notes=notes)


def dagostino_pearson(data):
    """D'Agostino-Pearson K² 정규성 검정 (skewness + kurtosis)"""
    _require_scipy()
    arr = _clean_1d(data)
    if len(arr) < 8:
        return None
    K2, p = sp_stats.normaltest(arr)
    summary = {'K2': float(K2), 'p': float(p), 'n': int(len(arr)), 'stars': p_to_stars(float(p))}
    notes = []
    if p < 0.05:
        notes.append('정규성 가정 기각 (p<0.05).')
    else:
        notes.append('정규성 가정을 벗어나지 않음 (p≥0.05).')
    return AnalysisResult(analysis_name="D'Agostino-Pearson K²", summary=summary,
                          assumptions={'normality_p': float(p)}, notes=notes)


def anderson_darling(data):
    """Anderson-Darling 정규성 검정"""
    _require_scipy()
    arr = _clean_1d(data)
    if len(arr) < 3:
        return None
    res = sp_stats.anderson(arr, dist='norm')
    # 유의수준별 임계값
    crit_tbl = pd.DataFrame({
        'significance_%': list(res.significance_level),
        'critical_value': list(res.critical_values),
    })
    crit_tbl['rejected'] = res.statistic > crit_tbl['critical_value']
    # 대략적인 p-value (5% 기준으로 판단)
    rejected_5 = bool(res.statistic > res.critical_values[
        list(res.significance_level).index(5.0)
    ]) if 5.0 in list(res.significance_level) else None
    summary = {
        'A2': float(res.statistic),
        'n': int(len(arr)),
        'rejected_at_5pct': rejected_5,
    }
    notes = []
    if rejected_5:
        notes.append('5% 수준에서 정규성 가정 기각.')
    else:
        notes.append('5% 수준에서 정규성 가정을 벗어나지 않음.')
    return AnalysisResult(analysis_name='Anderson-Darling',
                          summary=summary,
                          table=crit_tbl,
                          assumptions={'normality_rejected_5pct': rejected_5},
                          notes=notes)


# ------------------------------------------------------------
# 6. Variance equality tests
# ------------------------------------------------------------

def _clean_groups(args):
    cleaned = []
    for g in args:
        arr = _clean_1d(g)
        if len(arr) >= 2:
            cleaned.append(arr)
    return cleaned


def levene(*groups):
    """Levene 검정 (center='median' = Brown-Forsythe)"""
    _require_scipy()
    cg = _clean_groups(groups)
    if len(cg) < 2:
        return None
    W, p = sp_stats.levene(*cg, center='median')
    summary = {'W': float(W), 'p': float(p), 'k_groups': int(len(cg)),
               'stars': p_to_stars(float(p))}
    notes = []
    if p < 0.05:
        notes.append('등분산 가정 기각 (p<0.05) → Welch ANOVA 또는 비모수 검정 고려.')
    else:
        notes.append('등분산 가정을 벗어나지 않음 (p≥0.05).')
    return AnalysisResult(analysis_name='Levene (Brown-Forsythe)', summary=summary,
                          assumptions={'levene_p': float(p)}, notes=notes)


def bartlett(*groups):
    """Bartlett 등분산 검정 (정규성 전제)"""
    _require_scipy()
    cg = _clean_groups(groups)
    if len(cg) < 2:
        return None
    T, p = sp_stats.bartlett(*cg)
    summary = {'T': float(T), 'p': float(p), 'k_groups': int(len(cg)),
               'stars': p_to_stars(float(p))}
    notes = []
    if p < 0.05:
        notes.append('등분산 가정 기각 (p<0.05).')
    else:
        notes.append('등분산 가정을 벗어나지 않음 (p≥0.05).')
    notes.append('Bartlett은 정규성 위반에 민감합니다. 비정규 자료에는 Levene 권장.')
    return AnalysisResult(analysis_name='Bartlett', summary=summary,
                          assumptions={'bartlett_p': float(p)}, notes=notes)


def fligner(*groups):
    """Fligner-Killeen 등분산 검정 (비모수)"""
    _require_scipy()
    cg = _clean_groups(groups)
    if len(cg) < 2:
        return None
    T, p = sp_stats.fligner(*cg)
    summary = {'T': float(T), 'p': float(p), 'k_groups': int(len(cg)),
               'stars': p_to_stars(float(p))}
    notes = []
    if p < 0.05:
        notes.append('등분산 가정 기각 (p<0.05).')
    else:
        notes.append('등분산 가정을 벗어나지 않음 (p≥0.05).')
    return AnalysisResult(analysis_name='Fligner-Killeen', summary=summary,
                          assumptions={'fligner_p': float(p)}, notes=notes)


# ------------------------------------------------------------
# 7. Correlation
# ------------------------------------------------------------

def _fisher_ci(r, n, alpha=0.05):
    """Fisher z 변환 기반 95% CI"""
    if n < 4 or abs(r) >= 1:
        return (np.nan, np.nan)
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1.0 / np.sqrt(n - 3)
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    lo = z - z_crit * se
    hi = z + z_crit * se
    return (float(np.tanh(lo)), float(np.tanh(hi)))


def correlation(x, y, method='pearson'):
    """상관분석 + 95% CI (Pearson은 Fisher z)"""
    _require_scipy()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return None
    method = method.lower()
    if method == 'pearson':
        r, p = sp_stats.pearsonr(x, y)
        ci = _fisher_ci(r, n)
    elif method == 'spearman':
        r, p = sp_stats.spearmanr(x, y)
        ci = _fisher_ci(r, n)  # 근사
    elif method == 'kendall':
        r, p = sp_stats.kendalltau(x, y)
        ci = (np.nan, np.nan)
    else:
        raise ValueError(f"method는 'pearson', 'spearman', 'kendall' 중 하나여야 합니다.")

    summary = {
        'method': method,
        'r': float(r),
        'p': float(p),
        'n': int(n),
        'ci95_lower': ci[0],
        'ci95_upper': ci[1],
        'r_squared': float(r ** 2),
        'stars': p_to_stars(float(p)),
    }
    notes = []
    abs_r = abs(r)
    if abs_r < 0.1:
        strength = '매우 약함'
    elif abs_r < 0.3:
        strength = '약함'
    elif abs_r < 0.5:
        strength = '중간'
    elif abs_r < 0.7:
        strength = '강함'
    else:
        strength = '매우 강함'
    direction = '양의' if r > 0 else ('음의' if r < 0 else '없음')
    notes.append(f'{direction} {strength} 상관 (r={r:.3f}, p={_fmt_p(p)}).')
    return AnalysisResult(
        analysis_name=f'Correlation ({method.capitalize()})',
        summary=summary,
        notes=notes,
    )


def correlation_matrix(df, method='pearson'):
    """전체 수치형 열에 대한 상관행렬 + p-value 행렬"""
    _require_scipy()
    num = df.select_dtypes(include=[np.number])
    cols = list(num.columns)
    n_cols = len(cols)
    if n_cols < 2:
        return None
    r_mat = np.full((n_cols, n_cols), np.nan)
    p_mat = np.full((n_cols, n_cols), np.nan)
    for i in range(n_cols):
        for j in range(n_cols):
            if i == j:
                r_mat[i, j] = 1.0
                p_mat[i, j] = 0.0
                continue
            x = num[cols[i]].values
            y = num[cols[j]].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 3:
                continue
            if method == 'pearson':
                r, p = sp_stats.pearsonr(x[mask], y[mask])
            elif method == 'spearman':
                r, p = sp_stats.spearmanr(x[mask], y[mask])
            elif method == 'kendall':
                r, p = sp_stats.kendalltau(x[mask], y[mask])
            else:
                raise ValueError(f"method는 'pearson', 'spearman', 'kendall' 중 하나여야 합니다.")
            r_mat[i, j] = r
            p_mat[i, j] = p
    r_df = pd.DataFrame(r_mat, index=cols, columns=cols)
    p_df = pd.DataFrame(p_mat, index=cols, columns=cols)
    # 장방형 표로 결합
    rows = []
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            rows.append({
                'var1': a, 'var2': b,
                'r': r_mat[i, j],
                'p': p_mat[i, j],
                'stars': p_to_stars(float(p_mat[i, j])) if not np.isnan(p_mat[i, j]) else 'ns',
            })
    tbl = pd.DataFrame(rows)
    summary = {
        'method': method,
        'n_variables': int(n_cols),
        'n_pairs': int(len(tbl)),
        'n_significant': int((tbl['p'] < 0.05).sum()) if len(tbl) else 0,
    }
    notes = []
    strongest = tbl.iloc[tbl['r'].abs().idxmax()] if len(tbl) else None
    if strongest is not None:
        notes.append(f'가장 강한 상관: {strongest["var1"]}–{strongest["var2"]} '
                     f'(r={strongest["r"]:.3f}, p={_fmt_p(strongest["p"])})')
    return AnalysisResult(
        analysis_name=f'Correlation matrix ({method.capitalize()})',
        summary=summary,
        table=tbl,
        notes=notes,
    )


# ------------------------------------------------------------
# 8. Effect sizes
# ------------------------------------------------------------

def cohens_d(x, y, paired=False, hedges=False):
    """Cohen's d 효과크기 (Hedges' g 보정 옵션)"""
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    y = np.asarray(y, dtype=float); y = y[~np.isnan(y)]
    if paired:
        n = min(len(x), len(y))
        if n < 2:
            return None
        x, y = x[:n], y[:n]
        diff = x - y
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else np.nan
        df_v = n - 1
        n_total = n
        method = "Cohen's d (paired)"
    else:
        n1, n2 = len(x), len(y)
        if n1 < 2 or n2 < 2:
            return None
        s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
        sp2 = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
        if sp2 <= 0:
            d = np.nan
        else:
            d = (np.mean(x) - np.mean(y)) / np.sqrt(sp2)
        df_v = n1 + n2 - 2
        n_total = n1 + n2
        method = "Cohen's d (independent, pooled SD)"

    # Hedges' g = d * (1 - 3/(4*df - 1))
    if hedges and not np.isnan(d):
        J = 1 - 3.0 / (4 * df_v - 1)
        g = d * J
    else:
        g = np.nan

    abs_d = abs(d) if not np.isnan(d) else np.nan
    if np.isnan(abs_d):
        interp = 'N/A'
    elif abs_d < 0.2:
        interp = '무시할 수준'
    elif abs_d < 0.5:
        interp = '작은 효과'
    elif abs_d < 0.8:
        interp = '중간 효과'
    else:
        interp = '큰 효과'

    summary = {
        'method': method,
        'd': float(d),
        'hedges_g': float(g) if not np.isnan(g) else np.nan,
        'n_total': int(n_total),
        'interpretation': interp,
    }
    notes = [f"Cohen's d = {_fmt_num(d)} ({interp})."]
    if hedges:
        notes.append(f"Hedges' g = {_fmt_num(g)} (소표본 보정).")
    return AnalysisResult(analysis_name="Cohen's d", summary=summary, notes=notes)


# ------------------------------------------------------------
# 9. auto_compare — 정규성/등분산 검사 후 자동 선택
# ------------------------------------------------------------

def auto_compare(df_long, value_col, group_col, alpha=0.05):
    """
    정규성 + 등분산 검정 후 자동으로 파라메트릭(ANOVA+Tukey) 또는
    비모수(Kruskal+Dunn) 분석을 실행.
    """
    _require_scipy()
    d = df_long[[value_col, group_col]].dropna()
    if d[group_col].nunique() < 2:
        return None
    groups = _get_groups(df_long, value_col, group_col)
    group_arrays = list(groups.values())

    # 1) 정규성 (군별 Shapiro-Wilk)
    normality_ok = True
    normality_ps = {}
    for name, g in groups.items():
        if len(g) < 3:
            continue
        if len(g) <= 5000:
            _, p_n = sp_stats.shapiro(g)
        else:
            _, p_n = sp_stats.normaltest(g)
        normality_ps[name] = float(p_n)
        if p_n < alpha:
            normality_ok = False

    # 2) 등분산 (Levene, 중앙값 기준 = Brown-Forsythe)
    try:
        _, levene_p = sp_stats.levene(*group_arrays, center='median')
        levene_p = float(levene_p)
    except Exception:
        levene_p = np.nan
    equal_var = (not np.isnan(levene_p)) and (levene_p >= alpha)

    choice_notes = []
    if normality_ok and equal_var:
        # 파라메트릭 경로: One-way ANOVA → Tukey
        choice_notes.append('정규성 및 등분산 가정 충족 → 파라메트릭 경로.')
        aov = one_way_anova_long(df_long, value_col, group_col)
        post = tukey_hsd(df_long, value_col, group_col) if HAS_STATSMODELS else None
        result = aov if post is None else post
        # 통합
        final = AnalysisResult(
            analysis_name='Auto-compare → One-way ANOVA + Tukey HSD',
            summary={**(aov.summary if aov else {}),
                     **({'posthoc_' + k: v for k, v in (post.summary if post else {}).items()})},
            table=post.table if post else (aov.table if aov else None),
            assumptions={
                'normality_ok': normality_ok,
                'levene_p': levene_p,
                **{f'shapiro_p[{k}]': v for k, v in normality_ps.items()},
            },
            notes=choice_notes + (aov.notes if aov else []) + (post.notes if post else []),
        )
        if post is None and HAS_STATSMODELS is False:
            final.notes.append('statsmodels 미설치 → Tukey HSD 생략. pip install statsmodels 권장.')
        return final
    else:
        reasons = []
        if not normality_ok:
            reasons.append('정규성 위반')
        if not equal_var:
            reasons.append('등분산 위반')
        choice_notes.append(f'{" + ".join(reasons)} → 비모수 경로 (Kruskal-Wallis + Dunn).')
        kw = kruskal_wallis(df_long, value_col, group_col)
        post = None
        if kw is not None and kw.summary.get('p', 1.0) < alpha:
            try:
                post = dunn_test(df_long, value_col, group_col, p_adjust='bonferroni')
            except Exception:
                post = None
        final = AnalysisResult(
            analysis_name='Auto-compare → Kruskal-Wallis + Dunn (Bonferroni)',
            summary={**(kw.summary if kw else {}),
                     **({'posthoc_' + k: v for k, v in (post.summary if post else {}).items()})},
            table=post.table if post else None,
            assumptions={
                'normality_ok': normality_ok,
                'levene_p': levene_p,
                **{f'shapiro_p[{k}]': v for k, v in normality_ps.items()},
            },
            notes=choice_notes + (kw.notes if kw else []) + (post.notes if post else []),
        )
        return final


def one_way_anova_long(df_long, value_col, group_col):
    """long-format용 one-way ANOVA → AnalysisResult"""
    _require_scipy()
    groups = _get_groups(df_long, value_col, group_col)
    group_list = [g for g in groups.values() if len(g) >= 2]
    if len(group_list) < 2:
        return None
    F, p = sp_stats.f_oneway(*group_list)
    k = len(group_list)
    n = sum(len(g) for g in group_list)
    df_between = k - 1
    df_within = n - k
    # η² = SSB / SST
    grand = np.concatenate(group_list).mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in group_list)
    sst = sum(((g - grand) ** 2).sum() for g in group_list)
    eta_sq = ssb / sst if sst > 0 else np.nan

    tbl = pd.DataFrame([
        {'source': 'Between', 'SS': float(ssb), 'df': int(df_between),
         'MS': float(ssb / df_between) if df_between > 0 else np.nan,
         'F': float(F), 'p': float(p)},
        {'source': 'Within', 'SS': float(sst - ssb), 'df': int(df_within),
         'MS': float((sst - ssb) / df_within) if df_within > 0 else np.nan,
         'F': np.nan, 'p': np.nan},
        {'source': 'Residual', 'SS': float(sst - ssb), 'df': int(df_within),
         'MS': float((sst - ssb) / df_within) if df_within > 0 else np.nan,
         'F': np.nan, 'p': np.nan},
    ])

    summary = {
        'F': float(F),
        'p': float(p),
        'df': (int(df_between), int(df_within)),
        'k_groups': int(k),
        'n_total': int(n),
        'eta_sq': float(eta_sq) if not np.isnan(eta_sq) else np.nan,
        'stars': p_to_stars(float(p)),
    }
    notes = []
    if p < 0.05:
        notes.append(f'군간 평균에 유의한 차이 (F({df_between},{df_within})={F:.3f}, p={_fmt_p(p)}).')
    else:
        notes.append('군간 평균에 유의한 차이 없음.')
    return AnalysisResult(analysis_name='One-way ANOVA', summary=summary,
                          table=tbl, notes=notes)
