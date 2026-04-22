"""비선형 곡선 모델 라이브러리 (농약/생물 연구용).

Phase 3: 용량-반응, 효소반응, 성장/감쇠, 피크 피팅.
각 모델은 CurveModel을 상속하며 공통 FitResult API로 결과를 반환.
scipy.optimize.curve_fit 기반 Levenberg-Marquardt 피팅.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import pandas as pd

try:
    from scipy.optimize import curve_fit
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


_NO_SCIPY_MSG = "scipy가 필요합니다: pip install scipy"


# ------------------------------------------------------------------
# FitResult
# ------------------------------------------------------------------
@dataclass
class FitResult:
    """비선형 피팅 결과 컨테이너.

    params, param_se, param_ci_95는 파라미터 이름을 키로 하는 dict.
    derived는 EC50 같은 파생 수치를 담음. Korean 메시지는 GUI 표시용.
    """
    model_name: str = ''
    params: dict = field(default_factory=dict)
    param_se: dict = field(default_factory=dict)
    param_ci_95: dict = field(default_factory=dict)
    r_squared: float = float('nan')
    adjusted_r2: float = float('nan')
    rmse: float = float('nan')
    aicc: float = float('nan')
    bic: float = float('nan')
    n: int = 0
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    predicted_y: np.ndarray = field(default_factory=lambda: np.array([]))
    covariance: Optional[np.ndarray] = None
    converged: bool = False
    message: str = ''
    derived: dict = field(default_factory=dict)

    # 예측 함수 및 파라미터 순서 — fit()에서 채워짐
    _model_fn: Optional[Callable] = None
    _param_order: list = field(default_factory=list)
    _x_data: Optional[np.ndarray] = None
    _y_data: Optional[np.ndarray] = None

    def predict(self, x):
        """적합된 파라미터로 x에서 y를 예측."""
        if self._model_fn is None or not self._param_order:
            return np.full_like(np.asarray(x, dtype=float), np.nan)
        p = [self.params[k] for k in self._param_order]
        return self._model_fn(np.asarray(x, dtype=float), *p)

    def predict_ci(self, x, alpha=0.05):
        """델타 방법 기반 예측값의 신뢰구간 (lo, hi).

        공분산 행렬이 없으면 (predict, predict) 반환.
        수치미분으로 J 계산.
        """
        x_arr = np.asarray(x, dtype=float)
        y_hat = self.predict(x_arr)
        if (self.covariance is None or self._model_fn is None
                or not self._param_order or self.n <= len(self._param_order)):
            return y_hat, y_hat

        p = np.array([self.params[k] for k in self._param_order], dtype=float)
        eps = 1e-6
        # J: shape (len(x), k)
        J = np.zeros((len(x_arr), len(p)))
        for i in range(len(p)):
            dp = np.zeros_like(p)
            step = eps * max(1.0, abs(p[i]))
            dp[i] = step
            y_plus = self._model_fn(x_arr, *(p + dp))
            y_minus = self._model_fn(x_arr, *(p - dp))
            J[:, i] = (y_plus - y_minus) / (2 * step)

        var = np.einsum('ij,jk,ik->i', J, self.covariance, J)
        var = np.clip(var, 0.0, None)
        se = np.sqrt(var)
        dof = max(self.n - len(p), 1)
        if HAS_SCIPY:
            tval = float(sp_stats.t.ppf(1 - alpha / 2, dof))
        else:
            tval = 1.96
        return y_hat - tval * se, y_hat + tval * se

    def to_text(self) -> str:
        """한국어 보고서 문자열 생성."""
        lines = [f"모델: {self.model_name}"]
        if not self.converged:
            lines.append(f"피팅 실패: {self.message}")
            return '\n'.join(lines)
        lines.append(f"데이터 포인트 n = {self.n}")
        lines.append(f"R² = {self.r_squared:.4f}, adj.R² = {self.adjusted_r2:.4f}")
        lines.append(f"RMSE = {self.rmse:.4g}")
        lines.append(f"AICc = {self.aicc:.3f}, BIC = {self.bic:.3f}")
        lines.append("파라미터:")
        for name in self._param_order or list(self.params.keys()):
            val = self.params.get(name, float('nan'))
            se = self.param_se.get(name, float('nan'))
            ci = self.param_ci_95.get(name, (float('nan'), float('nan')))
            lines.append(f"  {name} = {val:.4g} ± {se:.4g}  (95% CI [{ci[0]:.4g}, {ci[1]:.4g}])")
        if self.derived:
            lines.append("파생 수치:")
            for k, v in self.derived.items():
                if isinstance(v, tuple) and len(v) == 2:
                    lines.append(f"  {k} = [{v[0]:.4g}, {v[1]:.4g}]")
                else:
                    try:
                        lines.append(f"  {k} = {float(v):.4g}")
                    except (TypeError, ValueError):
                        lines.append(f"  {k} = {v}")
        if self.message:
            lines.append(f"메시지: {self.message}")
        return '\n'.join(lines)


# ------------------------------------------------------------------
# CurveModel (abstract base)
# ------------------------------------------------------------------
class CurveModel(ABC):
    """모든 곡선 모델의 추상 기반.

    서브클래스는 name/param_names/equation/description을 클래스 속성으로,
    그리고 model() / initial_guess()를 구현해야 함.
    """
    name: str = ''
    param_names: list = []
    equation: str = ''
    description: str = ''

    @abstractmethod
    def model(self, x, *params):
        """순수 함수: x와 풀린 파라미터 → y."""
        raise NotImplementedError

    @abstractmethod
    def initial_guess(self, x, y) -> list:
        """데이터 패턴으로부터 초기값 자동 추정."""
        raise NotImplementedError

    def bounds(self, x, y):
        """파라미터 경계. 기본값은 제약 없음."""
        return (-np.inf, np.inf)

    def compute_derived(self, params_dict) -> dict:
        """EC50 등 모델 특화 파생 수치를 계산."""
        return {}

    # ------ 내부 유틸 ------
    def _as_arrays(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask]

    def _failed_result(self, msg: str, n: int = 0) -> FitResult:
        return FitResult(
            model_name=self.name,
            converged=False,
            message=msg,
            n=n,
            _param_order=list(self.param_names),
            _model_fn=self.model,
        )

    def _delta_ci_for(self, fn_of_params, params_vec, cov, dof, alpha=0.05):
        """델타 방법으로 스칼라 함수 g(params)의 95% CI 계산.
        fn_of_params: callable(params_vec) -> float
        반환: (value, se, (lo, hi))
        """
        value = float(fn_of_params(params_vec))
        if cov is None or not np.all(np.isfinite(cov)):
            return value, float('nan'), (float('nan'), float('nan'))
        grad = np.zeros_like(params_vec)
        eps = 1e-6
        for i in range(len(params_vec)):
            step = eps * max(1.0, abs(params_vec[i]))
            dp = np.zeros_like(params_vec)
            dp[i] = step
            try:
                vp = float(fn_of_params(params_vec + dp))
                vm = float(fn_of_params(params_vec - dp))
                grad[i] = (vp - vm) / (2 * step)
            except Exception:
                grad[i] = 0.0
        var = float(grad @ cov @ grad)
        if not np.isfinite(var) or var < 0:
            return value, float('nan'), (float('nan'), float('nan'))
        se = math.sqrt(var)
        if HAS_SCIPY and dof > 0:
            tval = float(sp_stats.t.ppf(1 - alpha / 2, dof))
        else:
            tval = 1.96
        return value, se, (value - tval * se, value + tval * se)

    # ------ 메인 fit ------
    def fit(self, x, y, weights=None, maxfev=5000) -> FitResult:
        """비선형 최소자승 피팅.

        weights: 각 점의 sigma (inverse-variance 가중). None이면 단위가중.
        반환: 전체 메트릭을 채운 FitResult.
        """
        if not HAS_SCIPY:
            return self._failed_result(_NO_SCIPY_MSG)

        x_arr, y_arr = self._as_arrays(x, y)
        n = len(x_arr)
        k = len(self.param_names)

        if n < k:
            return self._failed_result(
                "데이터 포인트가 파라미터보다 적습니다", n=n)
        if n < 2:
            return self._failed_result("데이터가 너무 적습니다", n=n)

        try:
            p0 = self.initial_guess(x_arr, y_arr)
        except Exception as e:
            return self._failed_result(f"초기값 추정 실패: {e}", n=n)

        try:
            bds = self.bounds(x_arr, y_arr)
        except Exception:
            bds = (-np.inf, np.inf)

        sigma = None
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if len(w) == len(y):
                # weights와 y를 동시에 필터
                full_mask = np.isfinite(np.asarray(x, dtype=float)) & np.isfinite(np.asarray(y, dtype=float))
                w = w[full_mask]
            if len(w) == n and np.all(w > 0):
                sigma = w

        absolute_sigma = sigma is not None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    self.model, x_arr, y_arr,
                    p0=p0, bounds=bds, maxfev=maxfev,
                    sigma=sigma, absolute_sigma=absolute_sigma,
                )
        except (RuntimeError, ValueError, TypeError) as e:
            return self._failed_result(f"수렴 실패: {e}", n=n)
        except Exception as e:
            return self._failed_result(f"피팅 오류: {e}", n=n)

        if pcov is None or not np.all(np.isfinite(pcov)):
            # 공분산이 계산되지 않은 경우 — 결과는 반환하되 SE는 NaN
            pcov_use = None
        else:
            pcov_use = np.array(pcov)

        # 예측값 / 잔차
        try:
            y_pred = self.model(x_arr, *popt)
        except Exception as e:
            return self._failed_result(f"예측 오류: {e}", n=n)

        residuals = y_arr - y_pred
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        dof = max(n - k, 1)
        adj_r2 = 1 - (1 - r2) * (n - 1) / dof if (n - k - 1) > 0 and np.isfinite(r2) else float('nan')
        rmse = math.sqrt(ss_res / n) if n > 0 else float('nan')

        # AIC / AICc / BIC
        if ss_res > 0 and n > 0:
            aic = n * math.log(ss_res / n) + 2 * k
            if n - k - 1 > 0:
                aicc = aic + 2 * k * (k + 1) / (n - k - 1)
            else:
                aicc = float('inf')
            bic = n * math.log(ss_res / n) + k * math.log(n)
        else:
            aic = aicc = bic = float('nan')

        # 파라미터 dict / SE / CI
        params_dict = {name: float(popt[i]) for i, name in enumerate(self.param_names)}
        if pcov_use is not None:
            diag = np.diag(pcov_use)
            diag = np.where(diag < 0, np.nan, diag)
            se_vec = np.sqrt(diag)
        else:
            se_vec = np.full(k, float('nan'))
        se_dict = {name: float(se_vec[i]) for i, name in enumerate(self.param_names)}

        if HAS_SCIPY and dof > 0:
            tval = float(sp_stats.t.ppf(0.975, dof))
        else:
            tval = 1.96
        ci_dict = {}
        for i, name in enumerate(self.param_names):
            se = se_vec[i]
            if np.isfinite(se):
                ci_dict[name] = (float(popt[i] - tval * se), float(popt[i] + tval * se))
            else:
                ci_dict[name] = (float('nan'), float('nan'))

        result = FitResult(
            model_name=self.name,
            params=params_dict,
            param_se=se_dict,
            param_ci_95=ci_dict,
            r_squared=float(r2) if np.isfinite(r2) else float('nan'),
            adjusted_r2=float(adj_r2) if np.isfinite(adj_r2) else float('nan'),
            rmse=float(rmse),
            aicc=float(aicc) if np.isfinite(aicc) else float('nan'),
            bic=float(bic) if np.isfinite(bic) else float('nan'),
            n=n,
            residuals=residuals,
            predicted_y=y_pred,
            covariance=pcov_use,
            converged=True,
            message="성공",
            _model_fn=self.model,
            _param_order=list(self.param_names),
            _x_data=x_arr,
            _y_data=y_arr,
        )

        # 파생 수치
        try:
            derived = self.compute_derived_with_ci(
                params_dict, popt, pcov_use, dof)
        except Exception as e:
            derived = {'_derived_error': str(e)}
        result.derived = derived
        return result

    def compute_derived_with_ci(self, params_dict, params_vec, cov, dof):
        """기본 구현: compute_derived() 호출. 서브클래스가 델타 방법을 쓰려면 오버라이드."""
        return self.compute_derived(params_dict)


# ------------------------------------------------------------------
# Dose-Response Models
# ------------------------------------------------------------------
def _heuristic_log_ec50(x, y):
    """중간값에 가장 가까운 X를 EC50의 초기 추정치로 사용."""
    y_min, y_max = float(np.min(y)), float(np.max(y))
    mid = (y_min + y_max) / 2
    idx = int(np.argmin(np.abs(y - mid)))
    return float(x[idx])


class Hill4PL(CurveModel):
    """4-parameter Hill (로지스틱). X는 log10(dose)."""
    name = "Hill 4PL"
    param_names = ['bottom', 'top', 'logEC50', 'hillslope']
    equation = "Y = bottom + (top - bottom) / (1 + 10^((logEC50 - X) * hillslope))"
    description = "농약 용량-반응(log-dose 스케일)의 표준 4PL. X는 log10(dose)."

    def model(self, x, bottom, top, logEC50, hillslope):
        return bottom + (top - bottom) / (1.0 + 10.0 ** ((logEC50 - x) * hillslope))

    def initial_guess(self, x, y):
        return [float(np.min(y)), float(np.max(y)),
                _heuristic_log_ec50(x, y), 1.0]

    def bounds(self, x, y):
        y_range = float(np.max(y) - np.min(y)) + 1.0
        lo = [float(np.min(y)) - 10 * y_range, float(np.min(y)) - 10 * y_range,
              float(np.min(x)) - 10, -20.0]
        hi = [float(np.max(y)) + 10 * y_range, float(np.max(y)) + 10 * y_range,
              float(np.max(x)) + 10, 20.0]
        return (lo, hi)

    def compute_derived_with_ci(self, params_dict, params_vec, cov, dof):
        log_ec50 = params_dict['logEC50']
        ec50 = 10.0 ** log_ec50

        # logEC50의 CI → EC50 CI (단조변환)
        se_log = math.nan
        if cov is not None:
            idx = self.param_names.index('logEC50')
            var = cov[idx, idx]
            if np.isfinite(var) and var >= 0:
                se_log = math.sqrt(var)

        if HAS_SCIPY and dof > 0:
            tval = float(sp_stats.t.ppf(0.975, dof))
        else:
            tval = 1.96

        if np.isfinite(se_log):
            lo = 10.0 ** (log_ec50 - tval * se_log)
            hi = 10.0 ** (log_ec50 + tval * se_log)
        else:
            lo, hi = float('nan'), float('nan')

        return {
            'EC50': ec50,
            'EC50_ci95': (lo, hi),
            'pEC50': -log_ec50,
            'hill': params_dict['hillslope'],
        }


class Hill5PL(CurveModel):
    """5PL: Hill 4PL + 비대칭 파라미터 S."""
    name = "Hill 5PL"
    param_names = ['bottom', 'top', 'logEC50', 'hillslope', 'S']
    equation = ("Y = bottom + (top - bottom) / "
                "(1 + 10^((logEC50 - X) * hillslope))^S")
    description = "비대칭 용량-반응. 상단/하단 수렴속도가 다를 때."

    def model(self, x, bottom, top, logEC50, hillslope, S):
        denom = (1.0 + 10.0 ** ((logEC50 - x) * hillslope)) ** S
        return bottom + (top - bottom) / denom

    def initial_guess(self, x, y):
        return [float(np.min(y)), float(np.max(y)),
                _heuristic_log_ec50(x, y), 1.0, 1.0]

    def bounds(self, x, y):
        y_range = float(np.max(y) - np.min(y)) + 1.0
        lo = [float(np.min(y)) - 10 * y_range, float(np.min(y)) - 10 * y_range,
              float(np.min(x)) - 10, -20.0, 0.01]
        hi = [float(np.max(y)) + 10 * y_range, float(np.max(y)) + 10 * y_range,
              float(np.max(x)) + 10, 20.0, 100.0]
        return (lo, hi)

    def compute_derived_with_ci(self, params_dict, params_vec, cov, dof):
        # 5PL에서 실제 EC50(절반 반응점)은 logEC50_eff = logEC50 + log10(2^(1/S) - 1)/hill
        bottom = params_dict['bottom']
        top = params_dict['top']
        logEC50 = params_dict['logEC50']
        hill = params_dict['hillslope']
        S = params_dict['S']
        try:
            adj = math.log10(2.0 ** (1.0 / S) - 1.0) / hill
            ec50_eff = 10.0 ** (logEC50 + adj)
        except Exception:
            ec50_eff = 10.0 ** logEC50
        return {
            'EC50_apparent': 10.0 ** logEC50,
            'EC50_effective': ec50_eff,
            'asymmetry_S': S,
            'span': top - bottom,
        }


class Probit(CurveModel):
    """Probit model: Y = 100 * Phi((X - mu)/sigma).

    X는 log-dose 또는 원 dose. LC50 = mu.
    """
    name = "Probit"
    param_names = ['mu', 'sigma']
    equation = "Y = 100 * Phi((X - mu) / sigma)"
    description = "독성학에서 사망률(%) 자료의 표준 probit 모형. LC50 = mu."

    def model(self, x, mu, sigma):
        if not HAS_SCIPY:
            # fallback: 정규 CDF 근사
            z = (x - mu) / sigma
            return 100.0 * 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2)))
        return 100.0 * sp_stats.norm.cdf((x - mu) / sigma)

    def initial_guess(self, x, y):
        # y가 0~100 범위의 퍼센트라고 가정
        mu0 = _heuristic_log_ec50(x, np.clip(y, 0, 100))
        sigma0 = max(float(np.std(x)), 0.1)
        return [mu0, sigma0]

    def bounds(self, x, y):
        x_range = float(np.max(x) - np.min(x)) + 1.0
        return ([float(np.min(x)) - 10 * x_range, 1e-6],
                [float(np.max(x)) + 10 * x_range, 100 * x_range])

    def compute_derived_with_ci(self, params_dict, params_vec, cov, dof):
        mu = params_dict['mu']
        sigma = params_dict['sigma']
        # LC50 = mu (같은 단위)
        se_mu = math.nan
        if cov is not None:
            var = cov[0, 0]
            if np.isfinite(var) and var >= 0:
                se_mu = math.sqrt(var)
        if HAS_SCIPY and dof > 0:
            tval = float(sp_stats.t.ppf(0.975, dof))
        else:
            tval = 1.96
        if np.isfinite(se_mu):
            ci = (mu - tval * se_mu, mu + tval * se_mu)
        else:
            ci = (float('nan'), float('nan'))
        return {
            'LC50': mu,
            'LC50_ci95': ci,
            'slope': 1.0 / sigma if sigma != 0 else float('nan'),
        }


class Logit(CurveModel):
    """로지스틱 (log-dose). Y = 100 / (1 + exp(-b*(X - X50)))."""
    name = "Logit"
    param_names = ['X50', 'b']
    equation = "Y = 100 / (1 + exp(-b * (X - X50)))"
    description = "log-dose 스케일의 로지스틱 회귀. X50은 50% 반응 지점."

    def model(self, x, X50, b):
        # 오버플로 방지
        z = -b * (x - X50)
        z = np.clip(z, -500, 500)
        return 100.0 / (1.0 + np.exp(z))

    def initial_guess(self, x, y):
        X50_0 = _heuristic_log_ec50(x, np.clip(y, 0, 100))
        return [X50_0, 1.0]

    def bounds(self, x, y):
        x_range = float(np.max(x) - np.min(x)) + 1.0
        return ([float(np.min(x)) - 10 * x_range, -50.0],
                [float(np.max(x)) + 10 * x_range, 50.0])

    def compute_derived_with_ci(self, params_dict, params_vec, cov, dof):
        X50 = params_dict['X50']
        se = math.nan
        if cov is not None and np.isfinite(cov[0, 0]) and cov[0, 0] >= 0:
            se = math.sqrt(cov[0, 0])
        if HAS_SCIPY and dof > 0:
            tval = float(sp_stats.t.ppf(0.975, dof))
        else:
            tval = 1.96
        if np.isfinite(se):
            ci = (X50 - tval * se, X50 + tval * se)
            # EC50 = 10^X50 (X가 log10-dose인 경우)
            ec50 = 10.0 ** X50
            ec50_ci = (10.0 ** ci[0], 10.0 ** ci[1])
        else:
            ci = (float('nan'), float('nan'))
            ec50 = 10.0 ** X50
            ec50_ci = (float('nan'), float('nan'))
        return {
            'X50': X50,
            'X50_ci95': ci,
            'EC50_if_logX': ec50,
            'EC50_if_logX_ci95': ec50_ci,
            'slope': params_dict['b'],
        }


class LogLogistic4P(CurveModel):
    """Log-logistic 4-parameter (독성학에서 일반적인 파라미터화).

    Y = c + (d - c) / (1 + exp(b * (log(X) - log(e))))
    여기서 X는 원 dose (로그 아님). 음수 b는 증가 곡선.
    """
    name = "LogLogistic 4PL"
    param_names = ['b', 'c', 'd', 'e']
    equation = "Y = c + (d - c) / (1 + exp(b * (log(X) - log(e))))"
    description = "독성학에서 LL.4로 알려진 파라미터화. e는 ED50."

    def model(self, x, b, c, d, e):
        # x는 원 dose(양수). log 도메인에서는 x=log10(dose)처럼 들어올 수 있으므로
        # 여기서는 x가 양수라고 가정하지 않고, 스칼라 e와 직접 비교되는 형태를 쓴다.
        # 일반화된 형태: Y = c + (d - c) / (1 + exp(b * (x - e')))로 동작.
        z = b * (x - e)
        z = np.clip(z, -500, 500)
        return c + (d - c) / (1.0 + np.exp(z))

    def initial_guess(self, x, y):
        # b: 기울기 부호. y가 증가면 b<0, 감소면 b>0 (LL.4 관례)
        y_sorted = y[np.argsort(x)]
        if y_sorted[-1] > y_sorted[0]:
            b0 = -1.0
        else:
            b0 = 1.0
        return [b0, float(np.min(y)), float(np.max(y)),
                _heuristic_log_ec50(x, y)]

    def bounds(self, x, y):
        y_range = float(np.max(y) - np.min(y)) + 1.0
        x_range = float(np.max(x) - np.min(x)) + 1.0
        return ([-50.0,
                 float(np.min(y)) - 10 * y_range,
                 float(np.min(y)) - 10 * y_range,
                 float(np.min(x)) - 10 * x_range],
                [50.0,
                 float(np.max(y)) + 10 * y_range,
                 float(np.max(y)) + 10 * y_range,
                 float(np.max(x)) + 10 * x_range])

    def compute_derived_with_ci(self, params_dict, params_vec, cov, dof):
        e = params_dict['e']
        # e가 x-도메인 값. x가 log10(dose)면 EC50 = 10^e
        se = math.nan
        if cov is not None:
            idx = self.param_names.index('e')
            var = cov[idx, idx]
            if np.isfinite(var) and var >= 0:
                se = math.sqrt(var)
        if HAS_SCIPY and dof > 0:
            tval = float(sp_stats.t.ppf(0.975, dof))
        else:
            tval = 1.96
        if np.isfinite(se):
            e_ci = (e - tval * se, e + tval * se)
        else:
            e_ci = (float('nan'), float('nan'))
        return {
            'ED50': e,
            'ED50_ci95': e_ci,
            'slope_b': params_dict['b'],
            'span': params_dict['d'] - params_dict['c'],
        }


# ------------------------------------------------------------------
# Enzyme Kinetics
# ------------------------------------------------------------------
class MichaelisMenten(CurveModel):
    """Michaelis-Menten: Y = Vmax * X / (Km + X)."""
    name = "Michaelis-Menten"
    param_names = ['Vmax', 'Km']
    equation = "Y = Vmax * X / (Km + X)"
    description = "효소반응 속도론의 기본 모형. X는 기질 농도."

    def model(self, x, Vmax, Km):
        return Vmax * x / (Km + x)

    def initial_guess(self, x, y):
        Vmax0 = float(np.max(y)) * 1.1 if np.max(y) > 0 else 1.0
        # Km: Vmax/2 지점의 x
        half = Vmax0 / 2
        idx = int(np.argmin(np.abs(y - half)))
        Km0 = max(float(x[idx]), 1e-3)
        return [Vmax0, Km0]

    def bounds(self, x, y):
        return ([0.0, 0.0], [np.inf, np.inf])

    def compute_derived_with_ci(self, params_dict, params_vec, cov, dof):
        return {
            'Vmax': params_dict['Vmax'],
            'Km': params_dict['Km'],
            'Vmax_over_Km': (params_dict['Vmax'] / params_dict['Km']
                             if params_dict['Km'] > 0 else float('nan')),
        }


class MichaelisMentenComp(CurveModel):
    """경쟁적 억제: Y = Vmax * X / (Km * (1 + I/Ki) + X).

    X는 [S], 모델 내에 상수 I(억제제 농도)가 있어야 하지만
    단일 피팅 API로는 fit 시점에 고정값을 알 수 없으므로 파라미터로 취급.
    """
    name = "Michaelis-Menten (competitive)"
    param_names = ['Vmax', 'Km', 'alpha']  # alpha = 1 + [I]/Ki
    equation = "Y = Vmax * X / (Km * alpha + X)"
    description = "경쟁적 억제. alpha = 1 + [I]/Ki — 억제 강도 지표."

    def model(self, x, Vmax, Km, alpha):
        return Vmax * x / (Km * alpha + x)

    def initial_guess(self, x, y):
        Vmax0 = float(np.max(y)) * 1.1 if np.max(y) > 0 else 1.0
        half = Vmax0 / 2
        idx = int(np.argmin(np.abs(y - half)))
        Km0 = max(float(x[idx]), 1e-3)
        return [Vmax0, Km0, 1.0]

    def bounds(self, x, y):
        return ([0.0, 0.0, 0.1], [np.inf, np.inf, 1e6])

    def compute_derived_with_ci(self, params_dict, params_vec, cov, dof):
        return {
            'Vmax': params_dict['Vmax'],
            'Km_apparent': params_dict['Km'] * params_dict['alpha'],
            'alpha': params_dict['alpha'],
        }


# ------------------------------------------------------------------
# Growth / Decay
# ------------------------------------------------------------------
class ExponentialGrowth(CurveModel):
    """Y = Y0 * exp(k * X)."""
    name = "Exponential Growth"
    param_names = ['Y0', 'k']
    equation = "Y = Y0 * exp(k * X)"
    description = "지수 성장. k > 0."

    def model(self, x, Y0, k):
        z = np.clip(k * x, -500, 500)
        return Y0 * np.exp(z)

    def initial_guess(self, x, y):
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        Y0_0 = float(ys[0]) if ys[0] > 0 else max(float(np.min(y)), 1e-3)
        if Y0_0 <= 0 or ys[-1] <= 0 or xs[-1] == xs[0]:
            k0 = 0.1
        else:
            k0 = math.log(ys[-1] / Y0_0) / (xs[-1] - xs[0])
        return [Y0_0, k0]

    def compute_derived_with_ci(self, params_dict, params_vec, cov, dof):
        k = params_dict['k']
        return {
            'doubling_time': math.log(2.0) / k if k > 0 else float('nan'),
            'rate_k': k,
        }


class ExponentialDecay(CurveModel):
    """Y = Y0 * exp(-k * X)."""
    name = "Exponential Decay"
    param_names = ['Y0', 'k']
    equation = "Y = Y0 * exp(-k * X)"
    description = "지수 감쇠. k > 0."

    def model(self, x, Y0, k):
        z = np.clip(-k * x, -500, 500)
        return Y0 * np.exp(z)

    def initial_guess(self, x, y):
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        Y0_0 = float(ys[0]) if ys[0] > 0 else max(float(np.max(y)), 1e-3)
        if Y0_0 <= 0 or ys[-1] <= 0 or xs[-1] == xs[0]:
            k0 = 0.1
        else:
            ratio = ys[-1] / Y0_0
            ratio = max(ratio, 1e-6)
            k0 = -math.log(ratio) / (xs[-1] - xs[0])
            if k0 <= 0:
                k0 = 0.1
        return [Y0_0, k0]

    def bounds(self, x, y):
        return ([0.0, 0.0], [np.inf, np.inf])

    def compute_derived_with_ci(self, params_dict, params_vec, cov, dof):
        k = params_dict['k']
        return {
            'half_life': math.log(2.0) / k if k > 0 else float('nan'),
            'rate_k': k,
        }


class OnePhaseDecay(CurveModel):
    """Y = plateau + (Y0 - plateau) * exp(-k * X)."""
    name = "One-Phase Decay"
    param_names = ['plateau', 'Y0', 'k']
    equation = "Y = plateau + (Y0 - plateau) * exp(-k * X)"
    description = "단상 감쇠. plateau로 수렴하는 감소."

    def model(self, x, plateau, Y0, k):
        z = np.clip(-k * x, -500, 500)
        return plateau + (Y0 - plateau) * np.exp(z)

    def initial_guess(self, x, y):
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        Y0_0 = float(ys[0])
        plateau0 = float(ys[-1])
        span = Y0_0 - plateau0
        if abs(span) < 1e-9 or xs[-1] == xs[0]:
            k0 = 0.1
        else:
            # 중간 지점에서 추정
            target = plateau0 + 0.5 * span
            idx = int(np.argmin(np.abs(ys - target)))
            dx = xs[idx] - xs[0]
            k0 = math.log(2.0) / dx if dx > 0 else 0.1
        return [plateau0, Y0_0, max(k0, 1e-4)]

    def bounds(self, x, y):
        return ([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf])

    def compute_derived_with_ci(self, params_dict, params_vec, cov, dof):
        k = params_dict['k']
        return {
            'half_life': math.log(2.0) / k if k > 0 else float('nan'),
            'span': params_dict['Y0'] - params_dict['plateau'],
        }


class TwoPhaseDecay(CurveModel):
    """Y = plateau + span_fast * exp(-k_fast * X) + span_slow * exp(-k_slow * X)."""
    name = "Two-Phase Decay"
    param_names = ['plateau', 'span_fast', 'k_fast', 'span_slow', 'k_slow']
    equation = ("Y = plateau + span_fast * exp(-k_fast * X) "
                "+ span_slow * exp(-k_slow * X)")
    description = "이상 감쇠. 빠른 위상과 느린 위상을 동시 피팅."

    def model(self, x, plateau, span_fast, k_fast, span_slow, k_slow):
        z1 = np.clip(-k_fast * x, -500, 500)
        z2 = np.clip(-k_slow * x, -500, 500)
        return plateau + span_fast * np.exp(z1) + span_slow * np.exp(z2)

    def initial_guess(self, x, y):
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        plateau0 = float(ys[-1])
        total_span = float(ys[0]) - plateau0
        return [plateau0, 0.5 * total_span, 1.0, 0.5 * total_span, 0.1]

    def bounds(self, x, y):
        return ([-np.inf, -np.inf, 1e-6, -np.inf, 1e-6],
                [np.inf, np.inf, np.inf, np.inf, np.inf])

    def compute_derived_with_ci(self, params_dict, params_vec, cov, dof):
        k1 = params_dict['k_fast']
        k2 = params_dict['k_slow']
        return {
            'half_life_fast': math.log(2.0) / k1 if k1 > 0 else float('nan'),
            'half_life_slow': math.log(2.0) / k2 if k2 > 0 else float('nan'),
        }


class Gompertz(CurveModel):
    """시그모이드 성장: Y = A * exp(-exp(-k * (X - Xi)))."""
    name = "Gompertz"
    param_names = ['A', 'k', 'Xi']
    equation = "Y = A * exp(-exp(-k * (X - Xi)))"
    description = "비대칭 시그모이드 성장. Xi는 변곡점."

    def model(self, x, A, k, Xi):
        z = np.clip(-k * (x - Xi), -500, 500)
        inner = np.clip(-np.exp(z), -500, 500)
        return A * np.exp(inner)

    def initial_guess(self, x, y):
        A0 = float(np.max(y)) if np.max(y) > 0 else 1.0
        # 변곡점: y ≈ A/e 지점 근처
        target = A0 / math.e
        idx = int(np.argmin(np.abs(y - target)))
        Xi0 = float(x[idx])
        x_range = float(np.max(x) - np.min(x))
        k0 = 4.0 / x_range if x_range > 0 else 1.0
        return [A0, k0, Xi0]


class LogisticGrowth(CurveModel):
    """Y = L / (1 + exp(-k * (X - X0)))."""
    name = "Logistic Growth"
    param_names = ['L', 'k', 'X0']
    equation = "Y = L / (1 + exp(-k * (X - X0)))"
    description = "대칭 시그모이드 성장(수용량 L). X0은 변곡점."

    def model(self, x, L, k, X0):
        z = np.clip(-k * (x - X0), -500, 500)
        return L / (1.0 + np.exp(z))

    def initial_guess(self, x, y):
        L0 = float(np.max(y)) * 1.05 if np.max(y) > 0 else 1.0
        X0_0 = _heuristic_log_ec50(x, y)
        x_range = float(np.max(x) - np.min(x))
        k0 = 4.0 / x_range if x_range > 0 else 1.0
        return [L0, k0, X0_0]


# ------------------------------------------------------------------
# Peak Fitting
# ------------------------------------------------------------------
class Gaussian(CurveModel):
    """Y = A * exp(-(X - mu)^2 / (2 sigma^2)) + offset."""
    name = "Gaussian"
    param_names = ['A', 'mu', 'sigma', 'offset']
    equation = "Y = A * exp(-(X - mu)^2 / (2 * sigma^2)) + offset"
    description = "정규분포 피크. A는 진폭, mu는 중심, sigma는 표준편차."

    def model(self, x, A, mu, sigma, offset):
        return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2)) + offset

    def initial_guess(self, x, y):
        offset0 = float(np.min(y))
        A0 = float(np.max(y) - offset0)
        mu0 = float(x[int(np.argmax(y))])
        sigma0 = max(float(np.std(x)) * 0.5, (float(np.max(x)) - float(np.min(x))) / 10, 1e-3)
        return [A0, mu0, sigma0, offset0]

    def bounds(self, x, y):
        x_range = float(np.max(x) - np.min(x)) + 1.0
        return ([-np.inf, float(np.min(x)) - 10 * x_range, 1e-9, -np.inf],
                [np.inf, float(np.max(x)) + 10 * x_range, 100 * x_range, np.inf])

    def compute_derived_with_ci(self, params_dict, params_vec, cov, dof):
        fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * params_dict['sigma']
        return {
            'FWHM': fwhm,
            'peak_center': params_dict['mu'],
            'amplitude': params_dict['A'],
        }


class Lorentzian(CurveModel):
    """Y = A / (1 + ((X - x0)/w)^2) + offset."""
    name = "Lorentzian"
    param_names = ['A', 'x0', 'w', 'offset']
    equation = "Y = A / (1 + ((X - x0) / w)^2) + offset"
    description = "로렌츠 피크. 꼬리가 긴 분포."

    def model(self, x, A, x0, w, offset):
        return A / (1.0 + ((x - x0) / w) ** 2) + offset

    def initial_guess(self, x, y):
        offset0 = float(np.min(y))
        A0 = float(np.max(y) - offset0)
        x0_0 = float(x[int(np.argmax(y))])
        w0 = max((float(np.max(x)) - float(np.min(x))) / 10, 1e-3)
        return [A0, x0_0, w0, offset0]

    def bounds(self, x, y):
        x_range = float(np.max(x) - np.min(x)) + 1.0
        return ([-np.inf, float(np.min(x)) - 10 * x_range, 1e-9, -np.inf],
                [np.inf, float(np.max(x)) + 10 * x_range, 100 * x_range, np.inf])

    def compute_derived_with_ci(self, params_dict, params_vec, cov, dof):
        return {
            'FWHM': 2.0 * abs(params_dict['w']),
            'peak_center': params_dict['x0'],
            'amplitude': params_dict['A'],
        }


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------
AVAILABLE_MODELS = {
    'Hill 4PL': Hill4PL,
    'Hill 5PL': Hill5PL,
    'Probit': Probit,
    'Logit': Logit,
    'LogLogistic 4PL': LogLogistic4P,
    'Michaelis-Menten': MichaelisMenten,
    'Michaelis-Menten (competitive)': MichaelisMentenComp,
    'Exponential Growth': ExponentialGrowth,
    'Exponential Decay': ExponentialDecay,
    'One-Phase Decay': OnePhaseDecay,
    'Two-Phase Decay': TwoPhaseDecay,
    'Gompertz': Gompertz,
    'Logistic Growth': LogisticGrowth,
    'Gaussian': Gaussian,
    'Lorentzian': Lorentzian,
}

MODELS_BY_CATEGORY = {
    '용량-반응 (Dose-Response)': [
        'Hill 4PL', 'Hill 5PL', 'Probit', 'Logit', 'LogLogistic 4PL',
    ],
    '효소반응 (Enzyme)': [
        'Michaelis-Menten', 'Michaelis-Menten (competitive)',
    ],
    '성장/감쇠 (Growth/Decay)': [
        'Exponential Growth', 'Exponential Decay',
        'One-Phase Decay', 'Two-Phase Decay',
        'Gompertz', 'Logistic Growth',
    ],
    '피크 (Peak)': [
        'Gaussian', 'Lorentzian',
    ],
}


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def compare_models(x, y, models, weights=None) -> pd.DataFrame:
    """여러 모델을 같은 데이터로 피팅 후 비교 테이블 반환.

    AICc 오름차순 정렬. 실패한 모델은 NaN 행으로 포함.
    """
    rows = []
    for m in models:
        if not isinstance(m, CurveModel):
            continue
        try:
            r = m.fit(x, y, weights=weights)
        except Exception as e:
            rows.append({
                'model': m.name, 'converged': False, 'n': 0,
                'r_squared': np.nan, 'adj_r2': np.nan,
                'aicc': np.nan, 'bic': np.nan, 'rmse': np.nan,
                'k': len(m.param_names), 'params': {},
                'message': f'예외: {e}',
            })
            continue
        rows.append({
            'model': r.model_name,
            'converged': r.converged,
            'n': r.n,
            'r_squared': r.r_squared,
            'adj_r2': r.adjusted_r2,
            'aicc': r.aicc,
            'bic': r.bic,
            'rmse': r.rmse,
            'k': len(m.param_names),
            'params': r.params,
            'derived': r.derived,
            'message': r.message,
        })
    df = pd.DataFrame(rows)
    if 'aicc' in df.columns:
        df = df.sort_values('aicc', na_position='last').reset_index(drop=True)
    return df


def f_test_nested(result_simple: FitResult, result_complex: FitResult) -> dict:
    """중첩 모델의 F-test.

    F = ((SSR_simple - SSR_complex)/(k_c - k_s)) / (SSR_complex/(n - k_c))
    """
    if (not result_simple.converged or not result_complex.converged):
        return {'F': float('nan'), 'p': float('nan'),
                'recommend': 'simple',
                'message': '한 쪽이 수렴하지 않아 검정 불가'}

    n = result_complex.n
    k_s = len(result_simple._param_order) if result_simple._param_order else len(result_simple.params)
    k_c = len(result_complex._param_order) if result_complex._param_order else len(result_complex.params)
    if k_c <= k_s:
        return {'F': float('nan'), 'p': float('nan'),
                'recommend': 'simple',
                'message': '복잡 모델이 단순 모델보다 파라미터가 많지 않음'}
    if n <= k_c:
        return {'F': float('nan'), 'p': float('nan'),
                'recommend': 'simple',
                'message': '자유도 부족'}

    ssr_s = float(np.sum(result_simple.residuals ** 2))
    ssr_c = float(np.sum(result_complex.residuals ** 2))
    if ssr_c <= 0:
        return {'F': float('inf'), 'p': 0.0, 'recommend': 'complex',
                'message': '복잡 모델 잔차가 0'}
    num = (ssr_s - ssr_c) / (k_c - k_s)
    den = ssr_c / (n - k_c)
    if den <= 0:
        return {'F': float('nan'), 'p': float('nan'),
                'recommend': 'simple',
                'message': '분모 자유도 문제'}
    F = num / den
    if HAS_SCIPY and F > 0:
        p = 1.0 - float(sp_stats.f.cdf(F, k_c - k_s, n - k_c))
    else:
        p = float('nan')
    recommend = 'complex' if (np.isfinite(p) and p < 0.05) else 'simple'
    return {
        'F': float(F),
        'p': float(p) if np.isfinite(p) else float('nan'),
        'df_num': k_c - k_s,
        'df_den': n - k_c,
        'recommend': recommend,
        'message': ('복잡 모델 유의미 (p<0.05)' if recommend == 'complex'
                    else '단순 모델로 충분'),
    }


def extrapolate_ecX(fit_result: FitResult, model: CurveModel,
                    level: float = 0.5) -> tuple:
    """임의 반응 수준 level(0~1)에 해당하는 ECx와 95% CI를 계산.

    level=0.5 → EC50. Hill 4PL/LogLogistic/Probit/Logit 등
    bottom~top이 정의되는 모델에서 유효. 반환: (ECx, lo, hi).
    """
    if not fit_result.converged or not fit_result._model_fn:
        return (float('nan'), float('nan'), float('nan'))

    params = fit_result.params
    name = fit_result.model_name
    level = float(level)

    # target y 계산
    def _target_y():
        if 'bottom' in params and 'top' in params:
            return params['bottom'] + level * (params['top'] - params['bottom'])
        if 'c' in params and 'd' in params:  # LL.4
            return params['c'] + level * (params['d'] - params['c'])
        # Probit/Logit: y는 0~100
        return 100.0 * level

    target = _target_y()

    # 수치적으로 inverse: x의 후보 범위에서 brentq/bisection
    try:
        from scipy.optimize import brentq
    except ImportError:
        return (float('nan'), float('nan'), float('nan'))

    p_order = fit_result._param_order
    p_vec = np.array([params[k] for k in p_order], dtype=float)

    def _solve(p_local):
        def f(xv):
            return float(fit_result._model_fn(np.array([xv]), *p_local)[0]) - target
        # x 검색 범위: 데이터 범위 확장
        if fit_result._x_data is not None and len(fit_result._x_data) > 0:
            x_lo = float(np.min(fit_result._x_data)) - 5
            x_hi = float(np.max(fit_result._x_data)) + 5
        else:
            x_lo, x_hi = -100.0, 100.0
        # 경계값 부호 확인, 필요시 확장
        for _ in range(8):
            try:
                f_lo = f(x_lo)
                f_hi = f(x_hi)
                if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi < 0:
                    return brentq(f, x_lo, x_hi, maxiter=200)
            except Exception:
                pass
            x_lo -= 10
            x_hi += 10
        raise RuntimeError('inverse 탐색 실패')

    try:
        ecx = _solve(p_vec)
    except Exception:
        return (float('nan'), float('nan'), float('nan'))

    # 델타 방법으로 CI
    cov = fit_result.covariance
    if cov is None:
        return (ecx, float('nan'), float('nan'))

    dof = max(fit_result.n - len(p_vec), 1)
    # ecx를 파라미터 함수로 보고 수치미분
    eps = 1e-6
    grad = np.zeros_like(p_vec)
    try:
        for i in range(len(p_vec)):
            step = eps * max(1.0, abs(p_vec[i]))
            dp = np.zeros_like(p_vec)
            dp[i] = step
            try:
                v_plus = _solve(p_vec + dp)
                v_minus = _solve(p_vec - dp)
                grad[i] = (v_plus - v_minus) / (2 * step)
            except Exception:
                grad[i] = 0.0
        var = float(grad @ cov @ grad)
        if not np.isfinite(var) or var < 0:
            return (ecx, float('nan'), float('nan'))
        se = math.sqrt(var)
        if HAS_SCIPY:
            tval = float(sp_stats.t.ppf(0.975, dof))
        else:
            tval = 1.96
        return (float(ecx), float(ecx - tval * se), float(ecx + tval * se))
    except Exception:
        return (float(ecx), float('nan'), float('nan'))
