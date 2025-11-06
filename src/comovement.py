"""
공행성 탐지 모듈

품목 간 선행-후행 관계를 탐지하는 다양한 방법을 제공합니다.
- CCF (Cross-Correlation Function)
- Granger Causality Test
- DTW (Dynamic Time Warping)
- FDR (False Discovery Rate) 다중 검정 보정
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from itertools import combinations
from statsmodels.tsa.stattools import ccf, grangercausalitytests
from dtw import dtw
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_ccf(x: pd.Series,
                 y: pd.Series,
                 max_lag: int = 6,
                 adjusted: bool = False) -> Tuple[np.ndarray, int, float]:
    """
    두 시계열 간의 교차 상관 계수(CCF)를 계산합니다.

    양수 lag: x가 y를 선행 (x(t) -> y(t+lag))
    음수 lag: y가 x를 선행 (y(t) -> x(t-lag))

    Parameters:
    -----------
    x : pd.Series
        첫 번째 시계열 (잠재적 선행 변수)
    y : pd.Series
        두 번째 시계열 (잠재적 후행 변수)
    max_lag : int
        최대 시차 (기본값: 6)
    adjusted : bool
        조정된 CCF 사용 여부 (기본값: False)

    Returns:
    --------
    Tuple[np.ndarray, int, float]
        (ccf_values, optimal_lag, max_ccf)
        - ccf_values: 각 시차별 CCF 값
        - optimal_lag: 최대 절대값을 갖는 시차
        - max_ccf: 최대 CCF 값
    """
    # 결측값 제거
    valid_idx = ~(x.isna() | y.isna())
    x_clean = x[valid_idx]
    y_clean = y[valid_idx]

    if len(x_clean) < max_lag + 2:
        logger.warning(f"시계열 길이({len(x_clean)})가 너무 짧습니다.")
        return np.array([np.nan] * (2 * max_lag + 1)), 0, np.nan

    # CCF 계산
    ccf_values = ccf(x_clean, y_clean, adjusted=adjusted)[:max_lag + 1]

    # 최대 절대값을 갖는 시차 찾기
    abs_ccf = np.abs(ccf_values)
    optimal_lag = np.argmax(abs_ccf)
    max_ccf = ccf_values[optimal_lag]

    return ccf_values, optimal_lag, max_ccf


def calculate_ccf_matrix(df_wide: pd.DataFrame,
                        max_lag: int = 6,
                        threshold: float = 0.5) -> pd.DataFrame:
    """
    모든 품목 쌍에 대해 CCF를 계산합니다.

    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format 데이터프레임 (index: date, columns: item_code)
    max_lag : int
        최대 시차 (기본값: 6)
    threshold : float
        CCF 임계값 (기본값: 0.5)

    Returns:
    --------
    pd.DataFrame
        각 품목 쌍의 CCF 결과
        컬럼: item_x, item_y, optimal_lag, max_ccf, is_significant
    """
    logger.info(f"CCF 계산 중 ({len(df_wide.columns)}개 품목)...")

    items = df_wide.columns.tolist()
    results = []

    # 모든 품목 쌍에 대해 CCF 계산
    for i, item_x in enumerate(items):
        for item_y in items[i+1:]:  # 중복 제거를 위해 i+1부터 시작
            x = df_wide[item_x]
            y = df_wide[item_y]

            try:
                ccf_values, optimal_lag, max_ccf = calculate_ccf(x, y, max_lag)

                results.append({
                    'item_x': item_x,
                    'item_y': item_y,
                    'optimal_lag': optimal_lag,
                    'max_ccf': max_ccf,
                    'abs_ccf': abs(max_ccf),
                    'is_significant': abs(max_ccf) >= threshold
                })
            except Exception as e:
                logger.warning(f"CCF 계산 실패 ({item_x}, {item_y}): {e}")
                continue

    results_df = pd.DataFrame(results)

    # 통계 요약
    n_significant = results_df['is_significant'].sum()
    logger.info(f"✓ CCF 계산 완료: {len(results_df)}개 쌍")
    logger.info(f"  유의미한 공행성 ({threshold} 이상): {n_significant}개 쌍 ({n_significant/len(results_df)*100:.1f}%)")

    return results_df.sort_values('abs_ccf', ascending=False).reset_index(drop=True)


def calculate_granger_causality(x: pd.Series,
                               y: pd.Series,
                               max_lag: int = 6,
                               test: str = 'ssr_ftest') -> Dict:
    """
    Granger 인과관계 검정을 수행합니다.

    귀무가설(H0): x는 y를 Granger-cause 하지 않는다.
    p-value < 0.05 이면 귀무가설 기각 -> x가 y를 Granger-cause

    Parameters:
    -----------
    x : pd.Series
        잠재적 원인 변수
    y : pd.Series
        잠재적 결과 변수
    max_lag : int
        최대 시차 (기본값: 6)
    test : str
        검정 유형 (기본값: 'ssr_ftest')

    Returns:
    --------
    Dict
        검정 결과 딕셔너리
    """
    # 결측값 제거 및 정렬
    df = pd.DataFrame({'x': x, 'y': y}).dropna()

    if len(df) < max_lag + 10:
        return {
            'best_lag': None,
            'best_p_value': np.nan,
            'is_causal': False,
            'error': 'Insufficient data'
        }

    try:
        # Granger 인과관계 검정 수행
        result = grangercausalitytests(df[['y', 'x']], maxlag=max_lag, verbose=False)

        # 각 시차별 p-value 추출
        p_values = {}
        for lag in range(1, max_lag + 1):
            p_values[lag] = result[lag][0][test][1]  # [0]은 테스트 결과, [1]은 p-value

        # 최소 p-value를 갖는 시차 선택
        best_lag = min(p_values, key=p_values.get)
        best_p_value = p_values[best_lag]

        return {
            'best_lag': best_lag,
            'best_p_value': best_p_value,
            'is_causal': best_p_value < 0.05,
            'all_p_values': p_values
        }
    except Exception as e:
        return {
            'best_lag': None,
            'best_p_value': np.nan,
            'is_causal': False,
            'error': str(e)
        }


def calculate_granger_matrix(df_wide: pd.DataFrame,
                            max_lag: int = 6,
                            p_threshold: float = 0.05) -> pd.DataFrame:
    """
    모든 품목 쌍에 대해 Granger 인과관계 검정을 수행합니다.

    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format 데이터프레임
    max_lag : int
        최대 시차 (기본값: 6)
    p_threshold : float
        p-value 임계값 (기본값: 0.05)

    Returns:
    --------
    pd.DataFrame
        각 품목 쌍의 Granger 인과관계 결과
    """
    logger.info(f"Granger 인과관계 검정 중 ({len(df_wide.columns)}개 품목)...")

    items = df_wide.columns.tolist()
    results = []

    # 모든 품목 쌍에 대해 양방향 검정
    for item_x in items:
        for item_y in items:
            if item_x == item_y:
                continue

            x = df_wide[item_x]
            y = df_wide[item_y]

            result = calculate_granger_causality(x, y, max_lag)

            results.append({
                'item_cause': item_x,
                'item_effect': item_y,
                'best_lag': result['best_lag'],
                'p_value': result['best_p_value'],
                'is_causal': result['is_causal']
            })

    results_df = pd.DataFrame(results)

    # 통계 요약
    n_causal = results_df['is_causal'].sum()
    logger.info(f"✓ Granger 검정 완료: {len(results_df)}개 쌍")
    logger.info(f"  인과관계 발견 (p < {p_threshold}): {n_causal}개 쌍 ({n_causal/len(results_df)*100:.1f}%)")

    return results_df.sort_values('p_value').reset_index(drop=True)


def calculate_dtw_distance(x: pd.Series,
                          y: pd.Series,
                          window_type: str = 'sakoechiba',
                          window_size: int = 3) -> float:
    """
    두 시계열 간의 DTW (Dynamic Time Warping) 거리를 계산합니다.

    Parameters:
    -----------
    x : pd.Series
        첫 번째 시계열
    y : pd.Series
        두 번째 시계열
    window_type : str
        DTW 창 유형 (기본값: 'sakoechiba')
    window_size : int
        창 크기 (기본값: 3)

    Returns:
    --------
    float
        정규화된 DTW 거리 (0에 가까울수록 유사)
    """
    # 결측값 제거
    valid_idx = ~(x.isna() | y.isna())
    x_clean = x[valid_idx].values.reshape(-1, 1)
    y_clean = y[valid_idx].values.reshape(-1, 1)

    # 정규화 (0-1 스케일)
    x_norm = (x_clean - x_clean.min()) / (x_clean.max() - x_clean.min() + 1e-8)
    y_norm = (y_clean - y_clean.min()) / (y_clean.max() - y_clean.min() + 1e-8)

    # DTW 거리 계산
    alignment = dtw(x_norm, y_norm, keep_internals=True,
                   window_type=window_type, window_args={'window_size': window_size})

    # 정규화된 거리 (경로 길이로 나눔)
    normalized_distance = alignment.distance / alignment.normalizedDistance

    return normalized_distance


def calculate_dtw_matrix(df_wide: pd.DataFrame,
                        threshold: float = 0.3) -> pd.DataFrame:
    """
    모든 품목 쌍에 대해 DTW 거리를 계산합니다.

    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format 데이터프레임
    threshold : float
        DTW 거리 임계값 (기본값: 0.3, 낮을수록 유사)

    Returns:
    --------
    pd.DataFrame
        각 품목 쌍의 DTW 거리
    """
    logger.info(f"DTW 거리 계산 중 ({len(df_wide.columns)}개 품목)...")

    items = df_wide.columns.tolist()
    results = []

    # 모든 품목 쌍에 대해 DTW 계산
    for i, item_x in enumerate(items):
        for item_y in items[i+1:]:
            x = df_wide[item_x]
            y = df_wide[item_y]

            try:
                dtw_dist = calculate_dtw_distance(x, y)

                results.append({
                    'item_x': item_x,
                    'item_y': item_y,
                    'dtw_distance': dtw_dist,
                    'is_similar': dtw_dist <= threshold
                })
            except Exception as e:
                logger.warning(f"DTW 계산 실패 ({item_x}, {item_y}): {e}")
                continue

    results_df = pd.DataFrame(results)

    # 통계 요약
    n_similar = results_df['is_similar'].sum()
    logger.info(f"✓ DTW 계산 완료: {len(results_df)}개 쌍")
    logger.info(f"  유사 패턴 (거리 <= {threshold}): {n_similar}개 쌍 ({n_similar/len(results_df)*100:.1f}%)")

    return results_df.sort_values('dtw_distance').reset_index(drop=True)


def apply_fdr_correction(p_values: np.ndarray,
                        alpha: float = 0.05,
                        method: str = 'bh') -> Tuple[np.ndarray, float]:
    """
    FDR (False Discovery Rate) 보정을 적용합니다.

    Benjamini-Hochberg 절차를 사용하여 다중 검정 문제를 보정합니다.

    Parameters:
    -----------
    p_values : np.ndarray
        원본 p-value 배열
    alpha : float
        FDR 수준 (기본값: 0.05)
    method : str
        보정 방법 ('bh': Benjamini-Hochberg, 기본값)

    Returns:
    --------
    Tuple[np.ndarray, float]
        (rejected, threshold)
        - rejected: 각 가설의 기각 여부 (Boolean 배열)
        - threshold: 조정된 p-value 임계값
    """
    # 결측값 제거
    valid_mask = ~np.isnan(p_values)
    p_clean = p_values[valid_mask]

    if len(p_clean) == 0:
        return np.array([False] * len(p_values)), alpha

    # p-value 정렬
    n = len(p_clean)
    sorted_idx = np.argsort(p_clean)
    sorted_p = p_clean[sorted_idx]

    # Benjamini-Hochberg 임계값 계산
    thresholds = alpha * np.arange(1, n + 1) / n

    # 기각할 가설 찾기
    rejected_sorted = sorted_p <= thresholds

    if rejected_sorted.any():
        # 마지막으로 기각된 가설의 인덱스
        max_idx = np.where(rejected_sorted)[0][-1]
        threshold = thresholds[max_idx]
    else:
        threshold = 0

    # 원래 순서로 복원
    rejected = np.zeros(n, dtype=bool)
    rejected[sorted_idx] = rejected_sorted

    # 결측값 위치에 False 삽입
    result = np.zeros(len(p_values), dtype=bool)
    result[valid_mask] = rejected

    return result, threshold


def apply_fdr_to_results(results_df: pd.DataFrame,
                        p_value_col: str = 'p_value',
                        alpha: float = 0.05) -> pd.DataFrame:
    """
    결과 데이터프레임에 FDR 보정을 적용합니다.

    Parameters:
    -----------
    results_df : pd.DataFrame
        p-value를 포함한 결과 데이터프레임
    p_value_col : str
        p-value 컬럼명 (기본값: 'p_value')
    alpha : float
        FDR 수준 (기본값: 0.05)

    Returns:
    --------
    pd.DataFrame
        FDR 보정이 추가된 결과 데이터프레임
    """
    logger.info(f"FDR 보정 적용 중 (alpha={alpha})...")

    df = results_df.copy()
    p_values = df[p_value_col].values

    # FDR 보정 적용
    rejected, threshold = apply_fdr_correction(p_values, alpha)

    df['fdr_rejected'] = rejected
    df['fdr_threshold'] = threshold

    n_rejected = rejected.sum()
    logger.info(f"✓ FDR 보정 완료")
    logger.info(f"  보정 전 유의: {(p_values < alpha).sum()}개")
    logger.info(f"  보정 후 유의: {n_rejected}개 (임계값: {threshold:.4f})")

    return df


def detect_comovement_comprehensive(df_wide: pd.DataFrame,
                                   ccf_threshold: float = 0.5,
                                   granger_alpha: float = 0.05,
                                   dtw_threshold: float = 0.3,
                                   max_lag: int = 6,
                                   apply_fdr: bool = True,
                                   fdr_alpha: float = 0.05) -> Dict[str, pd.DataFrame]:
    """
    CCF, Granger, DTW를 모두 사용하여 종합적인 공행성 탐지를 수행합니다.

    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format 데이터프레임
    ccf_threshold : float
        CCF 임계값
    granger_alpha : float
        Granger 검정 p-value 임계값
    dtw_threshold : float
        DTW 거리 임계값
    max_lag : int
        최대 시차
    apply_fdr : bool
        FDR 보정 적용 여부
    fdr_alpha : float
        FDR 수준

    Returns:
    --------
    Dict[str, pd.DataFrame]
        {'ccf': ccf_results, 'granger': granger_results, 'dtw': dtw_results}
    """
    logger.info("=" * 50)
    logger.info("종합 공행성 탐지 시작")
    logger.info("=" * 50)

    results = {}

    # 1. CCF 분석
    logger.info("\n[1/3] CCF 분석...")
    ccf_results = calculate_ccf_matrix(df_wide, max_lag, ccf_threshold)
    results['ccf'] = ccf_results

    # 2. Granger 인과관계 분석
    logger.info("\n[2/3] Granger 인과관계 분석...")
    granger_results = calculate_granger_matrix(df_wide, max_lag, granger_alpha)

    if apply_fdr:
        granger_results = apply_fdr_to_results(granger_results, 'p_value', fdr_alpha)

    results['granger'] = granger_results

    # 3. DTW 분석
    logger.info("\n[3/3] DTW 분석...")
    dtw_results = calculate_dtw_matrix(df_wide, dtw_threshold)
    results['dtw'] = dtw_results

    logger.info("\n" + "=" * 50)
    logger.info("✓ 종합 공행성 탐지 완료")
    logger.info("=" * 50)

    return results
