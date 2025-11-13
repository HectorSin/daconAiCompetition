"""
제출 파일 생성 스크립트

sample_submission.csv 형식으로 9,900개 쌍의 예측값을 생성합니다.
- 공행성이 없는 쌍: value = 0
- 공행성이 있는 쌍: value = 후행 품목의 2025.08 예측값
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.features import create_features_for_item

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_next_month(df_wide: pd.DataFrame,
                       item: str,
                       ccf_results: pd.DataFrame,
                       model_path: Path) -> float:
    """
    품목의 다음 달(2025.08) 값을 예측합니다.

    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format 데이터
    item : str
        품목명
    ccf_results : pd.DataFrame
        CCF 결과
    model_path : Path
        저장된 모델 경로

    Returns:
    --------
    float
        2025.08 예측값
    """
    # 모델 로드
    try:
        with open(model_path, 'rb') as f:
            result = pickle.load(f)
        model = result['model']
        feature_names = result['feature_names']
    except Exception as e:
        logger.warning(f"품목 {item} 모델 로드 실패: {e}")
        return 0.0

    # 특징 생성 (전체 데이터 사용)
    features = create_features_for_item(
        df_wide, item, ccf_results,
        lag_features=[1, 2, 3, 6],
        rolling_windows=[3, 6],
        include_leading=True,
        top_leading=3
    )

    # 마지막 행 (2025.07의 특징 = 2025.08 예측용)
    last_row = features.iloc[[-1]]

    # 필요한 특징만 선택
    try:
        X_pred = last_row[feature_names]
    except KeyError:
        logger.warning(f"품목 {item}: 특징 불일치")
        return 0.0

    # 결측값 체크
    if X_pred.isnull().any().any():
        logger.warning(f"품목 {item}: 예측 특징에 결측값 존재")
        return 0.0

    # 예측
    pred = model.predict(X_pred)[0]

    # 음수 방지
    pred = max(0, pred)

    return pred


def create_submission(df_wide: pd.DataFrame,
                     ccf_results: pd.DataFrame,
                     sample_submission_path: Path,
                     models_dir: Path,
                     output_path: Path,
                     ccf_threshold: float = 0.3) -> pd.DataFrame:
    """
    제출 파일을 생성합니다.

    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format 데이터
    ccf_results : pd.DataFrame
        CCF 결과
    sample_submission_path : Path
        샘플 제출 파일 경로
    models_dir : Path
        모델 저장 디렉토리
    output_path : Path
        출력 파일 경로
    ccf_threshold : float
        CCF 임계값 (이상이면 공행성 있음)

    Returns:
    --------
    pd.DataFrame
        제출 데이터프레임
    """
    logger.info("제출 파일 생성 시작...")

    # 1. 샘플 제출 파일 로드
    submission = pd.read_csv(sample_submission_path)
    logger.info(f"샘플 제출 파일: {len(submission)}개 쌍")

    # 2. 각 품목의 2025.08 예측값 계산
    logger.info("\n각 품목의 2025.08 예측값 계산 중...")
    predictions = {}

    for item in df_wide.columns:
        model_path = models_dir / f'model_{item}.pkl'

        if not model_path.exists():
            logger.warning(f"품목 {item}: 모델 파일 없음")
            predictions[item] = 0.0
            continue

        pred = predict_next_month(df_wide, item, ccf_results, model_path)
        predictions[item] = pred

        if len(predictions) % 10 == 0:
            logger.info(f"진행: {len(predictions)}/{len(df_wide.columns)}")

    logger.info(f"예측 완료: {len(predictions)}개 품목")

    # 3. 제출 파일의 각 쌍에 대해 value 채우기
    logger.info("\n9,900개 쌍의 value 계산 중...")

    values = []

    for idx, row in submission.iterrows():
        leading_item = row['leading_item_id']
        following_item = row['following_item_id']

        # CCF 결과에서 공행성 확인
        comovement = ccf_results[
            ((ccf_results['item_x'] == leading_item) & (ccf_results['item_y'] == following_item)) |
            ((ccf_results['item_x'] == following_item) & (ccf_results['item_y'] == leading_item))
        ]

        has_comovement = False
        if len(comovement) > 0:
            max_ccf = comovement['abs_ccf'].max()
            if max_ccf >= ccf_threshold:
                has_comovement = True

        if has_comovement:
            # 공행성 있음 → 후행 품목의 예측값 사용
            value = predictions.get(following_item, 0.0)
        else:
            # 공행성 없음 → 0
            value = 0.0

        values.append(value)

        if (idx + 1) % 1000 == 0:
            logger.info(f"진행: {idx+1}/{len(submission)}")

    submission['value'] = values

    # 4. 저장
    submission.to_csv(output_path, index=False)
    logger.info(f"\n제출 파일 저장 완료: {output_path}")

    # 5. 통계
    n_nonzero = (submission['value'] > 0).sum()
    n_zero = (submission['value'] == 0).sum()

    logger.info(f"\n제출 파일 통계:")
    logger.info(f"  총 쌍: {len(submission)}개")
    logger.info(f"  공행성 있음 (value > 0): {n_nonzero}개 ({n_nonzero/len(submission)*100:.1f}%)")
    logger.info(f"  공행성 없음 (value = 0): {n_zero}개 ({n_zero/len(submission)*100:.1f}%)")
    logger.info(f"  value 합계: {submission['value'].sum():,.0f}")
    logger.info(f"  value 평균: {submission['value'].mean():,.2f}")

    return submission


def main():
    print("=" * 60)
    print("제출 파일 생성")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1/3] 데이터 로드 중...")
    df_raw = pd.read_csv(Config.DATA_RAW / 'train.csv')
    df_raw['date'] = pd.to_datetime(df_raw[['year', 'month']].assign(day=1))

    df_agg = df_raw.groupby(['date', 'item_id']).agg({
        'value': 'sum'
    }).reset_index()

    df_wide = df_agg.pivot(index='date', columns='item_id', values='value').fillna(0)
    print(f"데이터 shape: {df_wide.shape}")
    print(f"마지막 날짜: {df_wide.index[-1]}")

    # 2. CCF 결과 로드
    print("\n[2/3] CCF 결과 로드 중...")
    ccf_results = pd.read_csv(Config.DATA_PROCESSED / 'ccf_results.csv')
    print(f"CCF 결과: {len(ccf_results)}개 쌍")

    # 3. 제출 파일 생성
    print("\n[3/3] 제출 파일 생성 중...")

    submission = create_submission(
        df_wide=df_wide,
        ccf_results=ccf_results,
        sample_submission_path=Config.DATA_RAW / 'sample_submission.csv',
        models_dir=Config.MODELS_DIR,
        output_path=Config.OUTPUT_DIR / 'submission.csv',
        ccf_threshold=0.3  # 임계값 조정 가능
    )

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"\n제출 파일: {Config.OUTPUT_DIR / 'submission.csv'}")
    print(f"파일 크기: {len(submission)}개 행")


if __name__ == "__main__":
    main()
