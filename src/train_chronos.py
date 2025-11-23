"""
Chronos Foundation Model을 사용한 시계열 예측 스크립트

Amazon의 Chronos-T5 모델을 활용하여 100개 품목에 대한
Zero-shot 시계열 예측을 수행합니다.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import torch
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def install_chronos():
    """Chronos 설치 확인 및 안내"""
    try:
        from chronos import ChronosPipeline
        logger.info("✅ Chronos already installed")
    except ImportError:
        logger.error("❌ Chronos not installed!")
        logger.error("Run: pip install git+https://github.com/amazon-science/chronos-forecasting.git")
        sys.exit(1)


def load_data():
    """데이터 로드 및 품목별 시계열 생성"""
    logger.info("데이터 로드 중...")
    df_raw = pd.read_csv(Config.DATA_RAW / 'train.csv')
    
    # 날짜 변환
    df_raw['ds'] = pd.to_datetime(df_raw[['year', 'month']].assign(day=1))
    
    # 품목별 월별 집계
    df_agg = df_raw.groupby(['ds', 'item_id']).agg({
        'value': 'sum'
    }).reset_index()
    
    # 품목별로 정렬
    df_agg = df_agg.sort_values(['item_id', 'ds'])
    
    logger.info(f"데이터 로드 완료: {df_agg.shape}")
    logger.info(f"품목 수: {df_agg['item_id'].nunique()}")
    logger.info(f"기간: {df_agg['ds'].min()} ~ {df_agg['ds'].max()}")
    
    return df_agg


def create_context_dict(df):
    """품목별 시계열 데이터를 딕셔너리로 변환"""
    context_dict = {}
    
    for item_id in df['item_id'].unique():
        item_data = df[df['item_id'] == item_id].sort_values('ds')
        # Chronos는 torch.Tensor 입력 필요
        context_dict[item_id] = torch.tensor(
            item_data['value'].values, 
            dtype=torch.float32
        )
    
    logger.info(f"✅ {len(context_dict)}개 품목 준비 완료")
    return context_dict


def predict_with_chronos(context_dict, model_name="amazon/chronos-t5-large"):
    """
    Chronos 모델로 예측 수행
    
    Args:
        context_dict: 품목별 과거 시계열 데이터
        model_name: 사용할 Chronos 모델
            - "amazon/chronos-t5-small" (8M params)
            - "amazon/chronos-t5-base" (46M params)
            - "amazon/chronos-t5-large" (200M params) ⭐ 추천
    """
    from chronos import ChronosPipeline
    
    logger.info(f"Chronos 모델 로드 중: {model_name}")
    
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # 모델 로드
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    
    logger.info("✅ 모델 로드 완료")
    
    # 예측 수행
    predictions = {}
    logger.info(f"100개 품목 예측 시작...")
    
    for idx, (item_id, context) in enumerate(context_dict.items(), 1):
        # Chronos는 확률적 예측 (num_samples개 샘플 생성)
        # API: predict(context_tensor, prediction_length, num_samples)
        forecast = pipeline.predict(
            context.unsqueeze(0),  # (1, seq_len) - 첫 번째 positional arg
            prediction_length=1,  # 1개월 예측
            num_samples=20,  # Monte Carlo 샘플 수
        )
        
        # Median 사용 (중앙값)
        # forecast shape: (num_samples, prediction_length)
        median_pred = torch.median(forecast[:, 0]).item()
        predictions[item_id] = max(0, median_pred)  # 음수 방지
        
        if idx % 10 == 0:
            logger.info(f"진행: {idx}/100")
    
    logger.info("✅ 예측 완료")
    return predictions


def create_submission_file(predictions, ccf_results, output_path, threshold=0.2):
    """
    예측값을 기반으로 제출 파일을 생성합니다.
    """
    logger.info(f"제출 파일 생성 중 (임계값: {threshold})...")
    
    sample_submission = pd.read_csv(Config.DATA_RAW / 'sample_submission.csv')
    
    values = []
    
    for idx, row in sample_submission.iterrows():
        leading_item = row['leading_item_id']
        following_item = row['following_item_id']
        
        # CCF 확인
        comovement = ccf_results[
            ((ccf_results['item_x'] == leading_item) & (ccf_results['item_y'] == following_item)) |
            ((ccf_results['item_x'] == following_item) & (ccf_results['item_y'] == leading_item))
        ]
        
        has_comovement = False
        if len(comovement) > 0:
            max_ccf = comovement['abs_ccf'].max()
            if max_ccf >= threshold:
                has_comovement = True
        
        if has_comovement:
            pred = predictions.get(following_item, 0.0)
            values.append(pred)
        else:
            values.append(0.0)
            
        if (idx + 1) % 1000 == 0:
            logger.info(f"진행: {idx+1}/{len(sample_submission)}")
            
    sample_submission['value'] = values
    sample_submission.to_csv(output_path, index=False)
    logger.info(f"저장 완료: {output_path}")
    
    # 통계
    n_nonzero = (sample_submission['value'] > 0).sum()
    logger.info(f"공행성 쌍: {n_nonzero}개 ({n_nonzero/len(sample_submission)*100:.1f}%)")
    logger.info(f"평균 예측값: {sample_submission['value'].mean():.2f}")
    logger.info(f"최대 예측값: {sample_submission['value'].max():.2f}")


def main():
    print("=" * 60)
    print("Chronos Foundation Model 예측")
    print("=" * 60)
    
    # 0. Chronos 설치 확인
    install_chronos()
    
    # 1. 데이터 준비
    df = load_data()
    
    # 2. 품목별 시계열 준비
    context_dict = create_context_dict(df)
    
    # 3. Chronos로 예측
    # 모델 크기 선택: small (빠름), base (균형), large (정확)
    predictions = predict_with_chronos(
        context_dict, 
        model_name="amazon/chronos-t5-large"  # 200M params
    )
    
    # 4. 제출 파일 생성
    ccf_results = pd.read_csv(Config.DATA_PROCESSED / 'ccf_results.csv')
    
    # 타임스탬프 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Config.OUTPUT_DIR / 'submission_log' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"출력 디렉토리: {output_dir}")
    
    output_path = output_dir / 'submission_chronos_large.csv'
    
    create_submission_file(predictions, ccf_results, output_path, threshold=0.2)
    
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
