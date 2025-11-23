"""
Chronos + N-HiTS 앙상블 스크립트

두 모델의 예측을 가중 평균하여 최종 예측을 생성합니다.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_predictions(chronos_path, nhits_path):
    """두 모델의 예측 결과를 로드"""
    logger.info("예측 파일 로드 중...")
    
    chronos_pred = pd.read_csv(chronos_path)
    nhits_pred = pd.read_csv(nhits_path)
    
    logger.info(f"Chronos 예측: {chronos_pred.shape}")
    logger.info(f"N-HiTS 예측: {nhits_pred.shape}")
    
    return chronos_pred, nhits_pred


def weighted_ensemble(chronos_pred, nhits_pred, weights=(0.6, 0.4)):
    """
    가중 평균 앙상블
    
    Args:
        chronos_pred: Chronos 예측 DataFrame
        nhits_pred: N-HiTS 예측 DataFrame
        weights: (chronos_weight, nhits_weight)
    """
    w_chronos, w_nhits = weights
    logger.info(f"앙상블 가중치: Chronos {w_chronos:.1%}, N-HiTS {w_nhits:.1%}")
    
    # 복사본 생성
    ensemble = chronos_pred.copy()
    
    # 가중 평균 계산
    ensemble['value'] = (
        w_chronos * chronos_pred['value'] + 
        w_nhits * nhits_pred['value']
    )
    
    # 음수 방지
    ensemble['value'] = ensemble['value'].clip(lower=0)
    
    logger.info(f"✅ 앙상블 완료")
    logger.info(f"평균 예측값: {ensemble['value'].mean():.2f}")
    logger.info(f"최대 예측값: {ensemble['value'].max():.2f}")
    logger.info(f"비영 예측: {(ensemble['value'] > 0).sum()}개")
    
    return ensemble


def grid_search_weights(chronos_pred, nhits_pred, output_dir):
    """
    다양한 가중치 조합으로 앙상블 파일 생성
    
    최적 가중치는 제출 후 점수로 확인
    """
    logger.info("가중치 그리드 서치 시작...")
    
    weight_combinations = [
        (0.5, 0.5),  # Equal
        (0.6, 0.4),  # Chronos 우세
        (0.7, 0.3),  # Chronos 강우세
        (0.4, 0.6),  # N-HiTS 우세
        (0.8, 0.2),  # Chronos 매우 우세
    ]
    
    results = []
    
    for w_chronos, w_nhits in weight_combinations:
        ensemble = weighted_ensemble(chronos_pred, nhits_pred, (w_chronos, w_nhits))
        
        # 파일명
        filename = f"submission_ensemble_{int(w_chronos*10)}c_{int(w_nhits*10)}n.csv"
        output_path = output_dir / filename
        
        ensemble.to_csv(output_path, index=False)
        logger.info(f"저장: {filename}")
        
        results.append({
            'weights': f"{w_chronos}/{w_nhits}",
            'filename': filename,
            'mean_value': ensemble['value'].mean(),
            'nonzero_count': (ensemble['value'] > 0).sum()
        })
    
    # 결과 요약
    results_df = pd.DataFrame(results)
    summary_path = output_dir / 'ensemble_weights_summary.csv'
    results_df.to_csv(summary_path, index=False)
    logger.info(f"\n요약 저장: {summary_path}")
    print("\n" + "=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)


def main():
    print("=" * 60)
    print("Chronos + N-HiTS 앙상블")
    print("=" * 60)
    
    # 경로 설정 (사용자가 직접 수정 필요)
    # 최신 제출 파일들의 경로를 지정하세요
    chronos_path = Config.OUTPUT_DIR / 'submission_log' / 'LATEST_CHRONOS' / 'submission_chronos_large.csv'
    nhits_path = Config.OUTPUT_DIR / 'submission_log' / 'LATEST_NHITS' / 'submission_nhits_0.2.csv'
    
    # 파일 존재 확인
    if not chronos_path.exists():
        logger.error(f"❌ Chronos 파일 없음: {chronos_path}")
        logger.info("먼저 src/train_chronos.py를 실행하세요")
        # 대체 경로 탐색
        latest_dirs = sorted((Config.OUTPUT_DIR / 'submission_log').glob('*'), reverse=True)
        for d in latest_dirs:
            chronos_files = list(d.glob('submission_chronos*.csv'))
            if chronos_files:
                chronos_path = chronos_files[0]
                logger.info(f"대체 파일 사용: {chronos_path}")
                break
    
    if not nhits_path.exists():
        logger.error(f"❌ N-HiTS 파일 없음: {nhits_path}")
        logger.info("먼저 src/train_nhits.py를 실행하세요")
        # 대체 경로 탐색
        latest_dirs = sorted((Config.OUTPUT_DIR / 'submission_log').glob('*'), reverse=True)
        for d in latest_dirs:
            nhits_files = list(d.glob('submission_nhits*.csv'))
            if nhits_files:
                nhits_path = nhits_files[0]
                logger.info(f"대체 파일 사용: {nhits_path}")
                break
    
    # 예측 로드
    chronos_pred, nhits_pred = load_predictions(chronos_path, nhits_path)
    
    # 출력 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Config.OUTPUT_DIR / 'submission_log' / f'ensemble_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"출력 디렉토리: {output_dir}")
    
    # 다양한 가중치 조합으로 앙상블 생성
    grid_search_weights(chronos_pred, nhits_pred, output_dir)
    
    print("\n✅ 완료!")
    print(f"📁 {output_dir}")
    print("각 파일을 제출하여 최적 가중치를 찾으세요")


if __name__ == "__main__":
    main()
