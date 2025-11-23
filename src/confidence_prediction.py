"""
신뢰도 기반 직접 예측

Chronos 예측값이 일정 수준 이상인 것만 제출
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

def main():
    print("=" * 60)
    print("신뢰도 기반 직접 예측")
    print("=" * 60)
    
    # Chronos 예측 로드
    chronos_path = Config.OUTPUT_DIR / 'submission_log' / '20251123_105903' / 'submission_chronos_large.csv'
    chronos_df = pd.read_csv(chronos_path)
    
    # 품목별 예측값 딕셔너리
    pred_dict = {}
    for idx, row in chronos_df.iterrows():
        if row['value'] > 0:
            if row['following_item_id'] not in pred_dict:
                pred_dict[row['following_item_id']] = []
            pred_dict[row['following_item_id']].append(row['value'])
    
    # 각 품목의 평균 예측값
    avg_pred_dict = {}
    for item, values in pred_dict.items():
        avg_pred_dict[item] = np.mean(values)
    
    # 다양한 신뢰도 임계값 테스트
    thresholds = [100000, 500000, 1000000, 5000000, 10000000]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Config.OUTPUT_DIR / 'submission_log' / f'confidence_pred_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for threshold in thresholds:
        sample_submission = pd.read_csv(Config.DATA_RAW / 'sample_submission.csv')
        
        # 신뢰도 임계값 이상만 예측
        values = []
        for idx, row in sample_submission.iterrows():
            following_item = row['following_item_id']
            pred = avg_pred_dict.get(following_item, 0.0)
            
            # 신뢰도 필터링
            if pred >= threshold:
                values.append(pred)
            else:
                values.append(0.0)
        
        sample_submission['value'] = values
        
        # 저장
        output_path = output_dir / f'submission_confidence_{threshold}.csv'
        sample_submission.to_csv(output_path, index=False)
        
        # 통계
        n_nonzero = (sample_submission['value'] > 0).sum()
        print(f"\n임계값: {threshold:,}")
        print(f"  비영 예측: {n_nonzero}개 ({n_nonzero/len(sample_submission)*100:.1f}%)")
        print(f"  평균 예측값: {sample_submission['value'].mean():.2f}")
    
    print(f"\n저장 위치: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
