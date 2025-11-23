"""
공행성 없이 직접 예측하는 전략

모든 following_item을 직접 예측하여 제출
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
    print("직접 예측 전략 (공행성 무시)")
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
    
    print(f"예측 가능한 품목: {len(avg_pred_dict)}개")
    
    # 제출 파일 생성
    sample_submission = pd.read_csv(Config.DATA_RAW / 'sample_submission.csv')
    
    # 모든 following_item에 대해 예측값 할당
    values = []
    for idx, row in sample_submission.iterrows():
        following_item = row['following_item_id']
        pred = avg_pred_dict.get(following_item, 0.0)
        values.append(pred)
    
    sample_submission['value'] = values
    
    # 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Config.OUTPUT_DIR / 'submission_log' / f'direct_pred_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'submission_direct_prediction.csv'
    sample_submission.to_csv(output_path, index=False)
    
    # 통계
    n_nonzero = (sample_submission['value'] > 0).sum()
    print(f"\n통계:")
    print(f"  비영 예측: {n_nonzero}개 ({n_nonzero/len(sample_submission)*100:.1f}%)")
    print(f"  평균 예측값: {sample_submission['value'].mean():.2f}")
    print(f"  최대 예측값: {sample_submission['value'].max():.2f}")
    print(f"\n저장: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
