"""
LightGBM categorical_feature 에러 진단 스크립트

LightGBM의 categorical_feature 파라미터 사용법을 테스트합니다.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

print("=" * 60)
print("LightGBM categorical_feature 테스트")
print("=" * 60)

# 1. 데이터 로드
print("\n[1] 데이터 로드 중...")
df_raw = pd.read_csv(Config.DATA_RAW / 'train.csv')
df_raw['date'] = pd.to_datetime(df_raw[['year', 'month']].assign(day=1))

df_agg = df_raw.groupby(['date', 'item_id']).agg({
    'value': 'sum'
}).reset_index()

df_wide = df_agg.pivot(index='date', columns='item_id', values='value').fillna(0)
print(f"데이터 shape: {df_wide.shape}")

# 2. 간단한 Global Dataset 생성 (테스트용)
print("\n[2] 테스트용 Global Dataset 생성...")
all_data = []

for item in df_wide.columns[:5]:  # 처음 5개 품목만 테스트
    item_df = pd.DataFrame({
        'date': df_wide.index,
        'item_id': item,
        'value': df_wide[item].values
    })
    
    # Lag 특징
    item_df['lag_1'] = item_df['value'].shift(1)
    item_df['lag_3'] = item_df['value'].shift(3)
    
    all_data.append(item_df)

global_df = pd.concat(all_data, ignore_index=True)
print(f"Global Dataset: {global_df.shape}")
print(f"\n컬럼: {global_df.columns.tolist()}")

# 3. item_id 인코딩
print("\n[3] item_id 인코딩...")
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
global_df['item_id_encoded'] = label_encoder.fit_transform(global_df['item_id'])

print(f"item_id 고유값: {global_df['item_id'].nunique()}개")
print(f"item_id_encoded 범위: {global_df['item_id_encoded'].min()} ~ {global_df['item_id_encoded'].max()}")
print(f"\n샘플 데이터:")
print(global_df[['item_id', 'item_id_encoded', 'value', 'lag_1']].head(10))

# 4. 학습 데이터 준비
print("\n[4] 학습 데이터 준비...")
df_clean = global_df.dropna()
print(f"NaN 제거 후: {len(df_clean)}행")

X = df_clean[['lag_1', 'lag_3', 'item_id_encoded']].copy()
y = df_clean['value'].copy()

print(f"\nX shape: {X.shape}")
print(f"X dtypes:\n{X.dtypes}")
print(f"\nX 샘플:")
print(X.head())

# 5. LightGBM 테스트 (3가지 방법)
print("\n" + "=" * 60)
print("LightGBM categorical_feature 테스트")
print("=" * 60)

# 방법 1: 잘못된 방법 (에러 발생)
print("\n[방법 1] categorical_feature=['item_id_encoded'] (에러 예상)")
try:
    model1 = LGBMRegressor(
        n_estimators=10,
        random_state=42,
        verbose=-1,
        categorical_feature=['item_id_encoded']
    )
    model1.fit(X, y)
    print("✅ 성공!")
except Exception as e:
    print(f"❌ 에러: {e}")

# 방법 2: 올바른 방법 - "name:" 접두사 사용
print("\n[방법 2] categorical_feature=['name:item_id_encoded'] (올바른 방법)")
try:
    model2 = LGBMRegressor(
        n_estimators=10,
        random_state=42,
        verbose=-1,
        categorical_feature=['name:item_id_encoded']
    )
    model2.fit(X, y)
    print("✅ 성공!")
except Exception as e:
    print(f"❌ 에러: {e}")

# 방법 3: 컬럼 인덱스 사용
print("\n[방법 3] categorical_feature=[2] (인덱스 사용)")
try:
    model3 = LGBMRegressor(
        n_estimators=10,
        random_state=42,
        verbose=-1,
        categorical_feature=[2]  # item_id_encoded는 3번째 컬럼 (인덱스 2)
    )
    model3.fit(X, y)
    print("✅ 성공!")
except Exception as e:
    print(f"❌ 에러: {e}")

# 방법 4: fit() 메서드에서 지정
print("\n[방법 4] fit(categorical_feature=['name:item_id_encoded'])")
try:
    model4 = LGBMRegressor(
        n_estimators=10,
        random_state=42,
        verbose=-1
    )
    model4.fit(X, y, categorical_feature=['name:item_id_encoded'])
    print("✅ 성공!")
except Exception as e:
    print(f"❌ 에러: {e}")

print("\n" + "=" * 60)
print("결론: 'name:' 접두사를 사용하거나 컬럼 인덱스를 사용해야 합니다!")
print("=" * 60)
