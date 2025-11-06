"""
preprocess.py 모듈에 대한 단위 테스트
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import pivot_data, check_negative_values, log_outliers


class TestPivotData:
    """pivot_data 함수 테스트"""

    def test_pivot_basic(self):
        """기본 피벗 기능 테스트"""
        # Given: Long format 데이터
        df_long = pd.DataFrame({
            'date': pd.to_datetime(['2022-01-01', '2022-01-01', '2022-02-01', '2022-02-01']),
            'item_code': ['A', 'B', 'A', 'B'],
            'value': [100, 200, 150, 250]
        })

        # When: Wide format으로 변환
        df_wide = pivot_data(df_long)

        # Then: 올바른 형태로 변환되었는지 확인
        assert df_wide.shape == (2, 2)  # 2 dates x 2 items
        assert list(df_wide.columns) == ['A', 'B']
        assert df_wide.loc[pd.to_datetime('2022-01-01'), 'A'] == 100
        assert df_wide.loc[pd.to_datetime('2022-02-01'), 'B'] == 250

    def test_pivot_with_custom_columns(self):
        """커스텀 컬럼명으로 피벗 테스트"""
        # Given: 다른 컬럼명을 사용하는 데이터
        df_long = pd.DataFrame({
            'timestamp': pd.to_datetime(['2022-01-01', '2022-01-01']),
            'product': ['X', 'Y'],
            'amount': [10, 20]
        })

        # When: 커스텀 컬럼명 지정
        df_wide = pivot_data(df_long, date_col='timestamp',
                           item_col='product', value_col='amount')

        # Then
        assert df_wide.shape == (1, 2)
        assert list(df_wide.columns) == ['X', 'Y']


class TestCheckNegativeValues:
    """check_negative_values 함수 테스트"""

    def test_no_negatives(self):
        """음수가 없는 경우"""
        # Given: 모두 양수인 데이터
        df = pd.DataFrame({
            'A': [10, 20, 30],
            'B': [5, 15, 25]
        })

        # When & Then: 에러 없이 실행
        result = check_negative_values(df)
        assert result.equals(df)

    def test_with_negatives(self, caplog):
        """음수가 있는 경우"""
        # Given: 음수를 포함하는 데이터
        df = pd.DataFrame({
            'A': [10, -5, 30],
            'B': [5, 15, -10]
        })

        # When: 체크 실행
        result = check_negative_values(df)

        # Then: 경고 로그가 출력되었는지 확인
        assert "음수 값 발견" in caplog.text
        assert result.equals(df)  # 데이터는 변경되지 않음


class TestLogOutliers:
    """log_outliers 함수 테스트"""

    def test_no_outliers(self):
        """이상치가 없는 경우"""
        # Given: 정규 분포 데이터
        np.random.seed(42)
        df = pd.DataFrame({
            'A': np.random.normal(100, 10, 50),
            'B': np.random.normal(200, 20, 50)
        })

        # When & Then: 에러 없이 실행
        result = log_outliers(df)
        assert result.equals(df)

    def test_with_outliers(self, caplog):
        """이상치가 있는 경우"""
        # Given: 이상치를 포함하는 데이터
        df = pd.DataFrame({
            'A': [10, 20, 30, 40, 1000]  # 1000이 이상치
        })

        # When: 체크 실행
        result = log_outliers(df, iqr_multiplier=1.5)

        # Then: 경고 로그 확인
        assert "이상치 발견" in caplog.text
        assert result.equals(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
