"""
경량 실험 로깅 시스템

MLflow 대신 CSV 기반 간단한 실험 추적 시스템.
설정 시간 0분, 즉시 사용 가능하며 Git으로 버전 관리 가능.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json


class ExperimentLogger:
    """경량 실험 로깅 클래스"""
    
    def __init__(self, log_path: str = "experiments/log_experiment.csv"):
        """
        Args:
            log_path: 로그 파일 경로
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.log_path.exists():
            self._create_log_file()
    
    def _create_log_file(self):
        """로그 파일 초기화"""
        df = pd.DataFrame(columns=[
            'timestamp', 'experiment_name', 'model_type',
            'params', 'cv_score', 'cv_std', 'test_score', 'notes'
        ])
        df.to_csv(self.log_path, index=False)
        print(f"✅ 실험 로그 파일 생성: {self.log_path}")
    
    def log(self, experiment_name: str, model_type: str, params: Dict[str, Any],
            cv_score: float, cv_std: float = None, test_score: float = None, notes: str = ""):
        """실험 결과 기록"""
        df = pd.read_csv(self.log_path)
        
        new_log = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_name': experiment_name,
            'model_type': model_type,
            'params': json.dumps(params, ensure_ascii=False),
            'cv_score': cv_score,
            'cv_std': cv_std if cv_std is not None else '',
            'test_score': test_score if test_score is not None else '',
            'notes': notes
        }
        
        df = pd.concat([df, pd.DataFrame([new_log])], ignore_index=True)
        df.to_csv(self.log_path, index=False)
        
        print(f"✅ 실험 기록: {experiment_name} | CV: {cv_score:.4f}" + 
              (f" (±{cv_std:.4f})" if cv_std else ""))
    
    def load(self) -> pd.DataFrame:
        """모든 실험 로그 로드"""
        return pd.read_csv(self.log_path)
    
    def compare(self, metric: str = 'cv_score', top_n: int = 10, ascending: bool = True) -> pd.DataFrame:
        """실험 결과 비교"""
        df = self.load()
        if len(df) == 0:
            print("⚠️ 기록된 실험이 없습니다.")
            return df
        
        df_sorted = df.sort_values(by=metric, ascending=ascending).head(top_n)
        display_cols = ['experiment_name', 'model_type', 'cv_score', 'cv_std', 'test_score', 'timestamp']
        return df_sorted[[col for col in display_cols if col in df_sorted.columns]].reset_index(drop=True)
    
    def get_best(self, metric: str = 'cv_score', ascending: bool = True) -> Dict[str, Any]:
        """최고 성능 실험 반환"""
        df = self.load()
        if len(df) == 0:
            return {}
        
        best_idx = df[metric].idxmin() if ascending else df[metric].idxmax()
        best_exp = df.loc[best_idx].to_dict()
        
        if 'params' in best_exp and isinstance(best_exp['params'], str):
            best_exp['params'] = json.loads(best_exp['params'])
        
        return best_exp
    
    def summary(self):
        """실험 요약 출력"""
        df = self.load()
        if len(df) == 0:
            print("⚠️ 기록된 실험이 없습니다.")
            return
        
        print(f"\n{'='*60}")
        print(f"총 실험 수: {len(df)} | 모델: {', '.join(df['model_type'].unique())}")
        
        best = self.get_best()
        print(f"\n최고 성능: {best.get('experiment_name', 'N/A')} | CV: {best.get('cv_score', 0):.4f}")
        print(f"\n최근 5개 실험:")
        print(df.tail(5)[['timestamp', 'experiment_name', 'cv_score']].to_string(index=False))


# 전역 로거 (싱글톤)
_global_logger = None

def get_logger(log_path: str = "experiments/log_experiment.csv") -> ExperimentLogger:
    """전역 로거 반환"""
    global _global_logger
    if _global_logger is None:
        _global_logger = ExperimentLogger(log_path)
    return _global_logger

def log_experiment(experiment_name: str, model_type: str, params: Dict[str, Any],
                   cv_score: float, cv_std: float = None, test_score: float = None, notes: str = ""):
    """실험 기록 (간편 함수)"""
    get_logger().log(experiment_name, model_type, params, cv_score, cv_std, test_score, notes)

def load_experiments() -> pd.DataFrame:
    """실험 로드"""
    return get_logger().load()

def compare_experiments(metric: str = 'cv_score', top_n: int = 10) -> pd.DataFrame:
    """실험 비교"""
    return get_logger().compare(metric, top_n)

def get_best_experiment(metric: str = 'cv_score') -> Dict[str, Any]:
    """최고 실험 반환"""
    return get_logger().get_best(metric)

def print_experiment_summary():
    """실험 요약"""
    get_logger().summary()


if __name__ == "__main__":
    print("경량 실험 로깅 시스템 테스트\n")
    
    logger = ExperimentLogger("experiments/test_log.csv")
    
    logger.log("baseline_lgbm", "LightGBM", 
               {'n_estimators': 100, 'lr': 0.05}, 1234.56, 123.45, notes="베이스라인")
    logger.log("tuned_lgbm", "LightGBM",
               {'n_estimators': 200, 'lr': 0.03}, 1123.45, 98.76, notes="튜닝 v1")
    
    print("\n상위 실험:")
    print(logger.compare(top_n=2))
    
    logger.summary()
    print("\n✅ 테스트 완료!")
