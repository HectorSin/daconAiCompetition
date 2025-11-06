"""
패키지 설치 검증 스크립트

주요 라이브러리들이 올바르게 설치되었는지 확인합니다.
"""
import sys
from typing import List, Tuple


def verify_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    패키지 설치를 검증합니다.

    Parameters:
    -----------
    package_name : str
        패키지 이름 (표시용)
    import_name : str
        실제 import 이름 (None이면 package_name 사용)

    Returns:
    --------
    Tuple[bool, str]
        (성공 여부, 버전 정보)
    """
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)


def main():
    """메인 검증 함수"""
    print("=" * 60)
    print("패키지 설치 검증")
    print("=" * 60)

    # 검증할 패키지 목록
    packages = [
        # Core
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('jupyterlab', 'jupyterlab'),

        # Modeling (ML)
        ('lightgbm', 'lightgbm'),
        ('xgboost', 'xgboost'),
        ('catboost', 'catboost'),

        # Modeling (Time-Series)
        ('statsmodels', 'statsmodels'),
        ('prophet', 'prophet'),
        ('dtw-python', 'dtw'),
        ('sktime', 'sktime'),

        # Hyperparameter Optimization
        ('optuna', 'optuna'),

        # Experiment Tracking
        ('mlflow', 'mlflow'),

        # Testing
        ('pytest', 'pytest'),

        # Visualization
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),

        # Utility
        ('tqdm', 'tqdm'),
        ('pyarrow', 'pyarrow'),
    ]

    results = []
    failed = []

    for package_name, import_name in packages:
        success, info = verify_package(package_name, import_name)
        results.append((package_name, success, info))

        if not success:
            failed.append(package_name)

    # 결과 출력
    print("\n검증 결과:")
    print("-" * 60)

    for package_name, success, info in results:
        status = "✓" if success else "✗"
        if success:
            print(f"{status} {package_name:20s} (v{info})")
        else:
            print(f"{status} {package_name:20s} - 설치 안됨")

    print("-" * 60)

    # 요약
    print(f"\n총 {len(packages)}개 패키지 중 {len(packages) - len(failed)}개 설치됨")

    if failed:
        print(f"\n❌ 설치 실패: {', '.join(failed)}")
        print("\n다음 명령으로 설치하세요:")
        print("  pip install -r requirements.txt")
        return 1
    else:
        print("\n✓ 모든 패키지가 정상적으로 설치되었습니다!")

        # Python 버전 확인
        print(f"\nPython 버전: {sys.version}")

        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
