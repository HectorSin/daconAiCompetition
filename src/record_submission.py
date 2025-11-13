"""
제출 이력 관리 스크립트

제출 파일을 버전별로 저장하고 점수를 기록합니다.
"""
import sys
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config


def record_submission(score: float,
                     description: str = "",
                     submission_file: Path = None):
    """
    제출 결과를 기록합니다.

    Parameters:
    -----------
    score : float
        대회 점수
    description : str
        제출 설명
    submission_file : Path
        제출 파일 경로 (None이면 output/submission.csv 사용)
    """
    if submission_file is None:
        submission_file = Config.OUTPUT_DIR / 'submission.csv'

    if not submission_file.exists():
        print(f"오류: {submission_file}이 존재하지 않습니다.")
        return

    # submissions 디렉토리 생성
    submissions_dir = Config.OUTPUT_DIR / 'submissions'
    submissions_dir.mkdir(parents=True, exist_ok=True)

    # 제출 이력 파일
    history_file = submissions_dir / 'submission_history.csv'

    # 버전 번호 결정
    if history_file.exists():
        history = pd.read_csv(history_file)
        version = len(history) + 1
    else:
        version = 1

    # 타임스탬프
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 버전별 파일명
    version_filename = f'v{version}_score_{score}_{timestamp}.csv'
    version_filepath = submissions_dir / version_filename

    # 파일 복사
    shutil.copy2(submission_file, version_filepath)
    print(f"[OK] 제출 파일 저장: {version_filepath}")

    # 메타데이터 저장
    metadata = {
        'version': version,
        'score': score,
        'timestamp': timestamp,
        'datetime': datetime.now().isoformat(),
        'description': description,
        'filename': version_filename,
        'filepath': str(version_filepath)
    }

    metadata_file = submissions_dir / f'v{version}_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[OK] 메타데이터 저장: {metadata_file}")

    # 제출 이력 업데이트
    new_record = {
        'version': version,
        'score': score,
        'datetime': datetime.now().isoformat(),
        'timestamp': timestamp,
        'description': description,
        'filename': version_filename
    }

    if history_file.exists():
        history = pd.read_csv(history_file)
        history = pd.concat([history, pd.DataFrame([new_record])], ignore_index=True)
    else:
        history = pd.DataFrame([new_record])

    history.to_csv(history_file, index=False)
    print(f"[OK] 제출 이력 업데이트: {history_file}")

    # 통계 출력
    print("\n" + "=" * 60)
    print("제출 이력")
    print("=" * 60)
    print(history.to_string(index=False))

    print("\n" + "=" * 60)
    print("최고 점수")
    print("=" * 60)
    best_idx = history['score'].idxmax()
    best = history.iloc[best_idx]
    print(f"Version: v{best['version']}")
    print(f"Score: {best['score']}")
    print(f"Date: {best['datetime']}")
    print(f"Description: {best['description']}")

    return version


def main():
    """메인 함수"""
    print("=" * 60)
    print("제출 결과 기록")
    print("=" * 60)

    # 점수 입력
    score_input = input("\n점수를 입력하세요: ")
    try:
        score = float(score_input)
    except ValueError:
        print("오류: 유효한 숫자를 입력하세요.")
        return

    # 설명 입력
    description = input("제출 설명 (선택사항): ")

    # 기록
    version = record_submission(score, description)

    print(f"\n✓ Version {version} 기록 완료!")


if __name__ == "__main__":
    main()
