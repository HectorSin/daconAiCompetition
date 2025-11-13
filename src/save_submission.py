"""
제출 파일을 버전별로 저장하고 메타데이터 기록
"""
import sys
from pathlib import Path
import shutil
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


def save_submission(version: str, score: float, description: str = ""):
    """
    제출 파일과 메타데이터 저장

    Parameters:
    -----------
    version : str
        버전 이름 (예: "v1", "v2_ccf_05")
    score : float
        대회 점수
    description : str
        설명
    """
    # 디렉토리 생성
    submissions_dir = Config.OUTPUT_DIR / 'submissions'
    submissions_dir.mkdir(parents=True, exist_ok=True)

    # 원본 파일
    source = Config.OUTPUT_DIR / 'submission.csv'
    if not source.exists():
        print(f"Error: {source} not found")
        return

    # 타임스탬프
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 파일명
    csv_name = f'{version}_score_{score}_{timestamp}.csv'
    json_name = f'{version}_score_{score}_{timestamp}.json'

    # CSV 복사
    dest_csv = submissions_dir / csv_name
    shutil.copy2(source, dest_csv)

    # 메타데이터
    metadata = {
        'version': version,
        'score': score,
        'description': description,
        'timestamp': timestamp,
        'datetime': datetime.now().isoformat(),
        'csv_file': csv_name
    }

    # JSON 저장
    dest_json = submissions_dir / json_name
    with open(dest_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved: {csv_name}")
    print(f"Saved: {json_name}")
    print(f"\nScore: {score}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('version', help='Version name (e.g., v1, v2)')
    parser.add_argument('score', type=float, help='Competition score')
    parser.add_argument('--desc', default='', help='Description')

    args = parser.parse_args()
    save_submission(args.version, args.score, args.desc)
