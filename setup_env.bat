@echo off
REM Windows용 환경 설정 스크립트

echo ========================================
echo 데이콘 AI 프로젝트 환경 설정
echo ========================================

REM Conda 환경 생성
echo.
echo [1/3] Conda 환경 생성 중...
call conda create -n daconai python=3.10 -y

if %errorlevel% neq 0 (
    echo ❌ Conda 환경 생성 실패
    pause
    exit /b 1
)

echo ✓ Conda 환경 생성 완료

REM 환경 활성화
echo.
echo [2/3] 환경 활성화 중...
call conda activate daconai

if %errorlevel% neq 0 (
    echo ❌ 환경 활성화 실패
    pause
    exit /b 1
)

echo ✓ 환경 활성화 완료

REM 패키지 설치
echo.
echo [3/3] 패키지 설치 중...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ 패키지 설치 실패
    pause
    exit /b 1
)

echo ✓ 패키지 설치 완료

echo.
echo ========================================
echo 설치 완료!
echo ========================================
echo.
echo 다음 명령으로 환경을 활성화하세요:
echo   conda activate daconai
echo.
echo 설치 검증:
echo   python verify_installation.py
echo.
pause
