# CaptureMate AI

CaptureMate AI는 CaptureMate Android 앱을 위한 FastAPI 기반 백엔드 서버입니다.

Android 앱은 스크린샷 OCR과 민감정보 마스킹을 기기 안에서 먼저 처리합니다.
이 서버는 마스킹된 텍스트만 전달받아 제목, 요약, 카테고리, 추천 액션,
알림 후보 정보를 반환합니다.

## 주요 기능

- 서버 상태 확인 API
- 마스킹된 OCR 텍스트 분석 API
- Pydantic 기반 요청/응답 검증
- 키워드 기반 간단 분류
- Uvicorn을 통한 로컬 개발 서버 실행

## 기술 스택

- Python 3.14
- FastAPI
- Pydantic
- Uvicorn

## 프로젝트 구조

```text
capturemate-ai/
+-- app/
|   +-- __init__.py
|   +-- main.py        # FastAPI 앱과 분석 로직
|   +-- models.py      # Pydantic 요청/응답 모델
+-- README.md
+-- requirements.txt
```

## 설치

```powershell
py -m pip install -r requirements.txt
```

가상환경을 사용하는 경우:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 실행

```powershell
py -m uvicorn app.main:app --reload --port 8001
```

서버 주소:

```text
http://localhost:8001
```

Android 에뮬레이터에서는 로컬 PC의 서버를 다음 주소로 접근합니다.

```text
http://10.0.2.2:8001
```

## API

### `GET /health`

서버 상태를 확인합니다.

응답:

```json
{
  "status": "ok"
}
```

### `POST /v1/analyze`

Android 앱에서 전달한 마스킹된 OCR 텍스트를 분석합니다.

요청:

```json
{
  "maskedText": "Meeting tomorrow at 3 PM",
  "locale": "ko-KR",
  "clientCapturedAt": 1760000000000
}
```

응답:

```json
{
  "serverMemoId": null,
  "title": "Meeting tomorrow at 3 PM",
  "summary": "Meeting tomorrow at 3 PM",
  "category": "calendar",
  "recommendedAction": "Create calendar item",
  "reminderAt": null
}
```

## 분류 규칙

현재 구현은 간단한 키워드 규칙으로 카테고리를 분류합니다.

- `calendar`: schedule, meeting, deadline, reservation
- `study`: study, exam, lecture, review
- `restaurant`: restaurant, cafe, menu, place
- `job`: job, resume, interview, recruit
- `memo`: 기본 카테고리

## 개인정보 처리 방향

이 백엔드는 마스킹된 텍스트만 받는 것을 전제로 설계되어 있습니다.
원본 스크린샷, 로컬 파일 경로, 마스킹되지 않은 OCR 원문은 Android 기기에만
보관해야 합니다.

