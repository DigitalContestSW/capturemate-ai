# CaptureMate AI

CaptureMate AI는 CaptureMate Android 앱을 위한 FastAPI 기반 백엔드 서버입니다.

여러 스크린샷 이미지를 받아 **OCR → 민감정보 마스킹 → 유사한 것끼리 그룹핑 → LLM 분석**을
거쳐, 그룹당 하나의 메모(제목·요약·카테고리·추천 액션·세부 정보)를 반환합니다.
이미지는 서버에 저장하지 않고 메모리에서만 처리 후 즉시 폐기합니다.

## 주요 기능

- 여러 이미지 배치 처리 (하루치 스크린샷을 한 요청으로)
- 백엔드 OCR (PaddleOCR, 한국어)
- 민감정보 마스킹 (티어링 + 문맥 기반 복원, 선택적 Presidio NER)
- 임베딩 유사도 + 시간 근접도 기반 그룹핑
- LLM 2단계 분석 (분류·유용성 판단 → 카테고리별 세부 추출)
- Google ID token 검증 + stateless JWT 인증
- 무중단 폴백 (LLM/OCR/임베딩 실패 시에도 응답)

## 기술 스택

- Python 3.12+, FastAPI, Pydantic, Uvicorn
- OCR: PaddleOCR (선택 의존성)
- LLM/임베딩: Gemini (공급자 교체 가능한 인터페이스)
- Auth: Google ID token verification, HS256 Access/Refresh JWT
- (선택) 마스킹 강화: Presidio + spaCy

## 프로젝트 구조

계층: **입력(OCR) → 보호(마스킹) → 지능(그룹핑·LLM)**

```text
app/
├─ main.py              # FastAPI 엔드포인트 (/health/*, /v1/auth/*, /v1/analyze)
├─ config.py            # .env 설정 로드
├─ models.py            # 요청/응답 DTO (Pydantic)
├─ auth.py              # Google ID token 검증 + JWT 발급/검증
├─ rate_limit.py        # Gemini 호출 레이트리밋(공유)
│
├─ ocr/                 # [입력] 이미지 → 텍스트
│  ├─ base.py           #   OcrEngine 인터페이스
│  └─ paddle_engine.py  #   PaddleOCR 구현
│
├─ privacy.py           # [보호] 마스킹(하드/소프트 티어링) + 복원
├─ privacy_ner.py       #   (선택) Presidio 이름/주소 마스킹
│
├─ embeddings/          # [지능] 유사도 임베딩 (base/factory/gemini)
├─ llm/                 # [지능] LLM 생성 (base/factory/gemini)
│
└─ analysis/            # [지능] 파이프라인 조립
   ├─ batch.py          #   BatchAnalyzer: 마스킹→임베딩→그룹핑→그룹별 분석
   ├─ preprocess.py     #   임베딩용 노이즈 제거
   ├─ grouping.py       #   cosine × 시간가중 클러스터링
   ├─ service.py        #   AnalysisService: 1·2단계 LLM 분석 + 복원
   ├─ prompt.py         #   1단계 분류/유용성 프롬프트
   ├─ fallback.py       #   LLM 실패 시 키워드 분류
   └─ categories/       #   2단계 카테고리별 세부추출
```

인터페이스 격리(`OcrEngine`/`EmbeddingClient`/`LlmClient`)로 구현체(PaddleOCR·Gemini 등)를
factory에서 교체할 수 있습니다.

## 처리 흐름

```text
POST /v1/analyze (Bearer Access JWT + 이미지 여러 장)
 → 각 이미지: OCR (실패 시 그 장만 스킵) → 유용성 안전망(글자 < 10자 제외)
 → BatchAnalyzer:
     ① 마스킹        (하드=[RRN]/[CARD] 영구, 소프트=[PHONE_1] 복원가능)
     ② 임베딩·그룹핑  (전처리 → embedding → cosine × 시간가중)
     ③ 그룹별 LLM     (1단계 분류+유용성 → 2단계 세부 → 소프트 토큰 복원)
 → { groups: [ { memberClientIds, analysis } ] }
```

## 설치

```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt          # 핵심(API + LLM)
python -m pip install -r requirements-ocr.txt      # 백엔드 OCR(PaddleOCR)
# (선택) 이름/주소 마스킹: python -m pip install -r requirements-ner.txt
```

`.env.example`을 복사해 `.env`를 만들고 `LLM_API_KEY`, `GOOGLE_WEB_CLIENT_ID`,
`JWT_ACCESS_SECRET`, `JWT_REFRESH_SECRET`, OCR 운영 설정 등을 채웁니다.

## 실행

```powershell
py -m uvicorn app.main:app --reload --port 8001
```

에뮬레이터에서는 로컬 PC 서버를 `http://10.0.2.2:8001`로 접근합니다.
`http://localhost:8001/docs`에서 API를 테스트할 수 있습니다.

## API

### `GET /health`
서버 상태 확인. 응답: `{ "status": "ok", "llmEnabled": true|false, "ocrReady": true|false }`

### `GET /health/live`
프로세스 생존 확인용 엔드포인트입니다. 로드밸런서 readiness가 아니라 liveness 확인에 사용합니다.

### `GET /health/ready`
트래픽 수신 준비 상태 확인용 엔드포인트입니다. PaddleOCR 모델 로딩이 끝나면 `200`, 아직 준비되지 않았거나
로딩에 실패하면 `503`을 반환합니다. ECS health check에는 이 경로를 사용합니다.

### `POST /v1/auth/google` (application/json)
Android Google 로그인에서 받은 Google ID token을 서버 JWT로 교환합니다.

요청:
```json
{ "idToken": "google-id-token" }
```

응답:
```json
{
  "accessToken": "...",
  "refreshToken": "...",
  "tokenType": "Bearer",
  "accessExpiresIn": 1800,
  "refreshExpiresIn": 259200
}
```

### `POST /v1/auth/refresh` (application/json)
refresh JWT로 새 access JWT를 발급합니다. 서버 세션 저장소가 없는 stateless 구조라
refresh token rotation이나 개별 토큰 revoke는 하지 않습니다.

요청:
```json
{ "refreshToken": "..." }
```

응답:
```json
{
  "accessToken": "...",
  "refreshToken": null,
  "tokenType": "Bearer",
  "accessExpiresIn": 1800,
  "refreshExpiresIn": null
}
```

### `POST /v1/analyze` (multipart/form-data)
여러 이미지를 분석해 그룹별 메모를 반환합니다.

헤더:
- `Authorization: Bearer <accessToken>`

요청 필드:
- `images`: 이미지 여러 장
- `locale`: (선택) 기본 `ko-KR`
- `metadata`: (선택) 이미지 순서와 1:1 JSON 배열
  `[{"clientId": "a.png", "capturedAt": 1760000000000}, ...]`

응답:
```json
{
  "groups": [
    {
      "memberClientIds": ["a.png", "b.png"],
      "analysis": {
        "title": "OO 장학금 신청 마감",
        "summary": "...",
        "category": "schedule",
        "isUseful": true,
        "recommendedAction": "캘린더에 일정 추가",
        "reminderAt": 1783695599000,
        "details": { "startAtIso": "...", "location": "..." }
      }
    }
  ]
}
```

카테고리: `schedule`, `study`, `life_info`, `restaurant`, `shopping` (+ 안전용 `unknown`)

## 개인정보 처리 방향

- 원본 이미지는 **서버에 저장하지 않고** 메모리에서만 처리 후 즉시 폐기.
- OCR 텍스트는 LLM 호출 전에 마스킹. 주민번호·카드 등 초민감정보는 **영구 마스킹**,
  전화·이메일·이름·주소 등은 인덱스 토큰으로 가린 뒤 **LLM이 문맥상 필요로 한 것만 복원**.
