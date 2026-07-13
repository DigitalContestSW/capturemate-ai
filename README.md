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

## Docker

로컬 컨테이너 빌드:

```bash
docker build -t capturemate-ai:local .
```

로컬 컨테이너 실행:

```bash
docker run --rm -p 8001:8001 \
  -e LLM_API_KEY="$LLM_API_KEY" \
  -e GOOGLE_WEB_CLIENT_ID="$GOOGLE_WEB_CLIENT_ID" \
  -e JWT_ACCESS_SECRET="$JWT_ACCESS_SECRET" \
  -e JWT_REFRESH_SECRET="$JWT_REFRESH_SECRET" \
  -e KAKAO_REST_API_KEY="$KAKAO_REST_API_KEY" \
  capturemate-ai:local
```

Apple Silicon Mac에서 ECR/ECS용 이미지를 로컬 검증할 때는 linux/amd64로 빌드합니다.

```bash
docker buildx build --platform linux/amd64 --build-arg BAKE_OCR_MODELS=false -t capturemate-ai:amd64 --load .
```

Apple Silicon에서 `linux/amd64` 이미지를 QEMU로 빌드할 때는 PaddleOCR 모델 로딩이 실패할 수 있어
`BAKE_OCR_MODELS=false`를 사용합니다. Native amd64 CI/빌더에서는 기본값(`true`)으로 모델을 이미지에
미리 포함할 수 있습니다.

컨테이너 health check:

```bash
curl http://localhost:8001/health/live
curl http://localhost:8001/health/ready
```

현재 Docker 이미지는 배포 안정성을 우선해 OCR/NER 의존성을 함께 설치합니다. 이미지 크기 최적화는
후속 작업에서 runtime requirements 분리로 진행합니다.

## 배포

현재 경진대회 검증용 배포 endpoint:

```text
https://api.cloudnetaaws.click
```

구성:

```text
Android
 → Route 53 (api.cloudnetaaws.click)
 → Application Load Balancer (HTTPS :443)
 → ECS Fargate task (HTTP :8001)
 → FastAPI
```

AWS 리소스:

- Region: `ap-northeast-2`
- ECR repository: `974346102227.dkr.ecr.ap-northeast-2.amazonaws.com/capturemate-ai`
- ECS cluster/service/task family: `capturemate-ai`
- ALB: `capturemate-ai-alb`
- ALB DNS: `capturemate-ai-alb-1247019725.ap-northeast-2.elb.amazonaws.com`
- Target group: `capturemate-ai-tg`
- Route 53 hosted zone: `cloudnetaaws.click`
- API domain: `api.cloudnetaaws.click`
- ACM certificate region: `ap-northeast-2`
- CloudWatch log group: `/ecs/capturemate-ai`

현재 배포된 이미지 태그:

```text
974346102227.dkr.ecr.ap-northeast-2.amazonaws.com/capturemate-ai:605edc8
```

운영 환경변수는 ECS task definition에서 관리합니다. `LLM_API_KEY`, `JWT_ACCESS_SECRET`,
`JWT_REFRESH_SECRET`, `KAKAO_REST_API_KEY` 같은 민감값은 Secrets Manager `ValueFrom`으로 주입합니다.
실제 secret 값은 저장소와 문서에 기록하지 않습니다.

배포 확인:

```bash
curl https://api.cloudnetaaws.click/health/live
curl https://api.cloudnetaaws.click/health/ready
```

로컬에서 새 이미지를 ECR에 올리는 절차:

```bash
aws ecr get-login-password --profile capturemate --region ap-northeast-2 \
  | docker login --username AWS --password-stdin 974346102227.dkr.ecr.ap-northeast-2.amazonaws.com

docker buildx build --platform linux/amd64 --build-arg BAKE_OCR_MODELS=false -t capturemate-ai:amd64 --load .

docker tag capturemate-ai:amd64 974346102227.dkr.ecr.ap-northeast-2.amazonaws.com/capturemate-ai:<commit-sha>
docker push 974346102227.dkr.ecr.ap-northeast-2.amazonaws.com/capturemate-ai:<commit-sha>
```

Apple Silicon 로컬 빌드는 `BAKE_OCR_MODELS=false`를 사용합니다. 최종 안정화용 이미지는 native amd64
빌더나 CI에서 기본값(`BAKE_OCR_MODELS=true`)으로 OCR 모델을 이미지에 포함해 빌드하는 것을 권장합니다.

기존 API Gateway HTTP API endpoint는 29~30초 통합 timeout 제한 때문에 최종 Android 배포 경로에서
사용하지 않습니다. Android release build는 `https://api.cloudnetaaws.click/`을 기본 backend URL로
사용합니다.

보안 그룹은 다음 구조를 기준으로 관리합니다.

- ALB security group: `80`, `443` inbound from `0.0.0.0/0`
- Fargate security group: `8001` inbound from ALB security group only

Fargate task의 public IP는 outbound 의존성(ECR pull, Secrets Manager, CloudWatch Logs, 모델 다운로드 등)을
고려해 유지할 수 있습니다. 외부 inbound 접근은 security group에서 ALB source로 제한합니다.

CI/CD는 GitHub Actions로 구성합니다. `main` 브랜치에 push되거나 수동 실행하면 테스트, linux/amd64
이미지 빌드, ECR push, ECS service 자동 배포, 배포 후 health check를 순서대로 수행합니다.

### GitHub Actions CI/CD

`.github/workflows/backend-ci-cd.yml`은 다음을 수행합니다.

- PR to `main`: Python 의존성 설치, py_compile, unit test
- Push to `main`: 위 검증 후 linux/amd64 Docker image build, ECR push, ECS service deploy
- Manual dispatch: OCR 모델 bake 여부를 선택해 image build, ECR push, ECS service deploy

이미지 태그:

- `<short-commit-sha>`
- `latest` (`main` push일 때만)

배포 job은 현재 ECS task definition을 조회한 뒤 `capturemate-ai` 컨테이너의 image만 새 ECR image로 교체해
새 revision을 등록합니다. 따라서 ECS 콘솔에서 설정한 환경변수, Secrets Manager 참조, 로그 설정, CPU/메모리,
포트 매핑은 그대로 유지됩니다.

배포 대상:

```text
ECS_CLUSTER=capturemate-ai
ECS_SERVICE=capturemate-ai
ECS_TASK_DEFINITION=capturemate-ai
ECS_CONTAINER_NAME=capturemate-ai
HEALTHCHECK_URL=https://api.cloudnetaaws.click/health/ready
```

GitHub Actions가 AWS에 접근하려면 repository secret을 설정합니다.

```text
AWS_ROLE_TO_ASSUME=arn:aws:iam::974346102227:role/<github-actions-backend-deploy-role>
```

권장 방식은 GitHub OIDC AssumeRole입니다. AWS access key를 GitHub secret에 저장하지 않습니다.
OIDC role trust policy는 이 repository의 `main` 브랜치만 허용하도록 제한합니다.

예시 trust policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::974346102227:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:DigitalContestSW/capturemate-ai:ref:refs/heads/main"
        }
      }
    }
  ]
}
```

Role permission policy는 ECR push와 ECS service 배포에 필요한 범위로 제한합니다. `iam:PassRole`의
resource에는 ECS task definition에 설정된 task execution role과 task role ARN을 지정합니다.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:CompleteLayerUpload",
        "ecr:DescribeRepositories",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:UploadLayerPart"
      ],
      "Resource": "arn:aws:ecr:ap-northeast-2:974346102227:repository/capturemate-ai"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecs:DescribeTaskDefinition",
        "ecs:RegisterTaskDefinition"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecs:DescribeServices",
        "ecs:UpdateService"
      ],
      "Resource": "arn:aws:ecs:ap-northeast-2:974346102227:service/capturemate-ai/capturemate-ai"
    },
    {
      "Effect": "Allow",
      "Action": [
        "iam:PassRole"
      ],
      "Resource": "arn:aws:iam::974346102227:role/ecsTaskExecutionRole"
    }
  ]
}
```

## API

### `GET /health`
서버 상태 확인. 응답: `{ "status": "ok", "llmEnabled": true|false, "ocrReady": true|false }`

### `GET /health/live`
프로세스 생존 확인용 엔드포인트입니다. 로드밸런서 readiness가 아니라 liveness 확인에 사용합니다.

### `GET /health/ready`
트래픽 수신 준비 상태 확인용 엔드포인트입니다. PaddleOCR 모델 로딩이 끝나면 `200`, 아직 준비되지 않았거나
로딩에 실패하면 `503`을 반환합니다. ALB/ECS liveness health check는 빠르게 응답하는 `/health/live`를
사용하고, 배포 후 readiness 검증에는 이 경로를 사용합니다.

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
