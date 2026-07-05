from typing import Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    maskedText: str = Field(min_length=1)
    locale: str = "ko-KR"
    clientCapturedAt: Optional[int] = None


class AnalyzeResponse(BaseModel):
    serverMemoId: Optional[str] = None
    title: str
    summary: str
    category: str
    recommendedAction: Optional[str] = None
    reminderAt: Optional[int] = None
    # 2단계(카테고리별 추출) 결과를 담는 자리. 카테고리마다 필드가 다르므로 일단
    # 유연한 dict로 둔다. 팀에서 카테고리별 필드가 확정되면 명시적 타입으로 승격.
    # (안드로이드 Json은 ignoreUnknownKeys=true라 이 필드가 늘어도 디코딩이 안 깨진다.)
    details: Optional[dict] = None


class AnalyzeBatchItem(BaseModel):
    clientId: str                       # 응답을 로컬 캡처에 다시 매핑하기 위한 식별자
    maskedText: str = Field(min_length=1)
    capturedAt: Optional[int] = None    # 캡처 시각(epoch ms) — 시간 근접 그룹핑에 사용


class AnalyzeBatchRequest(BaseModel):
    items: list[AnalyzeBatchItem]
    locale: str = "ko-KR"


class MemoGroup(BaseModel):
    memberClientIds: list[str]          # 이 메모로 묶인 스크린샷들의 clientId
    analysis: AnalyzeResponse           # 그룹 대표 분석 결과(그룹당 메모 1개)


class AnalyzeBatchResponse(BaseModel):
    groups: list[MemoGroup]


class AnalyzeImageResponse(BaseModel):
    """백엔드 OCR 테스트용 응답.

    이미지 -> OCR -> 마스킹 -> LLM 전체 파이프라인의 중간 결과를 모두 노출해,
    OCR이 무엇을 추출했고 LLM이 그걸 어떻게 복원·요약했는지 한눈에 확인한다.
    (프로덕션에서는 원문/마스킹 텍스트를 응답에 담지 않는 게 원칙 — 어디까지나 검증용)
    """

    ocrText: str
    maskedText: str
    analysis: AnalyzeResponse
