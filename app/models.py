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
