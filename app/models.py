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
