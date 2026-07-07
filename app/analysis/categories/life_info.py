from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.analysis.prompt import LlmAnalysis

# ── 생활정보(life_info) 담당자가 채워 확장하는 파일 ─────────────────────────
# 와이어프레임: "핵심 정보" 카드 = 혜택/대상/신청/마감 4개 고정 라벨 + 리마인드
#   - 혜택/대상/신청/마감 -> LLM이 각각 추출 (여기). 라벨이 고정이라 안드로이드가
#     자유 텍스트를 파싱하지 않고 필드별로 바로 렌더링한다.
#   - 마감(deadlineText)은 카드에 보여줄 표시용 텍스트일 뿐, 실제 알림 시각은
#     1단계 결과의 reminderAt(epoch ms)을 그대로 쓴다. 여기서 다시 계산하지 않는다.


class LifeInfoRecommendedAction(BaseModel):
    """카드 하단에 보여줄 구체적인 다음 행동 제안 (restaurant.py의 패턴을 따름)."""

    type: Literal["set_reminder", "save_info", "visit_site", "other"] = "other"
    title: str
    description: Optional[str] = None


class LifeInfoDetails(BaseModel):
    """생활정보 세부: 혜택/대상/신청/마감 카드. (담당자가 자유롭게 추가/수정)"""

    benefit: Optional[str] = None                # 혜택 (예: "월 10만원 저축 시 정부 매칭 30만원 → 3년 후 1,440만원")
    target: Optional[str] = None                 # 대상 (예: "만 19~34세 근로·사업소득 있는 청년")
    applyMethod: Optional[str] = None            # 신청 방법 (예: "복지로 또는 행복e음 사이트")
    deadlineText: Optional[str] = None           # 마감 표시용 텍스트 (예: "2026년 7월 31일")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    needsUserReview: bool = False
    recommendedActions: list[LifeInfoRecommendedAction] = Field(default_factory=list)
    recommendedAction: Optional[str] = None      # 카테고리 맞춤 추천 다음 행동 (최상위 단일 요약용)

    @model_validator(mode="after")
    def fill_derived_flags(self) -> "LifeInfoDetails":
        has_core_info = bool(self.benefit or self.target)
        if not has_core_info:
            self.needsUserReview = True
            self.confidence = min(self.confidence, 0.6)
        return self


def build_life_info_prompt(text: str, base: LlmAnalysis, locale: str, today_iso: str) -> str:
    return f"""다음 텍스트는 '생활정보' 카테고리로 분류되었습니다.
Android 앱이 "핵심 정보" 카드(혜택/대상/신청/마감 4개 고정 라벨)를 그대로 렌더링할 수 있도록
아래 필드를 추출하세요.

반드시 JSON 객체 하나만 출력하세요. 설명, 마크다운, 코드펜스는 출력하지 마세요.
텍스트에 근거가 없거나 확신이 낮은 값은 null로 두세요.
전화번호, 이메일, 계좌번호, 주민등록번호 같은 민감정보는 복원하거나 추측하지 마세요.

오늘 날짜: {today_iso}
사용자 로케일: {locale}

JSON 스키마:
{{
  "benefit": "핵심 혜택 한 문장. 금액·배율·조건이 있으면 구체적으로 (예: '월 10만원 저축 시 정부 매칭 30만원 → 3년 후 1,440만원'). 없으면 null",
  "target": "지원 대상/자격 조건 (예: '만 19~34세 근로·사업소득 있는 청년'), 없으면 null",
  "applyMethod": "신청 방법/경로 (예: '복지로 또는 행복e음 사이트'), 없으면 null",
  "deadlineText": "마감일을 사람이 읽기 좋은 문자열로 (예: '2026년 7월 31일'), 없으면 null",
  "confidence": 0.86,
  "needsUserReview": false,
  "recommendedActions": [
    {{
      "type": "set_reminder",
      "title": "마감 3일 전 알림 설정",
      "description": "신청 마감을 놓치지 않도록 미리 알림을 받으세요"
    }}
  ],
  "recommendedAction": "정보 저장 후 마감 전 알림 설정"
}}

추출 규칙:
- benefit/target/applyMethod/deadlineText는 텍스트에 실제 근거가 있을 때만 채우세요.
- deadlineText는 표시용 텍스트일 뿐입니다. 실제 알림 시각 계산은 다른 필드(reminderAtIso)에서
  이미 처리하므로 여기서는 신경 쓰지 마세요.
- benefit과 target 둘 다 근거가 없으면 needsUserReview를 true로, confidence를 0.6 이하로 두세요.
- recommendedActions는 최대 3개로 제한하고, 마감이 있으면 "set_reminder" 액션을 우선 포함하세요.

텍스트:
\"\"\"
{text}
\"\"\"
"""
