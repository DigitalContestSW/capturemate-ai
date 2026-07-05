from typing import Optional

from pydantic import BaseModel

from app.analysis.prompt import LlmAnalysis

# ── 학습(study) 담당자가 채워 확장하는 파일 ──────────────────────────────────
# 와이어프레임: 핵심 정리 + 복습 리마인드
#   - 핵심 정리 -> LLM이 keyPoints로 추출 (여기)
#   - 복습 리마인드 시각 -> 1단계 결과의 reminderAt 사용, 알림 예약은 안드로이드(WorkManager)


class StudyDetails(BaseModel):
    """학습 세부: 핵심 정리. (담당자가 자유롭게 추가/수정)"""

    keyPoints: list[str] = []                   # 복습용 핵심 정리 항목
    recommendedAction: Optional[str] = None     # 카테고리 맞춤 추천 다음 행동


def build_study_prompt(text: str, base: LlmAnalysis, locale: str, today_iso: str) -> str:
    return f"""이 텍스트는 '학습'으로 분류되었습니다. 복습에 쓸 '핵심 정리'를 뽑아
아래 필드를 추출하세요. JSON 객체 하나만 출력(설명·코드펜스 없이).

JSON 스키마:
{{
  "keyPoints": ["핵심 정리 항목1", "핵심 정리 항목2"],
  "recommendedAction": "추천 다음 행동 (예: 복습 노트 저장, 시험 전 리마인드 설정)"
}}

keyPoints는 시험/복습에 중요한 개념·범위·요점을 짧은 문장으로 3~7개 정리하세요.

텍스트:
\"\"\"
{text}
\"\"\"
"""
