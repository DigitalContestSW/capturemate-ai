from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.analysis.prompt import LlmAnalysis

# ── 학습(study) 담당자가 채워 확장하는 파일 ──────────────────────────────────
# 와이어프레임: 핵심 정리 + 복습 리마인드
#   - 핵심 정리 -> LLM이 keyPoints로 추출 (여기)
#   - 복습 리마인드 시각 -> 1단계 결과의 reminderAt 사용, 알림 예약은 안드로이드(WorkManager)


class StudyKeyPoint(BaseModel):
    """학습 카드에 번호와 함께 보여줄 핵심 정리 항목."""

    order: int
    text: str


class StudyDetails(BaseModel):
    """학습 세부: 학습 카드 렌더링 + 기존 핵심 정리 호환 필드."""

    type: Literal["study"] = "study"
    title: Optional[str] = None                 # 카드 제목
    summary: Optional[str] = None               # 카드 내부 요약
    keyPoints: list[str] = Field(default_factory=list)  # 기존 클라이언트 호환용 문자열 목록
    keyPointItems: list[StudyKeyPoint] = Field(default_factory=list)  # 카드 렌더링용 번호 목록
    recommendedAction: Optional[str] = None     # 카테고리 맞춤 추천 다음 행동


def build_study_prompt(text: str, base: LlmAnalysis, locale: str, today_iso: str) -> str:
    return f"""이 텍스트는 '학습'으로 분류되었습니다.
Android 앱이 학습 카드를 그대로 렌더링하고 복습 메모로 저장할 수 있도록 아래 필드를 추출하세요.

반드시 JSON 객체 하나만 출력하세요. 설명, 마크다운, 코드펜스는 출력하지 마세요.
텍스트에 근거가 없거나 확신이 낮은 값은 null 또는 빈 배열로 두세요. 없는 정보를 추측하지 마세요.
전화번호, 이메일, 계좌번호, 주민등록번호 같은 민감정보는 복원하거나 추측하지 마세요.

오늘 날짜: {today_iso}
사용자 로케일: {locale}

JSON 스키마:
{{
  "type": "study",
  "title": "학습 카드 제목. 예: '알고리즘 스터디 계획'",
  "summary": "학습 내용 요약 한 문장",
  "keyPoints": ["기존 클라이언트 호환용 핵심 정리 문자열1", "핵심 정리 문자열2"],
  "keyPointItems": [
    {{
      "order": 1,
      "text": "화·목 오후 8시 디스코드 채널 접속"
    }},
    {{
      "order": 2,
      "text": "BFS/DFS → DP → 그리디 → 분할정복 순"
    }}
  ],
  "recommendedAction": "추천 다음 행동. 예: '알고리즘 문제 10개 풀기'"
}}

작성 규칙:
- title이 애매하면 1단계 제목 "{base.title}"을 사용하세요.
- summary가 애매하면 1단계 요약 "{base.summary}"을 바탕으로 짧게 작성하세요.
- keyPoints와 keyPointItems는 같은 내용을 담되, keyPointItems에는 1부터 시작하는 order를 붙이세요.
- 학습 시간/장소/채널, 커리큘럼 순서, 필수 과제/문제 풀이, 시험 범위를 우선 정리하세요.
- keyPoints는 3~7개를 권장하되, 근거가 적으면 더 적게 작성해도 됩니다.
- recommendedAction은 사용자가 바로 할 수 있는 행동으로 작성하세요.

텍스트:
\"\"\"
{text}
\"\"\"
"""
