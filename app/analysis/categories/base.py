from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from app.analysis.prompt import LlmAnalysis

# 2단계 프롬프트 빌더의 시그니처.
# (원문 텍스트, 1단계 결과, 로케일, 오늘 날짜) -> 프롬프트 문자열
PromptBuilder = Callable[[str, LlmAnalysis, str, str], str]


@dataclass(frozen=True)
class CategoryStage2:
    """한 카테고리의 2단계 처리 묶음.

    - build_prompt: 그 카테고리 전용 추출 프롬프트를 만드는 함수
    - details_model: 추출 결과를 검증할 Pydantic 모델

    카테고리 담당자는 자기 파일(categories/<name>.py)에서 이 둘만 정의하고
    categories/__init__.py의 레지스트리에 등록하면 된다.
    """

    build_prompt: PromptBuilder
    details_model: type[BaseModel]
