class PresidioNerDetector:
    """Presidio(+ spaCy 한국어)로 이름/주소 등 자유형 PII를 탐지한다. (선택 기능)

    필요 설치:
        pip install -r requirements-ner.txt
        python -m spacy download ko_core_news_sm

    주의:
      - 무거운 의존성이라 use_presidio=true일 때만, 그리고 최초 사용 시 지연 로드된다.
      - 한국어 NER(PERSON/LOCATION) 품질은 모델에 따라 편차가 크니 실제 데이터로 검증할 것.
      - 여기서 탐지한 것은 privacy.mask_text에서 '소프트 토큰'(복원 가능)으로 처리된다.
    """

    # Presidio 엔티티 -> 우리 마스킹 라벨
    _ENTITY_MAP = {
        "PERSON": "NAME",
        "LOCATION": "ADDRESS",
    }

    def __init__(self) -> None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "ko", "model_name": "ko_core_news_sm"}],
            }
        )
        self._analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=["ko"],
        )

    def detect(self, text: str) -> list[tuple[int, int, str]]:
        """(start, end, label) 리스트 반환. label은 NAME/ADDRESS."""
        results = self._analyzer.analyze(
            text=text,
            language="ko",
            entities=list(self._ENTITY_MAP.keys()),
        )
        spans: list[tuple[int, int, str]] = []
        for result in results:
            label = self._ENTITY_MAP.get(result.entity_type)
            if label:
                spans.append((result.start, result.end, label))
        return spans
