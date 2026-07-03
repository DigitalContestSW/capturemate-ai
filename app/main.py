from typing import Optional

from fastapi import FastAPI

from app.models import AnalyzeRequest, AnalyzeResponse

app = FastAPI(title="CaptureMate AI")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    text = request.maskedText.strip()
    category = classify(text)

    return AnalyzeResponse(
        title=create_title(text),
        summary=summarize(text),
        category=category,
        recommendedAction=recommend_action(category),
        reminderAt=None,
    )


def create_title(text: str) -> str:
    if not text:
        return "New capture"
    return text.splitlines()[0][:40]


def classify(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ["schedule", "meeting", "deadline", "reservation"]):
        return "calendar"
    if any(keyword in lowered for keyword in ["study", "exam", "lecture", "review"]):
        return "study"
    if any(keyword in lowered for keyword in ["restaurant", "cafe", "menu", "place"]):
        return "restaurant"
    if any(keyword in lowered for keyword in ["job", "resume", "interview", "recruit"]):
        return "job"
    return "memo"


def summarize(text: str) -> str:
    compact = " ".join(text.split())
    if not compact:
        return "No content to summarize."
    return compact[:120]


def recommend_action(category: str) -> Optional[str]:
    actions = {
        "calendar": "Create calendar item",
        "study": "Save as study note",
        "restaurant": "Save as place candidate",
        "job": "Check application schedule",
        "memo": "Save as memo",
    }
    return actions.get(category)
