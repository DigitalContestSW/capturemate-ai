from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.auth as auth
import app.main as main
from app.config import settings


TEST_SETTINGS = replace(
    settings,
    google_web_client_id="google-web-client-id.apps.googleusercontent.com",
    jwt_access_secret="test-access-secret-with-enough-length",
    jwt_refresh_secret="test-refresh-secret-with-enough-length",
    jwt_access_ttl_seconds=1800,
    jwt_refresh_ttl_seconds=259200,
)


class FakeOcrEngine:
    def __init__(self, text: str = "충분한 OCR 테스트 텍스트입니다") -> None:
        self.text = text
        self.calls: list[bytes] = []

    def extract_text(self, image_bytes: bytes) -> str:
        self.calls.append(image_bytes)
        return self.text


class FakeBatchAnalyzer:
    def __init__(self) -> None:
        self.items = None
        self.locale = None

    def analyze_batch(self, items, locale):
        self.items = items
        self.locale = locale
        return []


class HealthAndOcrTest(TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)

    def test_readiness_reports_503_until_ocr_is_loaded(self) -> None:
        with (
            patch.object(main, "_ocr_engine", None),
            patch.object(main, "_ocr_load_error", None),
        ):
            live_response = self.client.get("/health/live")
            ready_response = self.client.get("/health/ready")

        self.assertEqual(live_response.status_code, 200)
        self.assertEqual(live_response.json(), {"status": "ok"})
        self.assertEqual(ready_response.status_code, 503)
        self.assertFalse(ready_response.json()["ocrReady"])

    def test_readiness_reports_200_after_ocr_is_loaded(self) -> None:
        with (
            patch.object(main, "_ocr_engine", FakeOcrEngine()),
            patch.object(main, "_ocr_load_error", None),
        ):
            response = self.client.get("/health/ready")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ocrReady"])

    def test_analyze_uses_loaded_ocr_engine_and_batch_analyzer(self) -> None:
        fake_engine = FakeOcrEngine()
        fake_batch_analyzer = FakeBatchAnalyzer()

        with (
            patch.object(auth, "settings", TEST_SETTINGS),
            patch.object(main, "_ocr_engine", fake_engine),
            patch.object(main, "_ocr_load_error", None),
            patch.object(main, "batch_analyzer", fake_batch_analyzer),
        ):
            access_token, _ = auth.issue_token_pair({"sub": "google-subject-123"})
            response = self.client.post(
                "/v1/analyze",
                headers={"Authorization": f"Bearer {access_token}"},
                data={
                    "locale": "ko-KR",
                    "metadata": '[{"clientId":"capture-1","capturedAt":1760000000000}]',
                },
                files={"images": ("capture.png", b"image-bytes", "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"groups": []})
        self.assertEqual(fake_engine.calls, [b"image-bytes"])
        self.assertEqual(fake_batch_analyzer.locale, "ko-KR")
        self.assertEqual(fake_batch_analyzer.items[0].clientId, "capture-1")
