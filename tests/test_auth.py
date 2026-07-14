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
    auth_disabled=False,
    jwt_access_ttl_seconds=1800,
    jwt_refresh_ttl_seconds=259200,
)


class AuthApiTest(TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)

    def test_google_auth_issues_token_pair_and_refresh_issues_access_only(self) -> None:
        google_claims = {
            "sub": "google-subject-123",
            "email": "user@example.com",
            "name": "Capture Mate",
            "picture": "https://example.com/avatar.png",
        }

        with (
            patch.object(auth, "settings", TEST_SETTINGS),
            patch.object(main, "settings", TEST_SETTINGS),
            patch.object(main, "verify_google_id_token", return_value=google_claims),
        ):
            auth_response = self.client.post(
                "/v1/auth/google",
                json={"idToken": "mock-google-id-token"},
            )
            self.assertEqual(auth_response.status_code, 200)
            auth_body = auth_response.json()
            self.assertEqual(auth_body["tokenType"], "Bearer")
            self.assertTrue(auth_body["accessToken"])
            self.assertTrue(auth_body["refreshToken"])
            self.assertEqual(auth_body["accessExpiresIn"], 1800)
            self.assertEqual(auth_body["refreshExpiresIn"], 259200)

            refresh_response = self.client.post(
                "/v1/auth/refresh",
                json={"refreshToken": auth_body["refreshToken"]},
            )
            self.assertEqual(refresh_response.status_code, 200)
            refresh_body = refresh_response.json()
            self.assertTrue(refresh_body["accessToken"])
            self.assertIsNone(refresh_body["refreshToken"])
            self.assertEqual(refresh_body["accessExpiresIn"], 1800)

    def test_auth_disabled_allows_local_dev_user_without_credentials(self) -> None:
        disabled_settings = replace(TEST_SETTINGS, auth_disabled=True)

        with patch.object(main, "settings", disabled_settings):
            user = main._require_user(None)

        self.assertEqual(user.subject, "local-dev-user")

    def test_analyze_requires_access_token_and_rejects_refresh_token(self) -> None:
        google_claims = {"sub": "google-subject-123"}

        with (
            patch.object(auth, "settings", TEST_SETTINGS),
            patch.object(main, "settings", TEST_SETTINGS),
        ):
            access_token, refresh_token = auth.issue_token_pair(google_claims)

            no_token_response = self.client.post(
                "/v1/analyze",
                files={"images": ("capture.png", b"not-used", "image/png")},
            )
            self.assertEqual(no_token_response.status_code, 401)

            refresh_token_response = self.client.post(
                "/v1/analyze",
                headers={"Authorization": f"Bearer {refresh_token}"},
                files={"images": ("capture.png", b"not-used", "image/png")},
            )
            self.assertEqual(refresh_token_response.status_code, 401)

            user = auth.decode_access_token(access_token)
            self.assertEqual(user.subject, "google-subject-123")
