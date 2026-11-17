"""Unit tests for secure cookie decoding, validation, and downloader module."""

import base64
import os
import unittest
from pathlib import Path
from src.downloader import (
    decode_and_validate_cookies,
    secure_cookie_file,
    load_config,
    VideoDownloader,
)


class TestSecureCookieDownloader(unittest.TestCase):

    def setUp(self):
        # Sample valid Netscape cookie file content
        self.valid_netscape_content = (
            "# Netscape HTTP Cookie File\n"
            "# https://curl.se/docs/http-cookies.html\n"
            ".youtube.com\tTRUE\t/\tTRUE\t1772456789\tVISITOR_INFO1_LIVE\tabc123_test\n"
            ".google.com\tTRUE\t/\tTRUE\t1772456789\tSID\tsecret_sid_token\n"
        )
        self.valid_b64 = base64.b64encode(self.valid_netscape_content.encode("utf-8")).decode("utf-8")

    def test_missing_env_var_raises_value_error(self):
        """Requirement 1: Missing or empty YT_COOKIES_B64 must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            decode_and_validate_cookies(None)
        self.assertIn("missing or empty", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            decode_and_validate_cookies("   ")
        self.assertIn("missing or empty", str(ctx.exception))

    def test_valid_b64_decoding_and_netscape_validation(self):
        """Requirement 2: Decode Base64 string to valid UTF-8 Netscape text."""
        decoded = decode_and_validate_cookies(self.valid_b64)
        self.assertIn("# Netscape HTTP Cookie File", decoded)
        self.assertIn(".youtube.com", decoded)

    def test_padding_errors_handled_gracefully(self):
        """Requirement 2: Ensure Base64 padding errors (missing '=' signs) are repaired."""
        unpadded_b64 = self.valid_b64.rstrip("=")
        decoded = decode_and_validate_cookies(unpadded_b64)
        self.assertIn(".youtube.com", decoded)

    def test_bot_detection_prevention_on_invalid_or_non_youtube_cookies(self):
        """Vulnerability Prevention: Reject empty or non-YouTube domain cookies."""
        # Non-YouTube domain
        dummy_content = (
            "example.org\tTRUE\t/\tTRUE\t1772456789\tsession_id\txyz999\n"
        )
        dummy_b64 = base64.b64encode(dummy_content.encode("utf-8")).decode("utf-8")
        with self.assertRaises(ValueError) as ctx:
            decode_and_validate_cookies(dummy_b64)
        self.assertIn("Bot-detection safeguard triggered", str(ctx.exception))

        # Blank content
        empty_b64 = base64.b64encode(b"").decode("utf-8")
        with self.assertRaises(ValueError):
            decode_and_validate_cookies(empty_b64)

    def test_secure_cookie_file_lifecycle_and_deletion(self):
        """Vulnerability Prevention: Verify cookies.txt is created, then shredded and deleted in finally."""
        cookie_target = "test_temp_cookies.txt"
        os.environ["YT_COOKIES_B64"] = self.valid_b64

        try:
            # Inside context manager, file must exist and contain decoded data
            with secure_cookie_file(cookie_filename=cookie_target) as c_path:
                self.assertTrue(c_path.exists())
                content = c_path.read_text(encoding="utf-8")
                self.assertIn(".youtube.com", content)

            # Outside context manager (after finally block), file MUST NOT exist
            self.assertFalse(Path(cookie_target).exists(), "Temporary cookie file must be wiped and deleted!")
        finally:
            if "YT_COOKIES_B64" in os.environ:
                del os.environ["YT_COOKIES_B64"]
            if Path(cookie_target).exists():
                Path(cookie_target).unlink()

    def test_config_loading_and_opts_building(self):
        """Requirement 3 & 4: Config loading, best quality options, and proxy options."""
        config = load_config("config.json")
        self.assertIn("sources", config)
        self.assertIn("download_dir", config)

        downloader = VideoDownloader("config.json")
        test_path = Path("fake_cookies.txt")
        opts = downloader._build_ydl_opts(test_path, proxy="http://127.0.0.1:8080")

        self.assertEqual(opts["cookiefile"], str(test_path))
        self.assertIn("bestvideo", opts["format"])
        self.assertEqual(opts["proxy"], "http://127.0.0.1:8080")


if __name__ == "__main__":
    unittest.main()
