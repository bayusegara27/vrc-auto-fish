import io
import unittest
from unittest import mock

from trainer_common.console import safe_print
from utils.logger import Logger


class AsciiOnlyStream(io.StringIO):
    encoding = "ascii"

    def write(self, s):
        if any(ord(ch) > 127 for ch in s):
            raise UnicodeEncodeError("ascii", s, 0, len(s), "ordinal not in range")
        return super().write(s)


class ConsoleLoggingTests(unittest.TestCase):
    def test_safe_print_replaces_unencodable_characters(self):
        stream = AsciiOnlyStream()
        safe_print("测试 ✅", file=stream)
        self.assertEqual(stream.getvalue(), "?? ?\n")

    def test_logger_safe_print_uses_shared_console_fallback(self):
        logger = Logger()
        stream = AsciiOnlyStream()
        with mock.patch("sys.stdout", stream):
            logger.info("日志 ✅")
        self.assertIn("[ INFO]", stream.getvalue())
        self.assertNotIn("✅", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
