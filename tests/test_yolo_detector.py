import unittest
from unittest.mock import patch

from core.yolo_detector import YoloDetector


class YoloDetectorTests(unittest.TestCase):
    def test_normalize_fish_class_name_supports_legacy_aliases(self):
        self.assertEqual(YoloDetector._normalize_fish_class_name("fish"), "fish_black")
        self.assertEqual(YoloDetector._normalize_fish_class_name("fish_generic"), "fish_black")
        self.assertEqual(YoloDetector._normalize_fish_class_name("fish_green"), "fish_green")
        self.assertEqual(YoloDetector._normalize_fish_class_name("fish_copper"), "fish_relic")
        self.assertEqual(YoloDetector._normalize_fish_class_name("fish_teal"), "fish_clover")
        self.assertEqual(YoloDetector._normalize_fish_class_name("fish_question"), "fish_question")

    def test_select_runtime_device_normalizes_legacy_gpu_name(self):
        self.assertEqual(
            YoloDetector.select_runtime_device("gpu", cuda_available=True),
            ("torch", 0, "cuda"),
        )
        self.assertEqual(
            YoloDetector.select_runtime_device("auto", cuda_available=False),
            ("torch", "cpu", "cpu"),
        )
        self.assertEqual(
            YoloDetector.select_runtime_device("cpu", cuda_available=True),
            ("torch", "cpu", "cpu"),
        )
        self.assertEqual(
            YoloDetector.select_runtime_device(
                "ncnn",
                cuda_available=False,
                ncnn_available=True,
            ),
            ("ncnn", "cpu", "ncnn"),
        )

    def test_select_runtime_device_rejects_forced_cuda_without_cuda(self):
        with self.assertRaises(RuntimeError):
            YoloDetector.select_runtime_device("cuda", cuda_available=False)

    def test_select_runtime_device_rejects_forced_ncnn_without_support(self):
        with self.assertRaises(RuntimeError):
            YoloDetector.select_runtime_device(
                "ncnn",
                cuda_available=False,
                ncnn_available=False,
            )

    def test_resolve_ncnn_model_path_uses_ultralytics_suffix(self):
        self.assertEqual(
            YoloDetector.resolve_ncnn_model_path(r"E:\fish\weights\best.pt"),
            r"E:\fish\weights\best_ncnn_model",
        )

    def test_can_auto_install_ncnn_is_disabled_for_frozen_builds(self):
        with patch("core.yolo_detector.sys.frozen", True, create=True):
            self.assertFalse(YoloDetector.can_auto_install_ncnn())

    def test_auto_install_ncnn_dependencies_skips_when_frozen(self):
        with patch("core.yolo_detector.sys.frozen", True, create=True):
            self.assertFalse(YoloDetector.auto_install_ncnn_dependencies())


if __name__ == "__main__":
    unittest.main()
