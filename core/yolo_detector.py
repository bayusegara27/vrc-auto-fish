"""
YOLO 目标检测器
==============
封装 ultralytics YOLO 推理，提供与模板匹配 Detector 兼容的接口。

检测类别:
  0 = fish_black / 旧 fish / fish_generic
  1-9 = fish_* 多颜色鱼
  10 = bar
  11 = track
  12 = progress
  13 = prog_hook
  14 = fish_clover
  15 = fish_question
"""

import os
import subprocess
import sys
import importlib.util
import cv2
import numpy as np
import config
from utils.logger import log

_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    pass

_NCNN_AVAILABLE = False
try:
    import ncnn  # type: ignore
    _NCNN_AVAILABLE = True
except ImportError:
    ncnn = None

_NCNN_REQUIREMENTS = ("ncnn>=1.0.20260114", "pnnx>=20260112")


def _refresh_ncnn_import():
    global _NCNN_AVAILABLE, ncnn
    try:
        import importlib
        ncnn = importlib.import_module("ncnn")  # type: ignore
        _NCNN_AVAILABLE = True
    except Exception:
        ncnn = None
        _NCNN_AVAILABLE = False
    return _NCNN_AVAILABLE


class YoloDetector:
    """YOLO-based fishing game detector."""

    CLASS_FISH = 0
    CLASS_BAR = 1
    CLASS_TRACK = 2
    CLASS_PROGRESS = 3

    @staticmethod
    def _normalize_fish_class_name(class_name: str) -> str | None:
        """兼容旧版 fish / fish_generic 与新版 fish_* 多颜色类别。"""
        if class_name == "fish":
            return "fish_black"
        class_name = config.LEGACY_FISH_KEY_ALIASES.get(class_name, class_name)
        if class_name.startswith("fish_"):
            return class_name
        return None

    @staticmethod
    def normalize_device_preference(device: str | None) -> str:
        return config.normalize_yolo_device(device)

    @staticmethod
    def select_runtime_device(
        device: str | None,
        cuda_available: bool,
        ncnn_available: bool = False,
    ):
        normalized = YoloDetector.normalize_device_preference(device)
        if normalized == "cpu":
            return "torch", "cpu", "cpu"
        if normalized == "cuda":
            if not cuda_available:
                raise RuntimeError("CUDA 不可用")
            return "torch", 0, "cuda"
        if normalized == "ncnn":
            if not ncnn_available:
                raise RuntimeError("NCNN 不可用")
            return "ncnn", "cpu", "ncnn"
        if cuda_available:
            return "torch", 0, "cuda"
        return "torch", "cpu", "cpu"

    @staticmethod
    def resolve_ncnn_model_path(model_path: str) -> str:
        return config.resolve_ncnn_model_path(model_path)

    @staticmethod
    def cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def ncnn_available() -> bool:
        return _NCNN_AVAILABLE

    @staticmethod
    def pnnx_available() -> bool:
        return importlib.util.find_spec("pnnx") is not None

    @staticmethod
    def can_auto_install_ncnn() -> bool:
        return not getattr(sys, "frozen", False)

    @staticmethod
    def auto_install_ncnn_dependencies() -> bool:
        if not YoloDetector.can_auto_install_ncnn():
            return False
        if YoloDetector.ncnn_available() and YoloDetector.pnnx_available():
            return True
        log.info_t("yolo.log.ncnnAutoInstallStart")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", *_NCNN_REQUIREMENTS],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as e:
            log.warning_t("yolo.log.ncnnAutoInstallFailed", error=e)
            return False
        ok = _refresh_ncnn_import()
        if ok and YoloDetector.pnnx_available():
            log.info_t("yolo.log.ncnnAutoInstallDone")
            return True
        log.warning_t("yolo.log.ncnnAutoInstallIncomplete")
        return False

    @staticmethod
    def select_ncnn_runtime_device():
        if not _NCNN_AVAILABLE:
            return "cpu", "ncnn"
        try:
            gpu_count = int(ncnn.get_gpu_count())
        except Exception:
            gpu_count = 0
        if gpu_count > 0:
            return "vulkan:0", "ncnn(vulkan:0)"
        return "cpu", "ncnn(cpu)"

    @staticmethod
    def ensure_ncnn_model(model_path: str) -> str:
        ncnn_model_path = YoloDetector.resolve_ncnn_model_path(model_path)
        if os.path.isdir(ncnn_model_path):
            return ncnn_model_path

        if not _YOLO_AVAILABLE:
            raise ImportError("ultralytics 未安装，无法导出 NCNN 模型")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO 模型未找到: {model_path}")

        log.info_t("yolo.log.ncnnExportStart", model_path=model_path)
        try:
            exported = YOLO(model_path).export(format="ncnn", imgsz=640)
        except Exception as e:
            log.warning_t("yolo.log.ncnnExportFailed", error=e)
            raise RuntimeError(f"NCNN 导出失败: {e}") from e

        exported_path = str(exported) if exported is not None else ncnn_model_path
        if os.path.isdir(exported_path):
            ncnn_model_path = exported_path
        if not os.path.isdir(ncnn_model_path):
            raise FileNotFoundError(f"NCNN 模型目录未生成: {ncnn_model_path}")
        log.info_t("yolo.log.ncnnExportDone", model_path=ncnn_model_path)
        return ncnn_model_path

    @staticmethod
    def build_runtime(model_path: str, device="auto"):
        if not _YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics 未安装。请运行: pip install ultralytics"
            )
        normalized_device = YoloDetector.normalize_device_preference(device)
        ncnn_model_path = YoloDetector.resolve_ncnn_model_path(model_path)
        if normalized_device == "ncnn" and (
            not YoloDetector.ncnn_available() or not YoloDetector.pnnx_available()
        ):
            YoloDetector.auto_install_ncnn_dependencies()
        if not os.path.exists(model_path) and not (
            normalized_device == "ncnn" and os.path.isdir(ncnn_model_path)
        ):
            raise FileNotFoundError(f"YOLO 模型未找到: {model_path}")

        dev_pref = normalized_device
        cuda_ok = YoloDetector.cuda_available()

        backend, target_dev, device_label = YoloDetector.select_runtime_device(
            dev_pref,
            cuda_ok,
            ncnn_available=_NCNN_AVAILABLE,
        )
        warmup_img = np.zeros((640, 640, 3), dtype=np.uint8)

        if backend == "ncnn":
            if not _NCNN_AVAILABLE:
                raise RuntimeError(
                    "NCNN 依赖未安装。请安装 requirements.txt 中的 ncnn / pnnx。"
                )
            ncnn_model_path = YoloDetector.ensure_ncnn_model(model_path)
            model = YOLO(ncnn_model_path, task="detect")
            runtime_device, runtime_label = YoloDetector.select_ncnn_runtime_device()
            try:
                model.predict(
                    warmup_img, conf=0.5, device=runtime_device,
                    verbose=False, imgsz=640,
                )
            except Exception as e:
                if runtime_device != "cpu":
                    log.warning_t("yolo.log.ncnnVulkanFallback", error=e)
                    runtime_device = "cpu"
                    runtime_label = "ncnn(cpu)"
                    model.predict(
                        warmup_img, conf=0.5, device=runtime_device,
                        verbose=False, imgsz=640,
                    )
                else:
                    raise RuntimeError(f"[YOLO] NCNN 初始化失败: {e}") from e
            log.info_t("yolo.log.ncnnReady", device=runtime_label, names=model.names)
            return {
                "model": model,
                "runtime_device": runtime_device,
                "device_label": runtime_label,
                "backend_label": "ncnn",
                "model_path": ncnn_model_path,
            }

        model = YOLO(model_path)
        if target_dev != "cpu":
            try:
                model.predict(
                    warmup_img, conf=0.5, device=target_dev,
                    verbose=False, imgsz=640,
                )
                for _ in range(2):
                    model.predict(
                        warmup_img, conf=0.5, device=target_dev,
                        verbose=False, imgsz=640,
                    )
                return {
                    "model": model,
                    "runtime_device": target_dev,
                    "device_label": device_label,
                    "backend_label": "torch",
                    "model_path": model_path,
                }
            except Exception as e:
                if dev_pref == "cuda":
                    raise RuntimeError(f"[YOLO] 强制 CUDA 模式但初始化失败: {e}") from e
                log.warning_t("yolo.log.gpuFallback", error=e)

        model.predict(
            warmup_img, conf=0.5, device="cpu",
            verbose=False, imgsz=640,
        )
        log.info_t("yolo.log.cpuReady", names=model.names)
        return {
            "model": model,
            "runtime_device": "cpu",
            "device_label": "cpu",
            "backend_label": "torch",
            "model_path": model_path,
        }

    def __init__(self, model_path: str, conf: float = 0.5, device="auto"):
        self.conf = conf
        runtime = self.build_runtime(model_path, device=device)
        self.model = runtime["model"]
        self._device = runtime["runtime_device"]
        self._device_label = runtime["device_label"]
        self._backend_label = runtime["backend_label"]
        self._model_path = runtime["model_path"]

    def detect(self, screen, roi=None):
        """
        对一帧画面执行 YOLO 推理。

        参数:
            screen: BGR 图像 (numpy array)
            roi:    [x, y, w, h] 检测区域 (可选)

        返回:
            dict: {
                'fish':  (x, y, w, h, conf) 或 None,
                'bar':   (x, y, w, h, conf) 或 None,
                'track': (x, y, w, h, conf) 或 None,
                'fish_name': str,  # 鱼的类别名称
                'raw': list,       # 所有检测结果
            }
        """
        ox, oy = 0, 0
        img = screen

        if roi:
            rx, ry, rw, rh = roi
            h_s, w_s = screen.shape[:2]
            rx = max(0, min(rx, w_s))
            ry = max(0, min(ry, h_s))
            rw = min(rw, w_s - rx)
            rh = min(rh, h_s - ry)
            if rw > 10 and rh > 10:
                img = screen[ry:ry+rh, rx:rx+rw].copy()
                ox, oy = rx, ry

        results = self.model.predict(
            img, conf=self.conf, device=self._device,
            verbose=False, imgsz=640,
        )

        detections = {
            "fish": None,
            "bar": None,
            "track": None,
            "progress": None,
            "prog_hook": None,
            "fish_name": "",
            "raw": [],
        }

        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for i in range(len(boxes)):
            cls = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            bx = int(x1) + ox
            by = int(y1) + oy
            bw = int(x2 - x1)
            bh = int(y2 - y1)

            det = (bx, by, bw, bh, conf)
            class_name = self.model.names.get(cls, f"cls{cls}")
            detections["raw"].append((class_name, det))

            fish_name = self._normalize_fish_class_name(class_name)
            if fish_name:
                if detections["fish"] is None or conf > detections["fish"][4]:
                    detections["fish"] = det
                    detections["fish_name"] = fish_name
            elif class_name == "bar":
                if detections["bar"] is None or conf > detections["bar"][4]:
                    detections["bar"] = det
            elif class_name == "track":
                if detections["track"] is None or conf > detections["track"][4]:
                    detections["track"] = det
            elif class_name == "progress":
                if detections["progress"] is None or conf > detections["progress"][4]:
                    detections["progress"] = det
            elif class_name == "prog_hook":
                if detections["prog_hook"] is None or conf > detections["prog_hook"][4]:
                    detections["prog_hook"] = det

        return detections

    def detect_track(self, screen, roi=None):
        """仅检测轨道是否存在"""
        result = self.detect(screen, roi)
        return result["track"]

    def detect_bar(self, screen, roi=None):
        """仅检测白条"""
        result = self.detect(screen, roi)
        return result["bar"]

    def detect_fish(self, screen, roi=None):
        """仅检测鱼"""
        result = self.detect(screen, roi)
        return result["fish"], result["fish_name"]
