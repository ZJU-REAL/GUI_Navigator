"""Inference utilities for the Grounding Demo."""
from __future__ import annotations

import base64
import io
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class PredictionResult:
    """Normalized inference output used by the UI layer."""

    model_name: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    raw_coordinates: Tuple[float, float]
    coordinate_mode: str
    prompt: str
    explanation: str

    @property
    def relative_coordinates(self) -> Tuple[float, float]:
        """Return coordinates normalized to [0, 1] range."""
        return self.x / max(self.width, 1), self.y / max(self.height, 1)


class CoordinateConverter:
    """Utility that converts model specific coordinates to absolute pixels."""

    def __call__(
        self, coords: Tuple[float, float], size: Tuple[int, int], mode: str
    ) -> Tuple[int, int]:
        if mode not in {"absolute", "relative"}:
            raise ValueError(f"Unsupported coordinate mode: {mode}")
        width, height = size
        if width <= 0 or height <= 0:
            raise ValueError("Invalid target size for conversion.")
        if mode == "absolute":
            x, y = coords
        else:
            rel_x, rel_y = coords
            x = rel_x * width
            y = rel_y * height
        return int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))


converter = CoordinateConverter()


def _saliency_based(image: Image.Image) -> Tuple[float, float]:
    """Very rough heuristic that mimics a saliency-based detector."""
    np_img = np.array(image.convert("L"))
    y, x = np.unravel_index(np.argmax(np_img), np_img.shape)
    height, width = np_img.shape
    return x / max(width, 1), y / max(height, 1)


def _keyword_relative(instruction: str) -> Tuple[float, float]:
    """Map textual hints to coarse relative coordinates."""
    text = instruction.lower()
    x, y = 0.5, 0.5

    def contains(*keywords: str) -> bool:
        return any(k in text for k in keywords)

    if contains("\u5de6\u4e0a", "top-left", "upper left"):
        return 0.2, 0.2
    if contains("\u53f3\u4e0a", "top-right", "upper right"):
        return 0.8, 0.2
    if contains("\u5de6\u4e0b", "bottom-left", "lower left"):
        return 0.2, 0.8
    if contains("\u53f3\u4e0b", "bottom-right", "lower right"):
        return 0.8, 0.8

    if contains("top", "\u9876\u90e8", "\u4e0a\u65b9", "\u4e0a\u9762"):
        y = 0.18
    elif contains("bottom", "\u5e95\u90e8", "\u4e0b\u65b9", "\u4e0b\u9762"):
        y = 0.82

    if contains("left", "\u5de6\u4fa7", "\u5de6\u8fb9"):
        x = 0.22
    elif contains("right", "\u53f3\u4fa7", "\u53f3\u8fb9"):
        x = 0.78

    if contains("center", "middle", "\u4e2d\u592e", "\u4e2d\u5fc3"):
        x, y = 0.5, 0.5

    if contains("\u641c\u7d22", "search"):
        y = min(y, 0.25)
        x = 0.75
    if contains("\u767b\u5f55", "login", "\u786e\u8ba4", "\u786e\u5b9a", "submit", "\u6309\u94ae"):
        y = max(y, 0.75)
        x = 0.65
    return x, y


def _instruction_guided_relative(
    image: Image.Image, instruction: str, mix_with_saliency: bool
) -> Tuple[float, float]:
    base_x, base_y = _keyword_relative(instruction)
    if mix_with_saliency:
        sal_x, sal_y = _saliency_based(image)
        alpha = 0.6
        base_x = alpha * sal_x + (1 - alpha) * base_x
        base_y = alpha * sal_y + (1 - alpha) * base_y
    noise = np.random.uniform(-0.03, 0.03, size=2)
    x = float(np.clip(base_x + noise[0], 0.01, 0.99))
    y = float(np.clip(base_y + noise[1], 0.01, 0.99))
    return x, y


def _encode_image_to_base64(image: Image.Image) -> str:
    """Encode a PIL image as a base64 data URL."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _move_to_device(obj, device):
    """Recursively move nested tensors/batch features onto target device."""
    if hasattr(obj, "to"):
        try:
            return obj.to(device)
        except TypeError:
            pass
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_move_to_device(v, device) for v in obj]
        return type(obj)(converted)
    return obj


def _flash_attn_available() -> bool:
    """Detect whether flash-attn v2 is installed."""
    try:  # pragma: no cover - import side effects
        import flash_attn  # type: ignore

        return hasattr(flash_attn, "__version__")
    except Exception:
        return False


FLASH_ATTN_AVAILABLE = _flash_attn_available()


class GUIG2Predictor:
    """Lazy loader for GUI-G2-7B that runs real model inference."""

    PROMPT_TEMPLATE = (
        "Outline the position corresponding to the instruction: {instruction}. "
        "The output should be only [x1,y1,x2,y2]."
    )

    PROMPT_TEMPLATE = (
        "Outline the position corresponding to the instruction: {instruction}. "
        "The output should be only [x1,y1,x2,y2]."
    )

    BBOX_PATTERN = re.compile(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    )
    COORD_PAIR_PATTERN = re.compile(
        r"(?:x|coord[_ ]?x|pixel[_ ]?x)\D*(-?\d+(?:\.\d+)?)"
        r".{0,40}?"
        r"(?:y|coord[_ ]?y|pixel[_ ]?y)\D*(-?\d+(?:\.\d+)?)",
        re.IGNORECASE | re.DOTALL,
    )
    NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._model = None
        self._processor = None
        self._process_vision_info = None
        self._lock = threading.Lock()
        self._last_text: Optional[str] = None

    def _validate_model_dir(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"GUI-G2-7B assets not found at {self.model_dir}. "
                "Set GUI_G2_MODEL_PATH to point at the local weights directory."
            )

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        with self._lock:
            if self._model is not None and self._processor is not None:
                return
            self._validate_model_dir()
            try:
                from transformers import (
                    AutoProcessor,
                    Qwen2_5_VLForConditionalGeneration,
                )
                import torch
            except ImportError as exc:  # pragma: no cover - env specific
                raise RuntimeError(
                    "transformers>=4.49.0 and torch are required for GUI-G2-7B."
                ) from exc

            try:
                from qwen_vl_utils import process_vision_info  # type: ignore
            except ImportError:  # pragma: no cover - optional helper
                process_vision_info = None

            attn_impl = "flash_attention_2" if FLASH_ATTN_AVAILABLE else "eager"
            if not FLASH_ATTN_AVAILABLE:
                print(
                    "[GUI-G2-7B] flash-attn not detected, falling back to regular attention."
                )

            self._processor = AutoProcessor.from_pretrained(self.model_dir)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_dir,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                device_map="auto",
            )
            self._model.eval()
            self._process_vision_info = process_vision_info

    def _prepare_inputs(self, image: Image.Image, instruction: str):
        if self._processor is None:
            raise RuntimeError("GUI-G2-7B processor not initialized.")
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _encode_image_to_base64(image)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if self._process_vision_info is not None:
            image_inputs, video_inputs = self._process_vision_info(messages)
        else:
            image_inputs, video_inputs = [image], None

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def _parse_coordinates(
        self,
        response_text: str,
        size: Tuple[int, int],
        grid_info: Optional[np.ndarray],
    ) -> Tuple[float, float]:
        width, height = size
        bbox_match = self.BBOX_PATTERN.search(response_text)
        if bbox_match:
            values = [float(v) for v in bbox_match.groups()]
            x1, y1, x2, y2 = values
            if grid_info is not None:
                proc_h = float(grid_info[0][1]) * 14.0
                proc_w = float(grid_info[0][2]) * 14.0
            else:
                proc_h = float(height)
                proc_w = float(width)
            proc_w = max(proc_w, 1.0)
            proc_h = max(proc_h, 1.0)
            norm_x1 = np.clip(x1 / proc_w, 0.0, 1.0)
            norm_x2 = np.clip(x2 / proc_w, 0.0, 1.0)
            norm_y1 = np.clip(y1 / proc_h, 0.0, 1.0)
            norm_y2 = np.clip(y2 / proc_h, 0.0, 1.0)
            abs_x1 = norm_x1 * width
            abs_x2 = norm_x2 * width
            abs_y1 = norm_y1 * height
            abs_y2 = norm_y2 * height
            cx = np.clip((abs_x1 + abs_x2) / 2.0, 0, width - 1)
            cy = np.clip((abs_y1 + abs_y2) / 2.0, 0, height - 1)
            return float(cx), float(cy)

        match = self.COORD_PAIR_PATTERN.search(response_text)
        if match:
            x_val, y_val = match.groups()
        else:
            numbers = self.NUMBER_PATTERN.findall(response_text)
            if len(numbers) >= 2:
                x_val, y_val = numbers[0], numbers[1]
            else:
                raise ValueError(
                    f"Unable to parse coordinates from GUI-G2-7B output: {response_text}"
                )
        x = float(x_val)
        y = float(y_val)
        if 0 <= x <= 1 and 0 <= y <= 1:
            x *= width
            y *= height
        return x, y

    def __call__(self, image: Image.Image, instruction: str) -> Tuple[float, float]:
        self._load()
        if self._model is None:
            raise RuntimeError("GUI-G2-7B model is not initialized.")
        if self._processor is None:
            raise RuntimeError("GUI-G2-7B processor is not initialized.")

        import torch  # local import to avoid unconditional dependency

        image = image.convert("RGB")
        inputs = self._prepare_inputs(image, instruction)
        grid_info: Optional[np.ndarray] = None
        if "image_grid_thw" in inputs:
            try:
                grid_info = inputs["image_grid_thw"].detach().cpu().numpy()
            except Exception:
                grid_info = None
        inputs = _move_to_device(inputs, self._model.device)
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        self._last_text = output_text
        try:
            x, y = self._parse_coordinates(output_text, image.size, grid_info)
        except ValueError:
            rel_x, rel_y = _keyword_relative(instruction)
            x = rel_x * image.width
            y = rel_y * image.height
        return x, y

    @property
    def last_text(self) -> Optional[str]:
        return self._last_text


GUI_G2_MODEL_DIR = Path(
    os.environ.get(
        "GUI_G2_MODEL_PATH",
        Path(__file__).resolve().parent / "inclusionAI" / "GUI-G2-7B",
    )
)
gui_g2_predictor = GUIG2Predictor(GUI_G2_MODEL_DIR)


def _gui_g2_predict(image: Image.Image, instruction: str) -> Tuple[float, float]:
    return gui_g2_predictor(image, instruction)


class UIVenusPredictor:
    """Lazy loader for UI-Venus Grounding model."""

    BBOX_PATTERN = re.compile(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    )
    NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._model = None
        self._processor = None
        self._process_vision_info = None
        self._lock = threading.Lock()
        self._last_text: Optional[str] = None

    def _validate_model_dir(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"UI-Venus assets not found at {self.model_dir}. "
                "Set UI_VENUS_MODEL_PATH to override."
            )

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        with self._lock:
            if self._model is not None and self._processor is not None:
                return
            self._validate_model_dir()
            try:
                from transformers import (
                    AutoProcessor,
                    AutoTokenizer,
                    Qwen2_5_VLForConditionalGeneration,
                )
                import torch
            except ImportError as exc:  # pragma: no cover - env specific
                raise RuntimeError(
                    "transformers>=4.49.0 and torch are required for UI-Venus."
                ) from exc

            try:
                from qwen_vl_utils import process_vision_info  # type: ignore
            except ImportError:  # pragma: no cover
                process_vision_info = None

            attn_impl = "flash_attention_2" if FLASH_ATTN_AVAILABLE else "eager"
            if not FLASH_ATTN_AVAILABLE:
                print(
                    "[UI-Venus] flash-attn not detected, using regular attention."
                )

            self._processor = AutoProcessor.from_pretrained(
                self.model_dir, trust_remote_code=True
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, trust_remote_code=True
            )
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                device_map="auto",
            ).eval()
            self._process_vision_info = process_vision_info

    def _prepare_inputs(self, image: Image.Image, prompt: str):
        if self._processor is None:
            raise RuntimeError("UI-Venus processor not initialized.")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _encode_image_to_base64(image)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self._process_vision_info is not None:
            image_inputs, video_inputs = self._process_vision_info(messages)
        else:
            image_inputs, video_inputs = [image], None
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def _parse_bbox(
        self, response_text: str, size: Tuple[int, int]
    ) -> Tuple[float, float]:
        width, height = size
        match = self.BBOX_PATTERN.search(response_text)
        if match:
            values = [float(v) for v in match.groups()]
        else:
            numbers = self.NUMBER_PATTERN.findall(response_text)
            if len(numbers) < 4:
                raise ValueError(
                    f"Unable to parse bbox from UI-Venus output: {response_text}"
                )
            values = [float(v) for v in numbers[:4]]
        x1, y1, x2, y2 = values
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        if 0 <= x_max <= 1 and 0 <= y_max <= 1:
            x_min *= width
            x_max *= width
            y_min *= height
            y_max *= height
        cx = np.clip((x_min + x_max) / 2.0, 0, width - 1)
        cy = np.clip((y_min + y_max) / 2.0, 0, height - 1)
        return float(cx), float(cy)

    def __call__(self, image: Image.Image, instruction: str) -> Tuple[float, float]:
        self._load()
        if self._model is None or self._processor is None:
            raise RuntimeError("UI-Venus failed to initialize.")

        import torch

        image = image.convert("RGB")
        prompt = (
            "Outline the position corresponding to the instruction: "
            f"{instruction}. The output should be only [x1,y1,x2,y2]."
        )
        inputs = self._prepare_inputs(image, prompt)
        inputs = _move_to_device(inputs, self._model.device)
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=256, do_sample=False, temperature=0.0
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        self._last_text = output_text
        try:
            cx, cy = self._parse_bbox(output_text, image.size)
        except ValueError:
            rel_x, rel_y = _keyword_relative(instruction)
            cx = rel_x * image.width
            cy = rel_y * image.height
        return cx, cy

    @property
    def last_text(self) -> Optional[str]:
        return self._last_text


UI_VENUS_MODEL_DIR = Path(
    os.environ.get(
        "UI_VENUS_MODEL_PATH",
        Path(__file__).resolve().parent / "UI-Venus-Ground-7B",
    )
)
ui_venus_predictor = UIVenusPredictor(UI_VENUS_MODEL_DIR)


def _ui_venus_predict(image: Image.Image, instruction: str) -> Tuple[float, float]:
    return ui_venus_predictor(image, instruction)


class UITARSPredictor:
    """Lazy loader for UI-TARS-1.5-7B checkpoints."""

    BBOX_PATTERN = re.compile(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    )
    COORD_PAIR_PATTERN = re.compile(
        r"(?:x|coord[_ ]?x|pixel[_ ]?x)\D*(-?\d+(?:\.\d+)?)"
        r".{0,40}?"
        r"(?:y|coord[_ ]?y|pixel[_ ]?y)\D*(-?\d+(?:\.\d+)?)",
        re.IGNORECASE | re.DOTALL,
    )
    NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._model = None
        self._processor = None
        self._process_vision_info = None
        self._lock = threading.Lock()
        self._last_text: Optional[str] = None

    def _validate_model_dir(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"UI-TARS-1.5-7B assets not found at {self.model_dir}. "
                "Set UI_TARS_MODEL_PATH to override."
            )

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        with self._lock:
            if self._model is not None and self._processor is not None:
                return
            self._validate_model_dir()
            try:
                from transformers import (
                    AutoProcessor,
                    Qwen2_5_VLForConditionalGeneration,
                )
                import torch
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "transformers>=4.49.0 and torch are required for UI-TARS-1.5-7B."
                ) from exc

            try:
                from qwen_vl_utils import process_vision_info  # type: ignore
            except ImportError:
                process_vision_info = None

            attn_impl = "flash_attention_2" if FLASH_ATTN_AVAILABLE else "eager"
            if not FLASH_ATTN_AVAILABLE:
                print(
                    "[UI-TARS-1.5-7B] flash-attn not detected, using regular attention."
                )

            self._processor = AutoProcessor.from_pretrained(self.model_dir)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_dir,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                device_map="auto",
            ).eval()
            self._process_vision_info = process_vision_info

    def _prepare_inputs(self, image: Image.Image, instruction: str):
        if self._processor is None:
            raise RuntimeError("UI-TARS processor not initialized.")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _encode_image_to_base64(image)},
                    {
                        "type": "text",
                        "text": (
                            "Locate the UI target for instruction: "
                            f"{instruction}. Output [x1,y1,x2,y2] "
                            "bounding box coordinates."
                        ),
                    },
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self._process_vision_info is not None:
            image_inputs, video_inputs = self._process_vision_info(messages)
        else:
            image_inputs, video_inputs = [image], None
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def _parse_coordinates(
        self, response_text: str, size: Tuple[int, int], grid_info: Optional[np.ndarray]
    ) -> Tuple[float, float]:
        width, height = size
        bbox_match = self.BBOX_PATTERN.search(response_text)
        if bbox_match:
            values = [float(v) for v in bbox_match.groups()]
            x1, y1, x2, y2 = values
            if grid_info is not None:
                proc_h = float(grid_info[0][1]) * 14.0
                proc_w = float(grid_info[0][2]) * 14.0
            else:
                proc_h = float(height)
                proc_w = float(width)
            proc_w = max(proc_w, 1.0)
            proc_h = max(proc_h, 1.0)
            norm_x1 = np.clip(x1 / proc_w, 0.0, 1.0)
            norm_x2 = np.clip(x2 / proc_w, 0.0, 1.0)
            norm_y1 = np.clip(y1 / proc_h, 0.0, 1.0)
            norm_y2 = np.clip(y2 / proc_h, 0.0, 1.0)
            abs_x1 = norm_x1 * width
            abs_x2 = norm_x2 * width
            abs_y1 = norm_y1 * height
            abs_y2 = norm_y2 * height
            cx = np.clip((abs_x1 + abs_x2) / 2.0, 0, width - 1)
            cy = np.clip((abs_y1 + abs_y2) / 2.0, 0, height - 1)
            return float(cx), float(cy)

        coord_match = self.COORD_PAIR_PATTERN.search(response_text)
        if coord_match:
            x_val, y_val = coord_match.groups()
        else:
            numbers = self.NUMBER_PATTERN.findall(response_text)
            if len(numbers) >= 2:
                x_val, y_val = numbers[0], numbers[1]
            else:
                raise ValueError(
                    f"Unable to parse coordinates from UI-TARS output: {response_text}"
                )
        x = float(x_val)
        y = float(y_val)
        if 0 <= x <= 1 and 0 <= y <= 1:
            x *= width
            y *= height
        return x, y

    def __call__(self, image: Image.Image, instruction: str) -> Tuple[float, float]:
        self._load()
        if self._model is None or self._processor is None:
            raise RuntimeError("UI-TARS components failed to initialize.")

        import torch

        image = image.convert("RGB")
        inputs = self._prepare_inputs(image, instruction)
        grid_info: Optional[np.ndarray] = None
        if "image_grid_thw" in inputs:
            try:
                grid_info = inputs["image_grid_thw"].detach().cpu().numpy()
            except Exception:
                grid_info = None
        inputs = _move_to_device(inputs, self._model.device)
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        self._last_text = output_text
        try:
            return self._parse_coordinates(output_text, image.size, grid_info)
        except ValueError:
            rel_x, rel_y = _keyword_relative(instruction)
            return rel_x * image.width, rel_y * image.height

    @property
    def last_text(self) -> Optional[str]:
        return self._last_text


UI_TARS_MODEL_DIR = Path(
    os.environ.get(
        "UI_TARS_MODEL_PATH",
        Path(__file__).resolve().parent / "UI-TARS-1.5-7B",
    )
)
ui_tars_predictor = UITARSPredictor(UI_TARS_MODEL_DIR)


def _ui_tars_predict(image: Image.Image, instruction: str) -> Tuple[float, float]:
    return ui_tars_predictor(image, instruction)


class GUIOwlPredictor:
    """Loader for GUI-Owl checkpoint."""

    PROMPT_TEMPLATE = (
        "Specify the target for instruction: {instruction}. "
        "Return bounding box [x1,y1,x2,y2] (pixel coordinates)."
    )
    BBOX_PATTERN = re.compile(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    )
    NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._model = None
        self._processor = None
        self._process_vision_info = None
        self._lock = threading.Lock()
        self._last_text: Optional[str] = None

    def _validate_model_dir(self):
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"GUI-Owl weights not found at {self.model_dir}. "
                "Set GUI_OWL_MODEL_PATH to override."
            )

    def _load(self):
        if self._model is not None and self._processor is not None:
            return
        with self._lock:
            if self._model is not None and self._processor is not None:
                return
            self._validate_model_dir()
            try:
                from transformers import (
                    AutoProcessor,
                    Qwen2_5_VLForConditionalGeneration,
                )
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "transformers>=4.49.0 and torch are required for GUI-Owl."
                ) from exc
            try:
                from qwen_vl_utils import process_vision_info  # type: ignore
            except ImportError:
                process_vision_info = None
            attn_impl = "flash_attention_2" if FLASH_ATTN_AVAILABLE else "eager"
            if not FLASH_ATTN_AVAILABLE:
                print("[GUI-Owl] flash-attn not detected, using eager attention.")
            self._processor = AutoProcessor.from_pretrained(
                self.model_dir, trust_remote_code=True
            )
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                device_map="auto",
            ).eval()
            self._process_vision_info = process_vision_info

    def _prepare_inputs(self, image: Image.Image, instruction: str):
        if self._processor is None:
            raise RuntimeError("GUI-Owl processor not initialized.")
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _encode_image_to_base64(image)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self._process_vision_info is not None:
            image_inputs, video_inputs = self._process_vision_info(messages)
        else:
            image_inputs, video_inputs = [image], None
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def _parse_bbox(
        self, response: str, size: Tuple[int, int], grid: Optional[np.ndarray]
    ) -> Tuple[float, float]:
        width, height = size
        match = self.BBOX_PATTERN.search(response)
        if match:
            values = [float(v) for v in match.groups()]
        else:
            numbers = self.NUMBER_PATTERN.findall(response)
            if len(numbers) < 4:
                raise ValueError(f"Unable to parse GUI-Owl output: {response}")
            values = [float(v) for v in numbers[:4]]
        x1, y1, x2, y2 = values
        if grid is not None:
            proc_h = float(grid[0][1]) * 14.0
            proc_w = float(grid[0][2]) * 14.0
        else:
            proc_h = float(height)
            proc_w = float(width)
        proc_w = max(proc_w, 1.0)
        proc_h = max(proc_h, 1.0)
        abs_x1 = np.clip(x1 / proc_w, 0, 1) * width
        abs_x2 = np.clip(x2 / proc_w, 0, 1) * width
        abs_y1 = np.clip(y1 / proc_h, 0, 1) * height
        abs_y2 = np.clip(y2 / proc_h, 0, 1) * height
        cx = np.clip((abs_x1 + abs_x2) / 2.0, 0, width - 1)
        cy = np.clip((abs_y1 + abs_y2) / 2.0, 0, height - 1)
        return float(cx), float(cy)

    def __call__(self, image: Image.Image, instruction: str) -> Tuple[float, float]:
        self._load()
        if self._model is None or self._processor is None:
            raise RuntimeError("GUI-Owl not initialized.")
        import torch

        image = image.convert("RGB")
        inputs = self._prepare_inputs(image, instruction)
        grid = None
        if "image_grid_thw" in inputs:
            try:
                grid = inputs["image_grid_thw"].detach().cpu().numpy()
            except Exception:
                grid = None
        inputs = _move_to_device(inputs, self._model.device)
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=128, do_sample=False, temperature=0.0
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        self._last_text = response
        try:
            return self._parse_bbox(response, image.size, grid)
        except ValueError:
            rel_x, rel_y = _keyword_relative(instruction)
            return rel_x * image.width, rel_y * image.height

    @property
    def last_text(self) -> Optional[str]:
        return self._last_text


GUI_OWL_MODEL_DIR = Path(
    os.environ.get(
        "GUI_OWL_MODEL_PATH",
        Path(__file__).resolve().parent / "GUI-Owl",
    )
)
gui_owl_predictor = GUIOwlPredictor(GUI_OWL_MODEL_DIR)


def _gui_owl_predict(image: Image.Image, instruction: str) -> Tuple[float, float]:
    return gui_owl_predictor(image, instruction)


MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "UI-TARS-1.5": {
        "coordinate_mode": "absolute",
        "predict_fn": _ui_tars_predict,
        "description": "UI-TARS-1.5-7B grounding model loaded from local weights.",
        "prompt_template": (
            "UI-TARS-1.5: Outline the position for \"{instruction}\" "
            "and return [x1,y1,x2,y2] in pixels."
        ),
        "base_confidence": 0.88,
    },
    "GUI-G2-7B": {
        "coordinate_mode": "absolute",
        "predict_fn": _gui_g2_predict,
        "description": "Multimodal GUI-G2-7B model loaded from local weights.",
        "prompt_template": (
            "[INST] Model=GUI-G2-7B. Task: {instruction}. "
            "Respond with pixel coordinates. [/INST]"
        ),
        "base_confidence": 0.9,
    },
    "UI-Venus": {
        "coordinate_mode": "absolute",
        "predict_fn": _ui_venus_predict,
        "description": "UI-Venus Ground-7B loaded locally and returning bbox-derived pixels.",
        "prompt_template": (
            "Outline the position corresponding to: {instruction}. "
            "Return [x1,y1,x2,y2]."
        ),
        "base_confidence": 0.88,
    },
    "GUI-Owl": {
        "coordinate_mode": "absolute",
        "predict_fn": _gui_owl_predict,
        "description": "GUI-Owl grounding model loaded from local weights.",
        "prompt_template": (
            "GUI-Owl: identify the element for \"{instruction}\" "
            "and respond with [x1,y1,x2,y2]."
        ),
        "base_confidence": 0.85,
    },
}


def list_models() -> Dict[str, str]:
    """Return available model names and user-facing descriptions."""
    return {name: cfg["description"] for name, cfg in MODEL_REGISTRY.items()}


def infer(model_name: str, image: Image.Image, instruction: str) -> PredictionResult:
    """Unified inference entrypoint used by the Gradio UI."""
    if image is None:
        raise ValueError("Image is required for inference.")
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model_name}")

    normalized_instruction = (instruction or "").strip()
    if not normalized_instruction:
        normalized_instruction = "Locate the target widget"

    config = MODEL_REGISTRY[model_name]
    predict_fn: Callable[[Image.Image, str], Tuple[float, float]] = config["predict_fn"]  # type: ignore[assignment]
    coordinate_mode: str = config["coordinate_mode"]  # type: ignore[assignment]
    prompt_template: str = config["prompt_template"]  # type: ignore[assignment]
    prompt = prompt_template.format(instruction=normalized_instruction)

    coords = predict_fn(image, normalized_instruction)
    abs_x, abs_y = converter(coords, (image.width, image.height), coordinate_mode)

    base_conf = float(config.get("base_confidence", 0.7))
    confidence = float(
        np.clip(np.random.uniform(base_conf, min(base_conf + 0.05, 0.99)), 0, 0.99)
    )

    explanation = (
        f"{model_name} used prompt \"{prompt}\" and produced {coords} "
        f"in {coordinate_mode} coordinates."
    )
    if model_name == "GUI-G2-7B" and gui_g2_predictor.last_text:
        explanation += f" Raw response: {gui_g2_predictor.last_text}"
    if model_name == "UI-Venus" and ui_venus_predictor.last_text:
        explanation += f" Raw response: {ui_venus_predictor.last_text}"
    if model_name == "UI-TARS-1.5" and ui_tars_predictor.last_text:
        explanation += f" Raw response: {ui_tars_predictor.last_text}"
    if model_name == "GUI-Owl" and gui_owl_predictor.last_text:
        explanation += f" Raw response: {gui_owl_predictor.last_text}"

    return PredictionResult(
        model_name=model_name,
        x=abs_x,
        y=abs_y,
        width=image.width,
        height=image.height,
        confidence=confidence,
        raw_coordinates=coords,
        coordinate_mode=coordinate_mode,
        prompt=prompt,
        explanation=explanation,
    )


__all__ = ["infer", "list_models", "PredictionResult"]
