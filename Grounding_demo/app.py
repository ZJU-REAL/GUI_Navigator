"""Gradio application for the GUI Grounding Demo."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import gradio as gr
from PIL import Image, ImageDraw

from inference import PredictionResult, infer, list_models

MODEL_INFO: Dict[str, str] = list_models()
POINTER_IMAGE_PATH = Path(__file__).resolve().parent / "2.png"
try:
    POINTER_IMAGE = Image.open(POINTER_IMAGE_PATH).convert("RGBA")
except Exception:
    POINTER_IMAGE = None


def draw_pointer(image: Image.Image, result: PredictionResult) -> Image.Image:
    overlay = image.convert("RGBA")
    if POINTER_IMAGE is None:
        radius = max(int(min(image.size) * 0.02), 8)
        draw = ImageDraw.Draw(overlay)
        bbox = [
            result.x - radius,
            result.y - radius,
            result.x + radius,
            result.y + radius,
        ]
        draw.ellipse(bbox, outline="red", width=4, fill=(255, 0, 0, 60))
        draw.line([(result.x, 0), (result.x, image.height)], fill="red", width=2)
        draw.line([(0, result.y), (image.width, result.y)], fill="red", width=2)
        return overlay

    pointer = POINTER_IMAGE
    base_size = int(min(image.size) * 0.12)
    pointer_w, pointer_h = pointer.size
    aspect = pointer_w / pointer_h if pointer_h else 1
    target_w = max(base_size, 32)
    target_h = int(target_w / aspect)
    pointer_resized = pointer.resize((target_w, max(target_h, 32)), Image.LANCZOS)

    paste_x = int(round(result.x - pointer_resized.width / 2))
    paste_y = int(round(result.y - pointer_resized.height / 2))
    paste_x = max(min(paste_x, overlay.width - pointer_resized.width), 0)
    paste_y = max(min(paste_y, overlay.height - pointer_resized.height), 0)

    overlay.paste(pointer_resized, (paste_x, paste_y), pointer_resized)
    return overlay


def predict(model_name: str, image: Image.Image, instruction: str):
    if image is None:
        raise gr.Error("Please upload a UI screenshot first.")
    if not instruction or not instruction.strip():
        raise gr.Error("Enter a task instruction, e.g. \"click the search button\".")
    result = infer(model_name, image, instruction)
    overlay = draw_pointer(image, result)
    metadata = {
        "model": result.model_name,
        "instruction": instruction,
        "prompt": result.prompt,
        "coordinate_mode": result.coordinate_mode,
        "raw_coordinates": [round(c, 4) for c in result.raw_coordinates],
        "absolute_coordinates": {"x": result.x, "y": result.y},
        "relative_coordinates": {
            "x": round(result.relative_coordinates[0], 4),
            "y": round(result.relative_coordinates[1], 4),
        },
        "confidence": round(result.confidence, 3),
        "explanation": result.explanation,
    }
    return overlay, metadata


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="GUI Grounding Demo") as demo:
        gr.Markdown(
            """
            # GUI Grounding Demo
            Upload a UI screenshot, enter a task instruction, choose a model, and inspect the predicted pointer.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=list(MODEL_INFO.keys()),
                    value=next(iter(MODEL_INFO)),
                    info="Select which grounding model to run.",
                )
                image_input = gr.Image(
                    label="Upload image", 
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                instruction_input = gr.Textbox(
                    label="Task instruction / Prompt",
                    placeholder="e.g. click the search button in the top right",
                    lines=3,
                    value="click the search button in the top right",
                )
                predict_btn = gr.Button("Predict", variant="primary")
            with gr.Column(scale=2):
                overlay_output = gr.Image(label="Pointer visualization", type="pil")
                metadata_output = gr.JSON(label="Prediction details")
        predict_btn.click(
            predict,
            inputs=[model_dropdown, image_input, instruction_input],
            outputs=[overlay_output, metadata_output],
        )
    return demo


def main():
    demo = build_interface()
    demo.launch()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
