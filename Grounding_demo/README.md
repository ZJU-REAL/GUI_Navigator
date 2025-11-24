# Grounding Demo

GUI grounding showcase built with Gradio. The goal is to demonstrate how a GUI Agent can predict interaction coordinates on screenshots while offering a clean web UI for demos and evangelism.

## Features
- Dropdown model selector for UI-TARS-1.5, GUI-G2-7B, GUI-Owl, UI-Venus powered by a unified `infer(model_name, image, instruction)` API (all models load their respective local weights).
- Image uploader plus task-instruction textbox so different prompt templates can be simulated.
- Visualization pane overlays the predicted pointer; JSON panel exposes absolute/relative coordinates, confidence, and the prompt actually used.

## Project Structure
```
.
|- app.py          # Gradio entrypoint
|- inference.py    # Coordinate conversion + model registry with prompt templates
`- requirements.txt
```

## Running the Demo
1. Create a Python environment (3.10+) and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the app:
   ```bash
   python app.py
   ```
3. Open the local Gradio link, upload a screenshot, type the task instruction, select a model, and click **Predict**.

## Extending with Real Models
1. Place real model weights under `inclusionAI/<ModelName>` (GUI-G2-7B already reads from `inclusionAI/GUI-G2-7B` or the `GUI_G2_MODEL_PATH` environment variable).
2. Add/update entries in `MODEL_REGISTRY` inside `inference.py` with the prompt template, coordinate mode, and inference hook.
3. Customize the prompt template if the model requires a specific format; `infer` automatically expands it.
4. The UI will immediately reflect new coordinates, confidence, prompt text, and visualization without further changes.

## GUI-G2-7B Setup Notes
- Install the heavy dependencies listed in `requirements.txt` (`torch`, `transformers==4.49.0`, `qwen-vl-utils`, etc.).
- Download the GUI-G2-7B weights to `inclusionAI/GUI-G2-7B` (or point to another directory via `set GUI_G2_MODEL_PATH=...` on Windows / `export GUI_G2_MODEL_PATH=...` on *nix).
- Ensure the machine has enough GPU memory; the loader defaults to `device_map="auto"` and uses `torch.bfloat16`.

## UI-TARS-1.5 Setup Notes
- Place the UI-TARS-1.5-7B weights under `UI-TARS-1.5-7B` (or set `UI_TARS_MODEL_PATH` to the actual directory).
- The predictor mirrors the official prompt format `[x1,y1,x2,y2]` and falls back to a heuristic only if parsing fails.
- Runtime requirements match GUI-G2-7B (PyTorch 2.4.1 CUDA wheels, `transformers==4.49.0`, `qwen-vl-utils`, optional `flash-attn`).

## GUI-Owl Setup Notes
- Place the weights under `GUI-Owl` (or override via `GUI_OWL_MODEL_PATH`).
- Prompt enforces `[x1,y1,x2,y2]` output; the loader converts the bbox to a click point and falls back to heuristics only if parsing fails.
- Dependencies保持与其它模型一致；如果未安装 flash-attn 会自动回退到标准注意力实现。