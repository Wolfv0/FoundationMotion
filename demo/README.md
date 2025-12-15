## Installation

```bash
pip install -r requirements.txt
```

Or install dependencies manually:

```bash
pip install gradio torch transformers accelerate qwen-vl-utils opencv-python-headless numpy pillow
```

---

## Run the demo

```bash
python app.py
```

Then open your browser and navigate to the local URL shown in the terminal (typically `http://127.0.0.1:7860`).

---

## 💡 Usage

1. **Upload** a video (mp4, mov, webm)
2. **Ask** a question (or leave blank for auto-description)
3. **Click** "Ask" and wait for the response

### Example Questions

- *"What is happening in this video?"*
- *"Describe the main objects and actions."*
- *"How is the person moving?"*

---

## ⚙️ Advanced Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| Max Frames | 8 | Number of frames to extract (4-16) |
| Max New Tokens | 256 | Maximum response length (64-512) |
| Temperature | 0.0 | Sampling temperature (0 = deterministic) |

---

## 📄 License

See the main project [README](../README.md) for license information.
