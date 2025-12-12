# FoundationMotion

Stay tuned — more will come in a few days! [12/11/2025]

## Test MotionBench

### Gemini

```bash
pip install google-generativeai tqdm fire huggingface_hub
```

### Usage

You can run the script from the command line with default configuration:

```bash
export GEMINI_API_KEY=<your gemini API key>
hf download zai-org/MotionBench --repo-type dataset --local-dir </your/path/to/motionbench>
python gemini_motionbench.py --max_workers=4 --model=gemini-2.5-pro --motionbench_base_path=</your/path/to/motionbench>
```

#### Options

You may specify command-line options for customization:

- `--max_workers=N` : Set the number of parallel workers (default: 8)
- `--model=MODEL_NAME` : Specify the Gemini model (e.g., `gemini-2.5-flash`)
- `--motionbench_base_path=PATH` : Use a local MotionBench dataset path instead of downloading
    - If `motionbench_base_path` is not specified, the script automatically downloads the public MotionBench dataset from Hugging Face.
 

#### FoundationMotion is also referred to as Wolf V2 🐺, the second chapter in the Wolf series: https://wolfv0.github.io/.
