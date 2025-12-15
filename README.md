# FoundationMotion (FM)

[Paper](https://arxiv.org/pdf/2512.10927) | [Project Page](https://yulugan.com/projects/FoundationMotion.html) | [Model](https://huggingface.co/WoWolf/models) | [Datasets](https://huggingface.co/datasets/WoWolf/v2-dev/tree/main) | [Citation](#citation)

FoundationMotion offers a scalable way to curate detailed motion datasets, enabling effective fine-tuning of diverse models (VLM / VLA / world models) to improve motion and spatial reasoning. 

![Dataset Example](assets/data_example.gif)


⏰ 👷🏻‍♂️ Stay tuned — more will come in a few days! [12/11/2025]


## Environment Setup

If you want to construct datasets using our dataset curation pipeline: see installation instructions in [data_pipeline/README.md](data_pipeline/README.md)

If you want to use our finetuned model:

```bash
pip install fire tqdm huggingface-hub
pip install -U https://github.com/NVlabs/VILA
```






## Examples

### [Data Curation] Process a single video
- [data_pipeline/process_single_video.py](data_pipeline/process_single_video.py) - script to process a single video to get trajectories, captions, and question–answer pairs.

```bash
python process_single_video.py --video_path /path/to/video.mp4 --base_output_dir /path/to/output
```

### [Model Inference] Process a single video
- [examples/process_single_video.py](data_pipeline/demo_nvila.py) - script to process a single video using our model.

```bash
python demo_nvila.py --video_path /path/to/video.mp4 --prompt "Your question here"
```


## Citation
If you use our work or our implementation in this repo, or find them helpful, please consider giving a citation in the following format.

```bash
@misc{gan2025foundationmotionautolabelingreasoningspatial,
    title={FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos}, 
    author={Yulu Gan and Ligeng Zhu and Dandan Shan and Baifeng Shi and Hongxu Yin and Boris Ivanovic and Song Han and Trevor Darrell and Jitendra Malik and Marco Pavone and Boyi Li},
    year={2025},
    eprint={2512.10927},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2512.10927}, 
}
```


#### FoundationMotion is also referred to as Wolf V2 🐺, the second chapter in the Wolf series: https://wolfv0.github.io/.
