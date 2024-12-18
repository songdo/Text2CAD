<div align="center">

# Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts


[Mohammad Sadil Khan*](https://scholar.google.com/citations?user=XIDQo_IAAAAJ&hl=en&authuser=1) · [Sankalp Sinha*](https://scholar.google.com/citations?user=QYcfOjEAAAAJ&hl=en&authuser=1&oi=ao) · [Talha Uddin Sheikh](https://scholar.google.com/citations?hl=en&authuser=1&user=yW7VfAgAAAAJ) · [Didier Stricker](https://scholar.google.com/citations?hl=en&authuser=1&user=ImhXfxgAAAAJ) · [Sk Aziz Ali](https://scholar.google.com/citations?hl=en&authuser=1&user=zywjMeMAAAAJ) · [Muhammad Zeshan Afzal](https://scholar.google.com/citations?user=kHMVj6oAAAAJ&hl=en&authuser=1&oi=ao)

_*equal contributions_

<h2> NeurIPS 2024 (Spotlight 🤩) </h2>

<a href="https://openreview.net/pdf?id=5k9XeHIK3L">
  <img src="https://img.shields.io/badge/Paper-3498db?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=2c3e50&borderRadius=10" alt="Paper" />
</a>
<a href="https://sadilkhan.github.io/text2cad-project/">
  <img src="https://img.shields.io/badge/Project-2ecc71?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=27ae60&borderRadius=10" alt="Project" />
</a>

<!-- <a href="link"><img src="https://img.shields.io/badge/Code-e74c3c?style=for-the-badge&logoWidth=40&logoColor=white" alt="Code" /></a> -->


</div>


# ⚙️ Installation

## 🌍 Environment

- 🐧 Linux
- 🐍 Python >=3.9

## 📦 Dependencies

```bash
$ conda env create --file environment.yml
```

# ✅ Todo List

- [x] Release Data Preparation Code
- [x] Release Training Code
- [ ] Release Inference Code

# 📊 Data Preparation

Download the DeepCAD data from [here](https://github.com/ChrisWu1997/DeepCAD?tab=readme-ov-file#data).

- Generate Vector Representation from DeepCAD Json

```bash
$ python3 json2vec.py --input_dir $DEEPCAD_JSON_DIR --split_json $TRAIN_TEST_VAL_JSON --output_dir $OUTPUT_DIR --bit 8 --max_cad_seq_len 272 --max_workers $NUM_WORKERS --padding --deduplicate
```

- Download the text annotations from [here](https://huggingface.co/datasets/SadilKhan/Text2CAD).

# 🚀 Training

In the `config/trainer.yaml`, provide the following path.

- `cache_dir`: The directory to load model weights from Huggingface.
- `cad_seq_dir`: The root directory that contains the ground truth CAD vector.
- `prompt_path`: Path for the text annotation.
- `split_filepath`: Json file containing the UIDs for train, test or validation.
- `log_dir`: Directory for saving _logs, outputs, checkpoints_.
- `checkpoint_path` (Optional): For resuming training after some epochs.

```bash
$ cd Cad_VLM
$ python3 train.py --config_path config/trainer.yaml
```

# 🤖 Inference



# 💻 Run Demo



# 👥 Contributors
Our project owes its success to the invaluable contributions of these remarkable individuals. We extend our heartfelt gratitude for their dedication and support.


<a href="https://scholar.google.com/citations?hl=en&authuser=1&user=QYcfOjEAAAAJ">
  <img src="https://av.dfki.de/wp-content/uploads/avatars/162/1722545138-bpfull.png" width="50" height="50" style="border-radius: 50%;">
</a>
<a href="https://github.com/saali14">
  <img src="https://github.com/saali14.png" width="50" height="50" style="border-radius: 50%;">
</a>
<a href="https://scholar.google.de/citations?user=yW7VfAgAAAAJ&hl=en">
  <img src="https://scholar.google.de/citations/images/avatar_scholar_128.png" width="50" height="50" style="border-radius: 50%;">
</a>


# 📜 Citation

If you use this dataset in your work, please consider citing the following publications.


```
@inproceedings{khan2024textcad,
title={Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts},
author={Mohammad Sadil Khan and Sankalp Sinha and Sheikh Talha Uddin and Didier Stricker and Sk Aziz Ali and Muhammad Zeshan Afzal},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=5k9XeHIK3L}
}

@InProceedings{Khan_2024_CVPR,
author = {Khan, Mohammad Sadil and Dupont, Elona and Ali, Sk Aziz and Cherenkova, Kseniya and Kacem, Anis and Aouada, Djamila},
title = {CAD-SIGNet: CAD Language Inference from Point Clouds using Layer-wise Sketch Instance Guided Attention},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2024},
pages = {4713-4722}
}
```
