# CascadeBERT

Implementation code of [CascadeBERT: Accelerating Inference of Pre-trained Language Models via Calibrated Complete Models Cascade](https://arxiv.org/abs/2012.14682), Findings of EMNLP 2021

## Requirements

We recommend using Anaconda for setting up the environment of experiments: 

```bash
git clone https://github.com/lancopku/CascadeBERT.git
cd CascadeBERT
conda create -n cascadebert python=3.7
conda activate cascadebert
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
pip install -r requirements
```
 

## Data & Model Preparement

We provide the training data with associated data difficulty for a 2L BERT-Complete model.

You can download it from [Google Drive](https://drive.google.com/file/d/1bBgfUjvvlxuY_S_ep0gU4X6Ze1Q2bZRB/view?usp=sharing)
, and 2L BERT-Complete model can be downloaded from [Google Drive](https://drive.google.com/file/d/18DZ-UoKZKIVuSQJORjKiTPTKi6r87-dM/view?usp=sharing)

## Training & Inference

We provide a sample running script for MRPC, unzip the downloaded data and model, modify the PATH in the  `glue_mrpc.sh`, and

> sh glue_mrpc.sh


You can obtain results in the `saved_models` path.


## Contact 

If you have any problems, raise a issue or contact [Lei Li](mailto:tobiaslee@foxmail.com)

## Citation

If you find this repo helpful, we'd appreciate it a lot if you can cite the corresponding paper:
```
@article{li2020accelerating,
  title={Accelerating pre-trained language models via calibrated cascade},
  author={Li, Lei and Lin, Yankai and Ren, Shuhuai and Chen, Deli and Ren, Xuancheng and Li, Peng and Zhou, Jie and Sun, Xu},
  journal={arXiv preprint arXiv:2012.14682},
  year={2020}
}
```

