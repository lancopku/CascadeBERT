# CascadeBERT



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

If you have any problems, raise a issue or contact with [me](mailto:tobiaslee@foxmail.com)

