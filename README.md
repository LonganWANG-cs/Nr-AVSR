# Robust Audio-Visual Speech Recognition System with Noisy Labels
PyTorch implementation for Robust Audio-Visual Speech Recognition System with Noisy. (NUS SOC CS 4347). 

## Nr-AVSR
The effectiveness of Audio-Visual Speech Recognition (AVSR) systems, typically powered by deep neural networks, is heavily reliant on high-quality, well-aligned audio-visual data.
However, real-world datasets often contain label noise due to annotation errors and data complexity, leading to mismatches between audio and textual ground truth.
This label noise can significantly hinder model performance by causing the model to memorize irrelevant noise patterns instead of generalizing from meaningful features.
To address these challenges, we propose the Noise Resistance Audio-Visual Speech Recognition (Nr-AVSR) framework, which is designed to enhance the resilience of AVSR systems against noisy labels.
The Nr-AVSR framework incorporates robust label selection and regularization techniques to minimize the impact of noisy labels and improve the model’s ability to focus on reliable data.
Extensive experiments on the SpeechSilde dataset demonstrate that Nr-AVSR significantly reduces Word Error Rate (WER) under various noisy conditions, showcasing its potential for reliable and robust speech recognition in real-world noisy environments.
The proposed framework offers an effective solution for improving AVSR systems' performance in practical applications where noise is inevitable.

## Play with Our Model
Before running the main script, you need to generate the the noise. To do this, run `generate_noise.py`:
```bash
python ./generate_noise.py
```
Once the noise are generated, you can run the main script `test.sh` and `train.sh` to play with the model on slrum:
```bash
sbatch test.sh
sbatch train.sh
```

# Installation
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout tags/v4.35.2
pip install -e .
cd ..
git clone https://github.com/huggingface/peft.git
cd peft
git checkout tags/v0.6.0
pip install -e .
cd ..
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/ddlBoJack/SLAM-LLM.git
cd SLAM-LLM
pip install  -e .
```

For some examples, you may need to use `fairseq`, the command line is as follows:
```
# you need to install fairseq before SLAM-LLM
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

The dataset and pretrained model are available at the [SLAM-LLM GitHub repository](https://github.com/X-LANCE/SLAM-LLM/blob/main/README.md).

# Acknowledge
- We borrow code from [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM/blob/main/README.md) for the inference process. 
- We borrow code from [Llama-Recipes](https://github.com/meta-llama/llama-recipes) for the training process. 
- We borrow code from [Fairseq](https://github.com/facebookresearch/fairseq) for deepspeed configuration. 

