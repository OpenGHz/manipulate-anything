# Manipulate-Anything:Automating Real-World Robots using Vision-Language Models

*A scalable automated generation method for real-world robotic manipulation.*

[[Project Page]([https://robo-point.github.io](https://robot-ma.github.io/))] [[Data](https://drive.google.com/drive/folders/1bq3P8ywJkFMxemq9ywvj2b7LHsAhx2kg)] [[Paper](https://robot-ma.github.io/MA_paper.pdf)]

**Manipulate-Anything:Automating Real-World Robots using Vision-Language Models** [[Paper]([https://arxiv.org/pdf/2406.10721](https://arxiv.org/pdf/2406.18915))] <br>
[Jiafei Duan*](https://duanjiafei.com), [Wentao Yuan*](https://wentaoyuan.github.io),[Wilbert Pumacay](https://wpumacay.github.io),[Yi Ru Wang](https://helen9975.github.io/), [Kiana Ehsani](https://ehsanik.github.io/), [Dieter Fox](https://homes.cs.washington.edu/~fox), [Ranjay Krishna](https://ranjaykrishna.com)

![Overview](figures/overview.gif)

## Introduction
RoboPoint is a VLM that predicts image keypoint affordances given language instructions. We introduce an automatic synthetic data generation pipeline that instruction-tunes VLMs to robotic domains and needs. Compared to alternative approaches, our method requires no real-world data collection or human demonstration. In addition, RoboPoint provides a generic action space that enables language-conditioned task execution in several downstream applications such as robot navigation, manipulation, and augmented reality (AR) assistance.

## Contents
- [Install](#install)
- [Weights](#model-zoo)
- [Demo](#demo)
- [Data](#prepare-data)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

```bash
./environment_setup.sh
```

or follow the instructions below in order.

```
conda create -n robopoint python=3.10 -y
conda activate robopoint

pip install --upgrade pip  # enable PEP 660 support

# this is optional if you prefer to system built-in nvcc.
conda install -c nvidia cuda=12.1 -y

pip install -e .

# this is optional if you don't need to train the model
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Model Zoo
Version | LLM | Projector | Size | Schedule 
--------|-----|-----------|------|---------
[robopoint-v1-vicuna-v1.5-13b](https://huggingface.co/wentao-yuan/robopoint-v1-vicuna-v1.5-13b) | Vicuna-v1.5 | MLP2x | 13B | Full FT 1 epoch
[robopoint-v1-llama-2-13b](https://huggingface.co/wentao-yuan/robopoint-v1-llama-2-13b) | Llama-2 | Linear | 13B | Full FT 1 epoch
[robopoint-v1-vicuna-v1.5-13b-lora](https://huggingface.co/wentao-yuan/robopoint-v1-vicuna-v1.5-13b-lora) | Vicuna-v1.5 | MLP2x | 13B | LoRA 1 epoch
[robopoint-v1-llama-2-13b-lora](https://huggingface.co/wentao-yuan/robopoint-v1-llama-2-13b-lora) | Llama-2 | Linear | 13B | LoRA 1 epoch
[robopoint-v1-vicuna-v1.5-7b-lora](https://huggingface.co/wentao-yuan/robopoint-v1-vicuna-v1.5-7b-lora) | Vicuna-v1.5 | MLP2x | 7B | LoRA 1 epoch
[robopoint-v1-llama-2-7b-lora](https://huggingface.co/wentao-yuan/robopoint-v1-llama-2-7b-lora) | Llama-2 | Linear | 7B | LoRA 1 epoch

## Demo

### Gradio Web UI

To launch a Gradio demo locally, please run the following commands one by one. If you plan to launch multiple model workers to compare between different checkpoints, you only need to launch the controller and the web server *ONCE*.

#### Launch a controller
```bash
python -m robopoint.serve.controller --host 0.0.0.0 --port 10000
```

#### Launch a gradio web server.
```bash
python -m robopoint.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload --share
```
You just launched the Gradio web interface. Now, you can open the web interface with the URL printed on the screen. You may notice that there is no model in the model list. Do not worry, as we have not launched any model worker yet. It will be automatically updated when you launch a model worker.

#### Launch a model worker

This is the actual *worker* that performs the inference on the GPU.  Each worker is responsible for a single model specified in `--model-path`.

```bash
python -m robopoint.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 20000 --worker http://localhost:20000 --model-path wentao-yuan/robopoint-v1-vicuna-v1.5-13b
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...".  Now, refresh your Gradio web UI, and you will see the model you just launched in the model list.

You can launch as many workers as you want, and compare between different model checkpoints in the same Gradio interface. Please keep the `--controller` the same, and modify the `--port` and `--worker` to a different port number for each worker.
```bash
python -m robopoint.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port <different from 20000, say 30000> --worker http://localhost:<change accordingly, i.e. 30000> --model-path <ckpt2>
```

If you are using an Apple device with an M1 or M2 chip, you can specify the mps device by using the `--device` flag: `--device mps`.

#### Launch a model worker (8-bit / 4-bit quantized)

You can launch the model worker with quantized bits, which allows you to run the inference with reduced GPU memory footprint, potentially allowing you to run on a GPU with as few as 12GB VRAM. Note that inference with quantized bits may not be as accurate as the full-precision model. Simply append `--load-8bit` or `--load-4bit` to the **model worker** command that you are executing. Below is an example of running with 4-bit quantization.

```bash
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 20000 --worker http://localhost:20000 --model-path wentao-yuan/robopoint-v1-vicuna-v1.5-13b --load-8bit
```

#### Launch a model worker (multiple GPUs)

If the VRAM of your GPU is less than 24GB (e.g., RTX 3090, RTX 4090, etc.), you may try running it with multiple GPUs. Our latest code base will automatically try to use multiple GPUs if you have more than one GPU. You can specify which GPUs to use with `CUDA_VISIBLE_DEVICES`. Below is an example of running with the first two GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 20000 --worker http://localhost:20000 --model-path wentao-yuan/robopoint-v1-vicuna-v1.5-13b
```

#### Launch a model worker (LoRA weights, unmerged)

You can launch the model worker with LoRA weights, without merging them with the base checkpoint, to save disk space. There will be additional loading time, while the inference speed is the same as the merged checkpoints. Unmerged LoRA checkpoints are usually much smaller (less than 1GB) than the merged checkpoints (13G for 7B, and 25G for 13B).

To load unmerged LoRA weights, you simply need to pass an additional argument `--model-base`, which is the base LLM that is used to train the LoRA weights.

```bash
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 20000 --worker http://localhost:20000 --model-path /home/wentaoy/checkpoints/robopoint-v1-vicuna-v1.5-13b-lora --model-base lmsys/vicuna-13b-v1.5
```

## Visual Instruction Tuning

### Download pretrained projector weights

We use pretrained projector weights from [LLaVA](https://github.com/haotian-liu/LLaVA). The projector is trained on image-text pairs from the 558K subset of the LAION-CC-SBU dataset with BLIP captions (see [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)). When using these projector weights, please make sure that the vision encoder and the projector type are set correctly.

For CLIP-L-336px vision encoder,
```
--vision_tower openai/clip-vit-large-patch14-336
```

For MLP-2x projector,
```
--mm_projector_type mlp2x_gelu
```

For Linear projector,
```
--mm_projector_type linear
```

| Base LLM | Vision Encoder | Projection | Pretrain Data | Download |
|----------|----------------|------------|---------------|----------|
| Vicuna-13B-v1.5 | CLIP-L-336px | MLP-2x | LCS-558K | [projector](https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5) |
| Vicuna-7B-v1.5 | CLIP-L-336px | MLP-2x | LCS-558K | [projector](https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5) |
| LLaMA-2-13B-Chat | CLIP-L-336px | Linear | LCS-558K | [projector](https://huggingface.co/liuhaotian/llava-336px-pretrain-llama-2-13b-chat) |
| LLaMA-2-7B-Chat | CLIP-L-336px | Linear | LCS-558K | [projector](https://huggingface.co/liuhaotian/llava-336px-pretrain-llama-2-7b-chat) |

### Prepare data

The data mix for instruction tuning can be found on HuggingFace at [wentao-yuan/robopoint-data](https://huggingface.co/datasets/wentao-yuan/robopoint_data).

The file `robopoint_1432k.json` contains a list of 1432K VQA instances. An example looks like this
```
{
    "id": "region_ref/1033888784-63bd2a7_cam05_obj5-obj18_left",
    "image": "region_ref/1033888784-63bd2a7_cam05_obj5-obj18.png",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nIn the image, there is an item encased within a red rectangle. Pinpoint several points within the vacant space situated to the left of the object that is highlighted. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
        },
        {
            "from": "gpt",
            "value": "[(0.461, 0.527), (0.498, 0.521), (0.481, 0.521), (0.445, 0.529)]"
        }
    ]
}
```
The data mix consists of the following parts:
- 347K object reference data from our synthetic data pipeline
- 320K free space reference data from our synthetic data pipeline
- 150K GPT-generated instruction-following data from [LLaVA](https://github.com/haotian-liu/LLaVA)
- 515K VQA data from academic-oriented tasks
- 100K object detection data from [LVIS](https://www.lvisdataset.org/)

### Training

Visual instruction tuning takes around 40 hours for on 8 A100 GPUs with 80GB memory. Training scripts can be found under `scripts`.

If you are do not have enough GPU memory, you can reduce `BATCH_PER_GPU` and increase the `GRAD_ACC_STEPS` accordingly. Always keep the global batch size the same: `NUM_NODES` x `NUM_GPUS` x `BATCH_PER_GPU` x `GRAD_ACC_STEPS`.

Hyperparameters used in instruction tuning are provided below.

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| ---: | ---: | ---: | ---: | ---: | ---: |
| RoboPoint-v1-13B | 128 | 2e-5 | 1 | 2048 | 0 |

## Evaluation

Where2Place, a benchmark for spatial free-space reference on challenging real world images, can be found on HuggingFace at [wentao-yuan/where2place](https://huggingface.co/datasets/wentao-yuan/where2place).

To evaluate on Where2Place, first run the following command to generate results
```
python robopoint/eval/model_vqa.py \
    --model-path wentao-yuan/robopoint-v1-vicuna-v1.5-13b \
    --image-folder datasets/where2place/images \
    --question-file datasets/where2place/point_questions.jsonl \
    --answer-file output/robopoint-v1-vicuna-v1.5-13b.jsonl
```
Then, run the following command to compute the accuracy
```
python robopoint/eval/summarize_vqa.py --answer output/robopoint-v1-vicuna-v1.5-13b.jsonl
```
If needed, the following command visualizes the outputs of different models together with the ground truth
```
python robopoint/eval/visualize_vqa.py \
    --label gpt-4o robopoint \
    --answer output/gpt-4o.jsonl output/robopoint-v1-vicuna-v1.5-13b.jsonl \
    --output output/gpt-4o-vs-robopoint \
    --num 10
```

## Citation

If you find RoboPoint useful for your research and applications, please consider citing our paper:
```bibtex
@article{yuan2024robopoint,
  title={RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics},
  author={Yuan, Wentao and Duan, Jiafei and Blukis, Valts and Pumacay, Wilbert and Krishna, Ranjay and Murali, Adithyavairavan and Mousavian, Arsalan and Fox, Dieter},
  journal={arXiv preprint arXiv:2406.10721},
  year={2024}
}
```

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon, including the visual instruction tuning pipeline.






# Manipulate-Anything
## Installation
1. Create conda environment
```bash
conda env create -n manip_any python=3.11
```
2. Install cuda
```bash
conda install cuda -c nvidia/label/cuda-11.7.0
```
3. Install requirements
```bash
pip install -r requirements
```
4. Install pointnet2_ops
```bash
pip install pointnet2_ops/
```
5. Install PyRep
PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 16.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Once you have downloaded CoppeliaSim, you can pull PyRep from git:

```bash
cd <install_dir>
git clone https://github.com/stepjam/PyRep.git
cd PyRep
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

**Warning**: CoppeliaSim might cause conflicts with ROS workspaces. 

Finally install the python library:

```bash
pip install -r requirements.txt
pip install .
```
6. Install RLBench
```bash
cd <install_dir>
git clone -b m2t2 https://github.com/wentaoyuan/RLBench.git # note: 'm2t2' branch

cd RLBench
pip install -r requirements.txt
python setup.py develop
```
7. Activate environment
```bash
conda activate manip_any
```

## Demo
1. Download [demo data](https://drive.google.com/file/d/1bq-MjSVNKire1PclTiosavW0YDQc666t/view?usp=sharing).
2. Download [checkpoint](https://drive.google.com/file/d/1ZK2IwhHcVk-hPEC0DSvtENYUi_n0lKYk/view?usp=sharing).
3. Run meshcat server. 
```bash
meshcat-server
```
4. For picking, click on the object. Example command:
```bash
python demo_rlbench.py \
eval.task=pick \
eval.checkpoint=checkpoints/pick_and_place.pth \
rlbench.demo_path=data/demos \
rlbench.episode_id=1 \
rlbench.frame_id=0
```
4. For placing, first click on the object in gripper, then draw the bounding box for placement region by clicking two times in the new window. Example command:
```bash
python demo_rlbench.py \
eval.task=place \
eval.checkpoint=checkpoints/pick_and_place.pth \
rlbench.demo_path=data/demos \
rlbench.episode_id=1 \
rlbench.frame_id=100
```
5. Open http://127.0.0.1:7000/static to see the visualization. Press enter in terminal to see the next pose generated.
