# Manipulate-Anything:Automating Real-World Robots using Vision-Language Models

*A scalable automated generation method for real-world robotic manipulation.*

[[Project Page]([https://robo-point.github.io](https://robot-ma.github.io/))] [[Data](https://drive.google.com/drive/folders/1bq3P8ywJkFMxemq9ywvj2b7LHsAhx2kg)] [[Paper](https://robot-ma.github.io/MA_paper.pdf)]

**Manipulate-Anything:Automating Real-World Robots using Vision-Language Models** [[Paper]([https://arxiv.org/pdf/2406.10721](https://arxiv.org/pdf/2406.18915))] <br>
[Jiafei Duan*](https://duanjiafei.com), [Wentao Yuan*](https://wentaoyuan.github.io), [Wilbert Pumacay](https://wpumacay.github.io), [Yi Ru Wang](https://helen9975.github.io/), [Kiana Ehsani](https://ehsanik.github.io/), [Dieter Fox](https://homes.cs.washington.edu/~fox), [Ranjay Krishna](https://ranjaykrishna.com)

![Overview](overview.gif)

## Introduction
MANIPULATE-ANYTHING, a scalable automated generation method for real-world robotic manipulation. Unlike prior work, our method can operate in real-world environments without any privileged state information, hand-designed skills, and can manipulate any static object.

## Contents
- [Env Setup & Installation]
- [Data Generation]
- [Evaluation]

## Env Setup & Installation

There is a need of four different repo (including this) to setup Manipulate-Anything.

1. Create conda environment
```bash
conda env create -n manip_any python=3.11
conda install cuda -c nvidia/label/cuda-11.7.0
conda activate manip_any
```

2. Setup and install Manipulate-Anything-QWenVL
```bash
git clone https://github.com/Robot-MA/QWen-VL-MA.git
```
Go into the [QWen-VL-MA](https://github.com/Robot-MA/QWen-VL-MA) and follow the steps.

3. Install PyRep
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


4. Install current repo
```bash
pip install pointnet2_ops/
cd pointnet2_ops
pip install -r requirements.txt
pip install .
```

```bash
git clone https://github.com/Robot-MA/manipulate-anything.git
cd RLBench
pip install -r requirements.txt
python setup.py develop
```

## Data Generation

1. Download [checkpoint](https://drive.google.com/file/d/1ZK2IwhHcVk-hPEC0DSvtENYUi_n0lKYk/view?usp=sharing).
2. Setup GPT4V API-key.
```bash
meshcat-server
```  
4. Run meshcat server. 
```bash
export OPENAI_API_KEY="your_api_key_here"
```
4. Zero-shot data generation. Example task (Put_block_on_target):
```bash
python demo_rlbench.py \
eval.task=pick \
eval.checkpoint=checkpoints/pick_and_place.pth \
rlbench.demo_path=data/demos \
rlbench.episode_id=1 \
rlbench.frame_id=0
```
5. Open http://127.0.0.1:7000/static to see the visualization. Press enter in terminal to see the next pose generated.

## Evaluation




## Citation

If you find Manipulate-Anything useful for your research and applications, please consider citing our paper:
```bibtex
@article{duan2024manipulate,
  title={Manipulate-anything: Automating real-world robots using vision-language models},
  author={Duan, Jiafei and Yuan, Wentao and Pumacay, Wilbert and Wang, Yi Ru and Ehsani, Kiana and Fox, Dieter and Krishna, Ranjay},
  journal={arXiv preprint arXiv:2406.18915},
  year={2024}
}
```

