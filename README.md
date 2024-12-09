# Manipulate-Anything:Automating Real-World Robots using Vision-Language Models

*A scalable automated generation method for real-world robotic manipulation.*

[[Project Page]([https://robo-point.github.io](https://robot-ma.github.io/))] [[Data](https://drive.google.com/drive/folders/1bq3P8ywJkFMxemq9ywvj2b7LHsAhx2kg)] [[Paper](https://robot-ma.github.io/MA_paper.pdf)]

**Manipulate-Anything:Automating Real-World Robots using Vision-Language Models** [[Paper]([https://arxiv.org/pdf/2406.10721](https://arxiv.org/pdf/2406.18915))] <br>
[Jiafei Duan*](https://duanjiafei.com), [Wentao Yuan*](https://wentaoyuan.github.io), [Wilbert Pumacay](https://wpumacay.github.io), [Yi Ru Wang](https://helen9975.github.io/), [Kiana Ehsani](https://ehsanik.github.io/), [Dieter Fox](https://homes.cs.washington.edu/~fox), [Ranjay Krishna](https://ranjaykrishna.com)

![Overview](figures/overview.gif)

## Introduction
MANIPULATE-ANYTHING, a scalable automated generation method for real-world robotic manipulation. Unlike prior work, our method can operate in real-world environments without any privileged state information, hand-designed skills, and can manipulate any static object.

## Contents
- [Env Setup & Installation](#install)
- [Data Generation](#train)
- [Evaluation](#evaluation)

## Env Setup & Installation

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



## Data Generation

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

