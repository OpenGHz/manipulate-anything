# Manipulate Anything
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