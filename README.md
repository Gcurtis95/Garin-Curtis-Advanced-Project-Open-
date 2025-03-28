# Garin Curtis Msc Advanced Project Code
 
This is the Github Repo for my project code.

## Work 

For marking the bulk of my work was completed from lines 1 - 343 and lines 1371 to 2555 in here >>> [Modified 2D Conditional U-Net](diffusers/models/unets/unet_2d_condition.py) <br>
And >>>>> [Inference Script](StreamDiffusion_Pipeline_Inference.py)


## References

https://github.com/cumulo-autumn/StreamDiffusion/tree/main<br>
https://github.com/dzluke/DAFX2024<br>
https://github.com/olegchomp/StreamDiffusion-NDI<br>
https://github.com/huggingface/diffusers



## Abstract

 

This paper introduces a novel approach to real-time manipulation of diffusion models by integrating network bending techniques into the 2D Conditional U-Net model within the StreamDiffusion pipeline. Leveraging the flexibility of TouchDesigner, we developed an interactive tool designed for seamless integration into artistic workflows, enabling users to manipulate generative outputs dynamically and expressively. Unlike traditional text-to-image models, our tool facilitates open-ended exploration of the latent space, producing a diverse range of outputs that actively diverge from the training data. Through artistic experimentation, we demonstrate the tool's ability to generate outputs ranging from subtle enhancements to abstract transformations, unlocking new creative possibilities. This research provides a foundation for advancing real-time, artist-driven interaction with generative AI models, bridging the gap between technical innovation and creative expression. 


## Installation 

Download Miniconda or Anaconda if you haven't already

Download this Repo and open Anaconda Prompt then CD to the main directory where this Repo was stored and run the following commands. 

The following installation Instructions are taken from https://github.com/cumulo-autumn/StreamDiffusion/tree/main

### Step1: Make Environment

```bash
conda create --name my_env python=3.10
conda activate my_env
```

### Step2: Install PyTorch 

CUDA 11.8

```bash
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu118
```

CUDA 12.1

```bash
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121
```

### Install Requirements 

```bash
python setup.py develop easy_install streamdiffusion[tensorrt]
pip install -r requirements.txt


```

### Download a SD1.5 model 

E.g https://huggingface.co/wavymulder/Analog-Diffusion

# To Run

## Settings

Go to settings.json to set the initial parameters like the model and dimensions you wish to run <br>
The default settings will run fine <br>

E.G

```bash
{
	"sd_model": "wavymulder/Analog-Diffusion",
	"width": 512,
	"height": 512
}

```

## Code

Run the following command in Anaconda Prompt to start the script

```bash

python StreamDiffusion_Pipeline_Inference.py

```


## TouchDesigner 

Open the TouchDesigner file named InterfaceFinal.toe  -- NOTE please disable any VPN as this could disrupt OSC messages

Go to the NDI in Tox and select Source Name as USER-LAPTOP (Touchdesigner) you should now see the image come through

Select perform mode in top left hand corner 



Enjoy!!!







