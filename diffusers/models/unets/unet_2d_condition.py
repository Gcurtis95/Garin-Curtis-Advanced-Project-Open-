# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
# import argparse

import scipy.linalg
import random
from kornia import morphology, filters

from pythonosc import dispatcher, osc_server, udp_client
import threading




#####   For marking the bulk of my work was completed from lines 1 - 343 and lines 1371 to 2555 ##############################
#
#
#
#
#
#######    Declaring Global OSC Variables ###########################################################################

Inversion_OSC_Radius = 0
add_full_Value = 0
add_sparse_value = 0  ## scalar only
add_noise_value = 0  ## scalar only
subtract_full_value = 0  ## scalar only
thresh_value = 0

DownSampleBend = 0
UpSampleBend = 0
# Return a fn that applies soft thresholding to x
# In soft thresholding, values less than r are set to 0 and values greater than r are shrunk towards zero
    
# soft_threshold_value = 0   ###Not sure if this works 
# soft_threshold2_value = 0
 #### each dim function needs an index and Dim value  
add_dim_value = 0
invert_dim_value= 0

dimension = 0  ## can be 0,1 0r 2
indexx = 0    ### depends on what layer, layer 4 max is 8 

add_rand_cols_value = 0  
add_rand_rows_value = 0## these 2 both have k values 
k_value = 0              ###### K has between 0 and 1



# add_normal_value = 0  ##  does not work 
rotate_z_value = 0
rotate_x_value = 0
rotate_y_value = 0
reflect_value = 0
hadamard1_value = 0  ### interetsing 
# hadamard2_value = 0 ## not wokring   
dilation_value = 0
erosion_value = 0
# sobel_value = True 
Layer = 3
upidx = 0
InnerLayerBend = 0
InnerLayer = 0

InnerLayerArray = []


osc_server_running = True

## OSC Transform Functions

def Inversion_OSC(address, *args):
    global Inversion_OSC_Radius
    if address == "/v1":
        Inversion_OSC_Radius = args[0]
        #print(f"Received OSC value: {Inversion_OSC_Radius}")

def Add_Full_OSC(address, *args):
    global add_full_Value
    if address == "/v2":
        add_full_Value = args[0]

# def Add_Sparse_OSC(address, *args):
#     global add_sparse_value
#     if address == "/v3":
#         add_sparse_value = args[0]

def Add_Noise_OSC(address, *args):
    global add_noise_value
    if address == "/v4":
        add_noise_value = args[0]
        
def Subtract_Ful_OSC(address, *args):
    global subtract_full_value
    if address == "/v5":
        subtract_full_value = args[0]
        
def Thresh_OSC(address, *args):
    global thresh_value
    if address == "/v6":
        thresh_value = args[0]
        
# def Soft_Thresh_OSC(address, *args):
#     global soft_threshold_value
#     if address == "":
#         soft_threshold_value = args[0]
        
# def Soft_Thresh2_OSC(address, *args):
#     global soft_threshold2_value
#     if address == "":
#         soft_threshold2_value = args[0]
        
        
 #### these three are for add_dim function       
def add_dim_value_OSC(address, *args):
    global add_dim_value
    if address == "/v7":
        add_dim_value = args[0]
        
def invert_dim_value_OSC(address, *args):
    global invert_dim_value
    if address == "/v8":
        invert_dim_value = args[0]
        
def dimension_value_OSC(address, *args):
    global dimension
    if address == "/v9":
        dimension = args[0]
        
def indexx_value_OSC(address, *args):
    global indexx
    if address == "v/10":
        indexx = args[0]
        
        
        
        
def add_rand_cols_value_OSC(address, *args):
    global add_rand_cols_value
    if address == "/v11":
        add_rand_cols_value = args[0]
        
def k_value_OSC(address, *args):
    global k_value
    if address == "/v13":
        k_value = args[0]
        
def add_rand_rows_value_value_OSC(address, *args):
    global add_rand_rows_value
    if address == "/v12":
        add_rand_rows_value = args[0]
        

        
        
# def add_normal_value_OSC(address, *args):
#     global add_normal_value
#     if address == "":
#         add_normal_value = args[0]
        
def rotate_z_value_OSC(address, *args):
    global rotate_z_value
    if address == "/v14":
        rotate_z_value = args[0]
        
def rotate_x_value_OSC(address, *args):
    global rotate_x_value
    if address == "/v15":
        rotate_x_value = args[0]
        

def rotate_y_value_OSC(address, *args):
    global rotate_y_value
    if address == "/v16":
        rotate_y_value = args[0]
        
def reflect_value_OSC(address, *args):
    global reflect_value
    if address == "":
        reflect_value = args[0]
        
def hadamard1_value_OSC(address, *args):
    global hadamard1_value
    if address == "/v17":
        hadamard1_value = args[0]
        
# def hadamard2_value_OSC(address, *args):
#     global hadamard2_value
#     if address == "":
#         hadamard2_value = args[0]
        
def dilation_value_OSC(address, *args):
    global dilation_value
    if address == "/v18":
        dilation_value = args[0]
        
        
def erosion_value_OSC(address, *args):
    global erosion_value
    if address == "/v19":
        erosion_value = args[0]
        
# def sobel_value_OSC(address, *args):
#     global sobel_value
#     if address == "":
#         sobel_value = args[0]
        
        
def LAYER_value_OSC(address, *args):
    global Layer
    if address == "/v20":
        Layer = args[0]

               
def DownSampleBend_OSC(address, *args):
    global DownSampleBend
    if address == "/v21":
        DownSampleBend = args[0]


def upidx_value_OSC(address, *args):
    global upidx
    if address == "/v22":
        upidx = args[0]
        
def UpSampleBendBend_OSC(address, *args):
    global UpSampleBend
    if address == "/v23":
        UpSampleBend = args[0]


        
        
        
    
        

    
############  Main OSC Function   ##################################################



def OSC():

    #OSC SERVER
    _dispatcher = dispatcher.Dispatcher()
    
    global osc_server_running
    
    #Inversion OSC
    _dispatcher.map("/v1", Inversion_OSC)
    # Add_Full_OSC
    _dispatcher.map("/v2", Add_Full_OSC)
    #A dd_Sparse_OSC
    # _dispatcher.map("/v3", Add_Sparse_OSC)
    # add noise osc
    _dispatcher.map("/v4", Add_Noise_OSC)
    # subtract full OSC
    _dispatcher.map("/v5", Subtract_Ful_OSC)
    # Thresh_OSC
    _dispatcher.map("/v6", Thresh_OSC)
    
    # # Soft_Thresh_OSC
    # _dispatcher.map("", Soft_Thresh_OSC)
    # # Soft_Thresh2_OSC
    # _dispatcher.map("", Soft_Thresh2_OSC)
    
     #### these four are for add_dim function  
    # add_dim_value_OSC
    _dispatcher.map("/v7", add_dim_value_OSC)
    # invert_dim_value_OSC
    _dispatcher.map("/v8", invert_dim_value_OSC)
    # dimension_value_OSC
    _dispatcher.map("/v9", dimension_value_OSC)
    # indexx_value_OSC
    _dispatcher.map("/v10", indexx_value_OSC)
    
    
    
    # add_rand_cols_value_OSC
    _dispatcher.map("/v11", add_rand_cols_value_OSC)
    # add_rand_cols_k_value_OSC
    _dispatcher.map("/v13", k_value_OSC)
    # add_rand_rows_value_value_OSC
    _dispatcher.map("/v12", add_rand_rows_value_value_OSC)
    # add_normal_value_OSC
    # _dispatcher.map("", add_normal_value_OSC)
    rotate_z_value_OSC
    _dispatcher.map("/v14", rotate_z_value_OSC)
    # rotate_x_value_OSC
    _dispatcher.map("/v15", rotate_x_value_OSC)
    # rotate_y_value_OSC
    _dispatcher.map("/v16", rotate_y_value_OSC)
    # reflect_value_OSC
    _dispatcher.map("", reflect_value_OSC)
    # hadamard1_value_OSC
    _dispatcher.map("v/17", hadamard1_value_OSC)
    # hadamard2_value_OSC
    # _dispatcher.map("", hadamard2_value_OSC)
    # dilation_value_OSC
    _dispatcher.map("/v18", dilation_value_OSC)
    # erosion_value_OSC
    _dispatcher.map("/v19", erosion_value_OSC)
    # LAYER_value_OSC
    _dispatcher.map("/v20", LAYER_value_OSC)
    
    # DownSampleBend_OSC On/Off
    _dispatcher.map("/v21", DownSampleBend_OSC)
    # upidx_value_OSC 
    _dispatcher.map("/v22", upidx_value_OSC)
    
    # UpSampleBend_OSC On/Off
    _dispatcher.map("/v23", UpSampleBendBend_OSC)
    
    
    
    
    
    server = osc_server.ThreadingOSCUDPServer(
        ("127.0.0.1", 5000), _dispatcher)
    #print("Serving on {}".format(server.server_address))
    server.serve_forever()
    
    while osc_server_running:
        server.handle_request()

    print("OSC server shut down.")
    
    
osc_thread = threading.Thread(target=OSC)
osc_thread.start()    


#######################################################################################################


import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from ..activations import get_activation
from ..attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)
from ..embeddings import (
    GaussianFourierProjection,
    GLIGENTextBoundingboxProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from ..modeling_utils import ModelMixin
from .unet_2d_blocks import (
    get_down_block,
    get_mid_block,
    get_up_block,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor = None


class UNet2DConditionModel(
    ModelMixin, ConfigMixin, FromOriginalModelMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin
):
    r"""
    A conditional 2D UNet model that takes a noisy sample, conditional state, and a timestep and returns a sample
    shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2DCrossAttn"`):
            Block type for middle of UNet, it can be one of `UNetMidBlock2DCrossAttn`, `UNetMidBlock2D`, or
            `UNetMidBlock2DSimpleCrossAttn`. If `None`, the mid block layer is skipped.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")`):
            The tuple of upsample blocks to use.
        only_cross_attention(`bool` or `Tuple[bool]`, *optional*, default to `False`):
            Whether to include self-attention in the basic transformer blocks, see
            [`~models.attention.BasicTransformerBlock`].
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, normalization and activation layers is skipped in post-processing.
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        cross_attention_dim (`int` or `Tuple[int]`, *optional*, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int`, `Tuple[int]`, or `Tuple[Tuple]` , *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unets.unet_2d_blocks.CrossAttnDownBlock2D`], [`~models.unets.unet_2d_blocks.CrossAttnUpBlock2D`],
            [`~models.unets.unet_2d_blocks.UNetMidBlock2DCrossAttn`].
        reverse_transformer_layers_per_block : (`Tuple[Tuple]`, *optional*, defaults to None):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`], in the upsampling
            blocks of the U-Net. Only relevant if `transformer_layers_per_block` is of type `Tuple[Tuple]` and for
            [`~models.unets.unet_2d_blocks.CrossAttnDownBlock2D`], [`~models.unets.unet_2d_blocks.CrossAttnUpBlock2D`],
            [`~models.unets.unet_2d_blocks.UNetMidBlock2DCrossAttn`].
        encoder_hid_dim (`int`, *optional*, defaults to None):
            If `encoder_hid_dim_type` is defined, `encoder_hidden_states` will be projected from `encoder_hid_dim`
            dimension to `cross_attention_dim`.
        encoder_hid_dim_type (`str`, *optional*, defaults to `None`):
            If given, the `encoder_hidden_states` and potentially other embeddings are down-projected to text
            embeddings of dimension `cross_attention` according to `encoder_hid_dim_type`.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
        num_attention_heads (`int`, *optional*):
            The number of attention heads. If not defined, defaults to `attention_head_dim`
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for ResNet blocks (see [`~models.resnet.ResnetBlock2D`]). Choose from `default` or `scale_shift`.
        class_embed_type (`str`, *optional*, defaults to `None`):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`,
            `"timestep"`, `"identity"`, `"projection"`, or `"simple_projection"`.
        addition_embed_type (`str`, *optional*, defaults to `None`):
            Configures an optional embedding which will be summed with the time embeddings. Choose from `None` or
            "text". "text" will use the `TextTimeEmbedding` layer.
        addition_time_embed_dim: (`int`, *optional*, defaults to `None`):
            Dimension for the timestep embeddings.
        num_class_embeds (`int`, *optional*, defaults to `None`):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
        time_embedding_type (`str`, *optional*, defaults to `positional`):
            The type of position embedding to use for timesteps. Choose from `positional` or `fourier`.
        time_embedding_dim (`int`, *optional*, defaults to `None`):
            An optional override for the dimension of the projected time embedding.
        time_embedding_act_fn (`str`, *optional*, defaults to `None`):
            Optional activation function to use only once on the time embeddings before they are passed to the rest of
            the UNet. Choose from `silu`, `mish`, `gelu`, and `swish`.
        timestep_post_act (`str`, *optional*, defaults to `None`):
            The second activation function to use in timestep embedding. Choose from `silu`, `mish` and `gelu`.
        time_cond_proj_dim (`int`, *optional*, defaults to `None`):
            The dimension of `cond_proj` layer in the timestep embedding.
        conv_in_kernel (`int`, *optional*, default to `3`): The kernel size of `conv_in` layer.
        conv_out_kernel (`int`, *optional*, default to `3`): The kernel size of `conv_out` layer.
        projection_class_embeddings_input_dim (`int`, *optional*): The dimension of the `class_labels` input when
            `class_embed_type="projection"`. Required when `class_embed_type="projection"`.
        class_embeddings_concat (`bool`, *optional*, defaults to `False`): Whether to concatenate the time
            embeddings with the class embeddings.
        mid_block_only_cross_attention (`bool`, *optional*, defaults to `None`):
            Whether to use cross attention with the mid block when using the `UNetMidBlock2DSimpleCrossAttn`. If
            `only_cross_attention` is given as a single boolean and `mid_block_only_cross_attention` is `None`, the
            `only_cross_attention` value is used as the value for `mid_block_only_cross_attention`. Default to `False`
            otherwise.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D", "CrossAttnUpBlock2D"]

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads: int = 64,
    ):
        super().__init__()

        self.sample_size = sample_size

        if num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19."
            )

        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = num_attention_heads or attention_head_dim

        # Check inputs
        self._check_config(
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
        )

        # input
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        time_embed_dim, timestep_input_dim = self._set_time_proj(
            time_embedding_type,
            block_out_channels=block_out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embedding_dim=time_embedding_dim,
        )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        self._set_encoder_hid_proj(
            encoder_hid_dim_type,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
        )

        # class embedding
        self._set_class_embedding(
            class_embed_type,
            act_fn=act_fn,
            num_class_embeds=num_class_embeds,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            time_embed_dim=time_embed_dim,
            timestep_input_dim=timestep_input_dim,
        )

        self._set_add_embedding(
            addition_embed_type,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
            addition_time_embed_dim=addition_time_embed_dim,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            time_embed_dim=time_embed_dim,
        )

        if time_embedding_act_fn is None:
            self.time_embed_act = None
        else:
            self.time_embed_act = get_activation(time_embedding_act_fn)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention

            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        if class_embeddings_concat:
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            
            
       
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = get_mid_block(
            mid_block_type,
            temb_channels=blocks_time_embed_dim,
            in_channels=block_out_channels[-1],
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            output_scale_factor=mid_block_scale_factor,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim[-1],
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            resnet_skip_time_act=resnet_skip_time_act,
            cross_attention_norm=cross_attention_norm,
            attention_head_dim=attention_head_dim[-1],
            dropout=dropout,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))
            if reverse_transformer_layers_per_block is None
            else reverse_transformer_layers_per_block
        )
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resolution_idx=i,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )

            self.conv_act = get_activation(act_fn)

        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        )

        self._set_pos_net_if_use_gligen(attention_type=attention_type, cross_attention_dim=cross_attention_dim)

    def _check_config(
        self,
        down_block_types: Tuple[str],
        up_block_types: Tuple[str],
        only_cross_attention: Union[bool, Tuple[bool]],
        block_out_channels: Tuple[int],
        layers_per_block: Union[int, Tuple[int]],
        cross_attention_dim: Union[int, Tuple[int]],
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple[int]]],
        reverse_transformer_layers_per_block: bool,
        attention_head_dim: int,
        num_attention_heads: Optional[Union[int, Tuple[int]]],
    ):
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )
        if isinstance(transformer_layers_per_block, list) and reverse_transformer_layers_per_block is None:
            for layer_number_per_block in transformer_layers_per_block:
                if isinstance(layer_number_per_block, list):
                    raise ValueError("Must provide 'reverse_transformer_layers_per_block` if using asymmetrical UNet.")

    def _set_time_proj(
        self,
        time_embedding_type: str,
        block_out_channels: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        return time_embed_dim, timestep_input_dim

    def _set_encoder_hid_proj(
        self,
        encoder_hid_dim_type: Optional[str],
        cross_attention_dim: Union[int, Tuple[int]],
        encoder_hid_dim: Optional[int],
    ):
        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
            )

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image_proj"` (Kandinsky 2.1)`
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2
            self.encoder_hid_proj = ImageProjection(
                image_embed_dim=encoder_hid_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
            )
        else:
            self.encoder_hid_proj = None

    def _set_class_embedding(
        self,
        class_embed_type: Optional[str],
        act_fn: str,
        num_class_embeds: Optional[int],
        projection_class_embeddings_input_dim: Optional[int],
        time_embed_dim: int,
        timestep_input_dim: int,
    ):
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif class_embed_type == "simple_projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'simple_projection' requires `projection_class_embeddings_input_dim` be set"
                )
            self.class_embedding = nn.Linear(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

    def _set_add_embedding(
        self,
        addition_embed_type: str,
        addition_embed_type_num_heads: int,
        addition_time_embed_dim: Optional[int],
        flip_sin_to_cos: bool,
        freq_shift: float,
        cross_attention_dim: Optional[int],
        encoder_hid_dim: Optional[int],
        projection_class_embeddings_input_dim: Optional[int],
        time_embed_dim: int,
    ):
        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        elif addition_embed_type == "text_image":
            # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image"` (Kandinsky 2.1)`
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif addition_embed_type == "image":
            # Kandinsky 2.2
            self.add_embedding = ImageTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type == "image_hint":
            # Kandinsky 2.2 ControlNet
            self.add_embedding = ImageHintTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type is not None:
            raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")

    def _set_pos_net_if_use_gligen(self, attention_type: str, cross_attention_dim: int):
        if attention_type in ["gated", "gated-text-image"]:
            positive_len = 768
            if isinstance(cross_attention_dim, int):
                positive_len = cross_attention_dim
            elif isinstance(cross_attention_dim, (list, tuple)):
                positive_len = cross_attention_dim[0]

            feature_type = "text-only" if attention_type == "gated" else "text-image"
            self.position_net = GLIGENTextBoundingboxProjection(
                positive_len=positive_len, out_dim=cross_attention_dim, feature_type=feature_type
            )

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def set_attention_slice(self, slice_size: Union[str, int, List[int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism from https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stage blocks where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of values that
        are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        for i, upsample_block in enumerate(self.up_blocks):
            setattr(upsample_block, "s1", s1)
            setattr(upsample_block, "s2", s2)
            setattr(upsample_block, "b1", b1)
            setattr(upsample_block, "b2", b2)

    def disable_freeu(self):
        """Disables the FreeU mechanism."""
        freeu_keys = {"s1", "s2", "b1", "b2"}
        for i, upsample_block in enumerate(self.up_blocks):
            for k in freeu_keys:
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    setattr(upsample_block, k, None)

    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedAttnProcessor2_0())

    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def get_time_embed(
        self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]
    ) -> Optional[torch.Tensor]:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)
        return t_emb

    def get_class_embed(self, sample: torch.Tensor, class_labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        class_emb = None
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
        return class_emb

    def get_aug_embed(
        self, emb: torch.Tensor, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        aug_emb = None
        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            # SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb = self.add_embedding(image_embs, hint)
        return aug_emb

    def process_encoder_hidden_states(
        self, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            if hasattr(self, "text_encoder_hid_proj") and self.text_encoder_hid_proj is not None:
                encoder_hidden_states = self.text_encoder_hid_proj(encoder_hidden_states)

            image_embeds = added_cond_kwargs.get("image_embeds")
            image_embeds = self.encoder_hid_proj(image_embeds)
            encoder_hidden_states = (encoder_hidden_states, image_embeds)
        return encoder_hidden_states
    
############################### transforms here #######################################
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers
        #print(timestep)
        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None
        


        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        
        ##### Transforms  #####################################################################################################################
        # if timestep == 1:
        
        # timestep_hack = torch.tensor([999, 679, 359, 99], device='cuda:0')
        # if (timestep == timestep_hack).all():
        #     sample = inversion(sample, r=0.15)

        if rotate_z_value != 0:
            sample = rotate_z(sample, rotate_z_value)
            print("Rotate Z Value: ", rotate_z_value)

                    
        if rotate_x_value != 0:
            sample = rotate_x(sample, rotate_x_value)
            print("Rotate X Value: ", rotate_x_value)

                    
        if rotate_y_value != 0:
            sample = rotate_y(sample, rotate_y_value)
            print("Rotate Y Value: ", rotate_y_value)

            
        # sample = rotate_X(sample)
        #print(sample.size())

        # 2. pre-process
        sample = self.conv_in(sample)
        

    
        # sample = rotate_z(sample, 0.2)
        
        
        
    #    # sample = torch.squeeze(sample, 0) ## potential to add transformation here
        #print(sample.size())

    #     #sample = torch.unsqueeze(sample, 0)
        
        


        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                       and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                       for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True
        # a = torch.ones_like(emb).to('cuda')
        # emb = torch.cat((a[:,:10], emb[:,10:]),dim=1)
        ## this looping through unet downsample// 
        down_block_res_samples = (sample,)
        for idx, downsample_block in enumerate(self.down_blocks):
            #print(self.down_blocks)
            #if idx == 2:
           # print(idx)
                #print(downsample_block)
            #print(f'idx: {idx}, shape: {emb[:,0]}')

            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)
                    

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)
            # print(sample.shape)
            # print(idx)
            #print(sample[1, :])
            
            
            


            
            
            

            
      ######################################################################################################################      
#############################################################################################################################            
 ######  Conditional to apply network bending to layer of DOWNSAMPLE UNet idx = LAYER ################################################
            if DownSampleBend == 1: 
                    
                #print("DownSample On")
                if idx == Layer:    
                    #print(Layer)
                    
                    
                    if Inversion_OSC_Radius != 0:
                        sample = inversion(sample, Inversion_OSC_Radius)
                       # print("Inversion Value: ", Inversion_OSC_Radius)
                        #print("Layer: ", Layer)
                    
                    if  add_full_Value != 0:
                        ADDFULL = add_full(add_full_Value)
                        sample = ADDFULL(sample)
                        #print("Add full value: ", add_full_Value)
                        #print("Layer: ", Layer)
                    
                    if add_sparse_value != 0:
                        ADDSparse = add_sparse(add_sparse_value)
                        sample = ADDSparse(sample)
                        #print("Sparse Value: ", add_sparse_value)
                        #print("Layer: ", Layer)
                    
 
                    if add_noise_value != 0:
                        ADDNoise= add_noise(add_noise_value)
                        sample = ADDNoise(sample)
                       # print("Noise Value: ", add_noise_value)
                        #print("Layer: ", Layer)

                    if subtract_full_value != 0:
                        SUBTRACTFULLVALUE= subtract_full(subtract_full_value)
                        sample = SUBTRACTFULLVALUE(sample)
                       # print("Subtract Value: ", subtract_full_value)
                        #print("Layer: ", Layer)
                    
                    if thresh_value != 0:
                        sample = thresh(sample, thresh_value)
                       # print("Thresh Value: ", thresh_value)
                        #print("Layer: ", Layer)
                    
                # if soft_threshold_value != 0:
                #     sample = soft_threshold(sample, soft_threshold_value)
                #     print(soft_threshold_value)
                    
                    
                # if soft_threshold2_value != 0:
                #     sample = soft_threshold2(sample, soft_threshold2_value)
                #     print(soft_threshold2_value)
                    
                    if add_dim_value != 0:
                        sample = add_dim(add_dim_value, sample, dimension, indexx)
                      #  print(" Add Dim Value: ", add_dim_value)
                      #  print("Dim ", dim)
                     #   print("Index: ", indexx)
                        #print("Layer: ", Layer)
                    
                    if add_rand_cols_value != 0:
                        sample = add_rand_cols(sample, add_rand_cols_value, k_value)
                       # print("Add Rand Cols Value: ", add_rand_cols_value)
                      #  print("K Value: ", k_value)
                        #print("Layer: ", Layer)
                    
                    if add_rand_rows_value != 0:
                        sample = add_rand_rows(sample, add_rand_rows_value, k_value)
                       # print("Add Rand Rows Value: ", add_rand_rows_value)
                       # print("K Value: ", k_value)
                        #print("Layer: ", Layer)
                    
                    if invert_dim_value != 0:
                        sample = invert_dim(sample, invert_dim_value, dimension, indexx)
                      #  print("Invert Dim Value: ", invert_dim_value)
                      #  print("Dim ", dim)
                      #  print("Index: ", indexx)
                        #print("Layer: ", Layer)
                    
                # if add_normal_value != 0:
                #     sample = add_normal(sample, add_normal_value)
                #     print(add_normal_value)
                    

                    
                    if reflect_value != 0:
                        sample = reflect(sample, reflect_value)
                      #  print(reflect_value)
                      #  print("Layer: ", Layer)
                    
                    if hadamard1_value != 0:
                        sample = hadamard1(sample, hadamard1_value)
                     #   print("hadamard1: ", hadamard1_value)
                     #   print("Layer: ", Layer)
                
                # if hadamard2_value != 0:
                #     sample = hadamard2(sample, hadamard2_value)
                #     print(hadamard2_value)
                    
                    if dilation_value != 0:
                        sample = dilation(sample, dilation_value)
                      #  print("Dilation Value: ", dilation_value)
                      #  print("Layer: ", Layer)
                    
                    if erosion_value != 0:
                        sample = erosion(sample, erosion_value)
                     #   print("Erosion Value: ", erosion_value)
                     #   print("Layer: ", Layer)
                    

                    #sample = sobel(sample, 100)
                    #sample = top_Hat(sample, 1)


                    

                    
        

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples
        


        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)
                
            #print(self.mid_block)
                
            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)
    
        if is_controlnet:
            sample = sample + mid_block_additional_residual
        #print(sample.shape)
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            
      ######################################################################################################################      
#############################################################################################################################            
 ######  Conditional to apply network bending to layer of UPSAMPLE UNet idx = LAYER ################################################
            #print(self.up_blocks)
            # print(sample.shape)
            # print(i)
            
            if UpSampleBend == 1:
               # print("UpSampleBend On")
                if i == upidx:    
                

                    if Inversion_OSC_Radius != 0:
                        sample = inversion(sample, Inversion_OSC_Radius)
                       # print("Inversion Value: ", Inversion_OSC_Radius)
                        #print("Layer: ", upidx)
                    
                    if  add_full_Value != 0:
                        ADDFULL = add_full(add_full_Value)
                        sample = ADDFULL(sample)
                      #  print("Add full value: ", add_full_Value)
                      #  print("Layer: ", upidx)
                    
                    if add_sparse_value != 0:
                        ADDSparse = add_sparse(add_sparse_value)
                        sample = ADDSparse(sample)
                       # print("Sparse Value: ", add_sparse_value)
                      #  print("Layer: ", upidx)
                    
 
                    if add_noise_value != 0:
                        ADDNoise= add_sparse(add_noise_value)
                        sample = ADDNoise(sample)
                       # print("Noise Value: ", add_noise_value)
                      #  print("Layer: ", upidx)

                    if subtract_full_value != 0:
                        SUBTRACTFULLVALUE= add_sparse(subtract_full_value)
                        sample = SUBTRACTFULLVALUE(sample)
                     #   print("Subtract Value: ", subtract_full_value)
                      #  print("Layer: ", upidx)
                    
                    if thresh_value != 0:
                        sample = thresh(sample, thresh_value)
                    #    print("Thresh Value: ", thresh_value)
                     #   print("Layer: ", upidx)
                    
                # if soft_threshold_value != 0:
                #     sample = soft_threshold(sample, soft_threshold_value)
                #     print(soft_threshold_value)
                    
                    
                # if soft_threshold2_value != 0:
                #     sample = soft_threshold2(sample, soft_threshold2_value)
                #     print(soft_threshold2_value)
                    
                    if add_dim_value != 0:
                        sample = add_dim(add_dim_value, sample, dimension, indexx)
                      #  print(" Add Dim Value: ", add_dim_value)
                      #  print("Dim ", dim)
                      #  print("Index: ", indexx)
                      #  print("Layer: ", upidx)
                    
                    if add_rand_cols_value != 0:
                        sample = add_rand_cols(sample, add_rand_cols_value, k_value)
                      #  print("Add Rand Cols Value: ", add_rand_cols_value)
                      #  print("K Value: ", k_value)
                     #   print("Layer: ", upidx)
                    
                    if add_rand_rows_value != 0:
                        sample = add_rand_rows(sample, add_rand_rows_value, k_value)
                      #  print("Add Rand Rows Value: ", add_rand_rows_value)
                      #  print("K Value: ", k_value)
                     #   print("Layer: ", upidx)
                    
                    if invert_dim_value != 0:
                        sample = invert_dim(sample, invert_dim_value, dimension, indexx)
                       # print("Invert Dim Value: ", invert_dim_value)
                      #  print("Dim ", dim)
                     #   print("Index: ", indexx)
                     #   print("Layer: ", upidx)
                    
                # if add_normal_value != 0:
                #     sample = add_normal(sample, add_normal_value)
                #     print(add_normal_value)
                    
                    if rotate_z_value != 0:
                        sample = rotate_z(sample, rotate_z_value)
                     #   print("Rotate Z Value: ", rotate_z_value)
                      #  print("Layer: ", upidx)
                    
                    if rotate_x_value != 0:
                        sample = rotate_x(sample, rotate_x_value)
                       # print("Rotate X Value: ", rotate_x_value)
                      #  print("Layer: ", upidx)
                    
                    if rotate_y_value != 0:
                        sample = rotate_y(sample, rotate_y_value)
                       # print("Rotate Y Value: ", rotate_y_value)
                      #  print("Layer: ", upidx)
                    
                    if reflect_value != 0:
                        sample = reflect(sample, reflect_value)
                     #   print(reflect_value)
                     #   print("Layer: ", upidx)
                    
                    if hadamard1_value != 0:
                        sample = hadamard1(sample, hadamard1_value)
                      #  print("hadamard1: ", hadamard1_value)
                      #  print("Layer: ", upidx)
                

                    
                    if dilation_value != 0:
                        sample = dilation(sample, dilation_value)
                      #  print("Dilation Value: ", dilation_value)
                      #  print("Layer: ", upidx)
                    
                    if erosion_value != 0:
                        sample = erosion(sample, erosion_value)
                     #   print("Erosion Value: ", erosion_value)
                     #   print("Layer: ", upidx)
                    


            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
        #print(sample.shape)
        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        
        #print(sample)
        #print(sample.shape)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)
        
        

        return UNet2DConditionOutput(sample=sample)








    
###### Transformations #################################

#
#
#
#       all taken directly from https://github.com/dzluke/DAFX2024/blob/main/util.py
#
#
#
#
#
    
    
def add_full(r):
    """
    Return a fn that takes in a latent tensor and returns a tensor of the same shape, but with the value r
    added to every element
    """
    return lambda x: x + (torch.ones_like(x) * r)


def add_sparse(r):
    """
    Return a fn that takes in a latent tensor and returns a tensor of the same shape, but with the value r
    added to 25% of the elements
    """
    return lambda x: x + ((torch.rand_like(x) < 0.05) * r)


def add_noise(r):
    """
    Return a fn that adds Gaussian noise mutliplied by r to x
    """

    return lambda x: x + (torch.randn_like(x) * r)



def subtract_full(r):
    """
    Return a fn that takes in a latent tensor and returns a tensor of the same shape, but with the value r
    subtracted from every element
    """
    return lambda x: x - (torch.ones_like(x) * r)



def thresh(x, r):
    device = x.get_device()
    x = x.cpu()
    x = x.apply_(lambda y: y if abs(y) >= r else 0)
    x = x.to(device)
    return x



    """
    Return a fn that applies soft thresholding to x
    In soft thresholding, values less than r are set to 0 and values greater than r are shrunk towards zero
    source: https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html
    """
def soft_threshold(x, r):
    x = x / x.abs() * torch.maximum(x.abs() - r, torch.zeros_like(x))
    return x



    """
    Return a fn that applies soft thresholding to x
    In soft thresholding, values less than r are set to 0 and values greater than r are shrunk towards zero
    source: https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html
    """
def soft_threshold2(x, r):
    device = x.get_device()
    x = x.cpu()
    x = x.apply_(lambda y: 0 if abs(y) < r else y*(1-r))
    x = x.to(device)
    return x




def inversion(x, r):
    device = x.get_device()
    x = x.cpu()
    x = x.apply_(lambda y: 1./r - y)
    x = x.to(device)
    return x



def inversion2(x):
    device = x.get_device()
    x = x.cpu()
    x = x.apply_(lambda x: -1 * x)
    x = x.to(device)
    return x


def log(r):
    return lambda x: torch.log(x)


def power(r):
    return lambda x: torch.pow(x, r)


def add_dim(r, x, dim, i):
    
    """
    Add value r to the given dim at index i
    dim = 0 means adding in "z" dimension (shape = 4)
    dim= 1 means adding to row i
    dim = 2 means adding to column i
    @return: modified x
    """
    if dim == 0:
        for a in range(x.shape[0]):
            x[a, i, i] += r
    elif dim == 1:
            x[:, i:i + 1] += r
    elif dim == 2:
            x[:, :, i:i + 1] += r
    else:
        raise NotImplementedError(f"Cannot apply to dimension {dim}")
    return x

 


    """
    Return a fn that will add value 'r' to a fraction of the cols of a tensor
    Assume x is a 3-tensor and rows refer to the third dimension
    k should be between 0 and 1
    """
def add_rand_cols(x, r, k):
    dim = x.shape[1]
    cols = random.sample(range(dim), int(k * dim))
    for col in cols:
        x[:, :, col:col + 1] += r
    return x





"""
Return a fn that will add value 'r' to a fraction of the rows of a tensor
Assume x is a 3-tensor and rows refer to the second dimension
k should be between 0 and 1
"""
def add_rand_rows(x, r, k):
    dim = x.shape[2]
    rows = random.sample(range(dim), int(k * dim))
    for row in rows:
        x[:, row:row + 1] += r
    return x



def invert_dim(x, r, dim, i):
        """
        Apply inversion (1/r. - x) at the given dim at index i
        dim = 0 means applying in "z" dimension (shape = 4)
        dim = 1 means applying to row i
        dim = 2 means applying to column i
        @return: modified x
        """
        invert = lambda val: inversion(val, r)
        #invert = inversion(x, r)
        if dim == 0:
            for a in range(x.shape[0]):
                x[a, i, i] = invert(x[a, i, i])
        elif dim == 1:
            x[:, i:i + 1] = invert(x[:, i:i + 1])
        elif dim == 2:
            x[:, :, i:i + 1] += invert(x[:, :, i:i + 1])
        else:
            raise NotImplementedError(f"Cannot apply to dimension {dim}")
        return x





def apply_to_dim(x, func, r, dim, i):
        """
        Apply func at the given dim at i
        dim = 0 means applying in "z" dimension (shape = 4)
        dim = 1 means applying to row i
        dim = 2 means applying to column i
        @return: modified x
        """
        fn = func(r)
        if dim == 0:
            for a in range(x.shape[0]):
                try:
                    x[a, i[0], i[1]] = fn(x[a, i[0], i[1]])
                except TypeError:
                    x[a, i, i] = fn(x[a, i, i])
        elif dim == 1:
            try:
                x[:, i[0]:i[1]] = fn(x[:, i[0]:i[1]])
            except TypeError:
                x[:, i:i + 1] = fn(x[:, i:i + 1])
        elif dim == 2:
            try:
                x[:, :, i[0]:i[1]] = fn(x[:, :, i[0]:i[1]])
            except TypeError:
                x[:, :, i:i + 1] = fn(x[:, :, i:i + 1])
        else:
            raise NotImplementedError(f"Cannot apply to dimension {dim}")
        return x





"""
return a function that applies the given function a random fraction of the elements, as determined by 'sparsity'
0 < sparsity < 1
"""
def apply_sparse(x, func, sparsity):
    mask = torch.rand_like(x) < sparsity
    x = (x * ~mask) + (func(x) * mask)
    return x




    """
    Add a 2D normal gaussian (bell curve) to the center of the tensor
    """
def add_normal(x, r):
    # chatgpt wrote this
    # Define the size of the matrix
    size = 64

    # Generate grid coordinates centered at (0,0)
    a = np.linspace(-5, 5, size)
    b = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(a, b)

    # Standard deviation
    sigma = 1.5  # You can adjust this value as desired

    # Generate a 2D Gaussian distribution with peak at the center and specified standard deviation
    Z = np.exp(-0.5 * ((X / sigma) ** 2 + (Y / sigma) ** 2)) / (2 * np.pi * sigma ** 2)
    Z *= r
    Z = torch.from_numpy(Z).to(x.get_device())

    for i in range(x.shape[0]):
        # Assuming Z has more channels than x[i], you can slice Z
       # Z = Z[:, :x.shape[1], :, :]  # Take only the first x.shape[1] channels from Z

# Or, if Z has fewer channels, you can repeat it
        Z = Z.repeat(1, x.shape[1] // Z.shape[1], 1, 1)  # Repeat Z's channels to match x

        x[i] += Z
    return x




    """
    Return a fn that computes the matrix exponential of a given tensor
    """
def tensor_exp(x, r):
    device = x.get_device().float16()
    x = x.cpu().numpy()
    x = scipy.linalg.expm(x)
    x = torch.from_numpy(x).to(device)
    return x




    """
    Return a fn that rotates a 3-tensor by r radians
    Rotates along "z" axis
    """
def rotate_z(x, r):
    
    #x = torch.squeeze(x, 1)
    device = x.get_device()
    c = math.cos(r)
    s = math.sin(r)
    rotation_matrix = [
        [c, -1 * s, 0, 0],
        [s,    c,   0, 0],
        [0,    0,   1, 0],
        [0,    0,   0, 1]
    ]
    

    op = torch.tensor(rotation_matrix).to(device)
    op = op.to(dtype=x.dtype)
    x = torch.tensordot(op, x, dims=1)
   # x = torch.unsqueeze(x, 1)
    return x




    """
    Return a fn that rotates a 3-tensor by r radians
    Rotates along "x" axis
    """
def rotate_x(x, r):

    device = x.get_device()
    print(x)
    c = math.cos(r)
    s = math.sin(r)
    rotation_matrix = [
        [1, 0,   0,    0],
        [0, c, -1 * s, 0],
        [0, s,   c,    0],
        [0, 0,   0,    1]
    ]
    op = torch.tensor(rotation_matrix).to(device)
    op = op.to(dtype=x.dtype)
    x = torch.tensordot(op, x, dims=1)

    return x




    """
    Return a fn that rotates a 3-tensor by r radians
    Rotates along "y" axis
    """
def rotate_y(x, r):

    device = x.get_device()
    c = math.cos(r)
    s = math.sin(r)
    rotation_matrix = [
        [c,      0, s, 0],
        [0,      1, 0, 0],
        [-1 * s, 0, c, 0],
        [0,      0, 0, 1]
    ]
    op = torch.tensor(rotation_matrix).to(device)
    op = op.to(dtype=x.dtype)
    x = torch.tensordot(op, x, dims=1)

    return x




    """
    Return a fn that rotates a 3-tensor by r radians
    Rotates along "y" axis
    """
def rotate_y2(x, r):
    device = x.get_device()
    c = math.cos(r)
    s = math.sin(r)
    rotation_matrix = [
        [c,      0, 0, s],
        [0,      1, 0, 0],
        [0,      0, 1, 0],
        [-1 * s, 0, 0, c]
    ]
    op = torch.tensor(rotation_matrix).to(device)
    x = torch.tensordot(op, x, dims=1)
    return x




    """
    Return a fn that reflects across the given dimension r
    r can be 0, 1, 2, or 3
    """
def reflect(x, r):
    device = x.get_device()
    op = torch.eye(4)  # identity matrix
    op[r, r] *= -1
    op = op.to(device)
    op = op.to(dtype=x.dtype)
    x = torch.tensordot(op, x, dims=1)
    return x




def hadamard1(x, r):
    device = x.get_device()
    h = scipy.linalg.hadamard(4)
    op = torch.tensor(h).to(torch.float16).to(device)
    op = op.to(dtype=x.dtype)
    x = torch.tensordot(op, x, dims=1)
    return x



# def hadamard2(x, r):
#     device = x.get_device()
#     h = scipy.linalg.hadamard(64)
#     op = torch.tensor(h).to(torch.float16).to(device)
#     op = op.to(dtype=x.dtype)
#     x = torch.tensordot(x, op, dims=[[1], [1]])
#     return x

def hadamard2(x, r):
    h = scipy.linalg.hadamard(r) # Define your hadamard2 operation here
    device = x.get_device()
    # Convert `h` to a tensor
    op = torch.tensor(h).to(torch.float16).to(device)
    
    # Ensure that `op` has the same dtype and size as `x` along the contracting dimension
    op = op.to(dtype=x.dtype)
    
    if op.shape[1] != x.shape[1]:
        # If sizes don't match, adjust op's size
        if op.shape[1] < x.shape[1]:
            op = op.repeat(1, x.shape[1] // op.shape[1], 1, 1)
        else:
            op = op[:, :x.shape[1], :, :]  # Slice if op has more elements
    
    # Now perform tensordot
    x = torch.tensordot(x, op, dims=[[1], [1]])
    
    return x




def apply_both(x, fn1, fn2, r):
    return fn1(fn2(r))





    """
    First apply func to the latent tensor, then normalize the result
    """
def normalize(x, func):
    x = func(x)  # first apply the network bending function
    # then normalize the result
    max = x.abs().max()
    x = x / max
    return x




    """
    First apply func to the latent tensor, then normalize the result
    """
def normalize2(x, func):
    x = func(x)  # first apply the network bending function
    # then normalize the result
    x = torch.nn.functional.normalize(x, dim=0)
    return x




    """
        First apply func to the latent tensor, then normalize the result
    """
def normalize3(x, func):
    x = func(x)
    x = x - x.mean()
    return x




    """
        First apply func to the latent tensor, then normalize the result
    """
def normalize4(x, func, dim=0):
    x = func(x)
    x = x - torch.mean(x, dim=dim, keepdim=True)
    return x
   



def gradient(x, r):

    kernel = torch.ones(r, r).to(x.get_device()).to(dtype=x.dtype)
    x = morphology.gradient(x, kernel)

    return x




def dilation(x, r):
   # x = x.unsqueeze(0)
   
    kernel1 = torch.ones(r, r).to(x.get_device()).to(dtype=x.dtype)

    x = morphology.dilation(x, kernel1)
   # x = x.squeeze(0)
    return x




def erosion(x, r):
    #x = x.unsqueeze(0)
    kernel = torch.ones(r, r).to(x.get_device()).to(dtype=x.dtype)
    x = morphology.erosion(x, kernel)
    #x = x.squeeze(0)
    return x




def sobel(x, r=True):
    #x = x.unsqueeze(0)
    x = filters.sobel(x, normalized=r).to(dtype=x.dtype)
    #x = x.squeeze(0)
    return x

def top_Hat(x, r):
    kernel = torch.ones(r, r).to(x.get_device()).to(dtype=x.dtype)
    x = morphology.top_hat(x, kernel)
    return x





    """
    Return a fn that computes the absolute value of a tensor
    """
def absolute(x, r):
    device = x.get_device()
    x = x.cpu()
    x = x.apply_(lambda y: abs(y))
    x = x.to(device)
    return x




    """
    Return a fn that computes the log of a tensor with base r. Must first ensure that it is non-negative
    """
def log(x, r=math.e):
    device = x.get_device() if x.get_device() is not None else 'cpu'
    x = x.cpu()
    x = x.apply_(lambda y: abs(y))
    x = x.to(device)
    return torch.log(x) / math.log(r)



    """
    Return a fn that clamps a tensor between min and max
    """
def clamp(x, r):
    min, max = r
    device = x.get_device()
    x = x.cpu()
    x = x.apply_(lambda y: min if y < min else max if y > max else y)
    x = x.to(device)
    return x



    """
    Return a fn that scales a tensor between min and max based on the tensor's min and max
    """
def scale(x, r):
    min, max = r
    device = x.get_device()
    x = x.cpu()
    xmin = x.min()
    xmax = x.max()  
    x = x.apply_(lambda y: (y - xmin) / (xmax - xmin) *(max - min) + min)
    x = x.to(device)
    return x

