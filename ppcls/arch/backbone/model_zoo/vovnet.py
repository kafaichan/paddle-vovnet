# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# reference: https://arxiv.org/abs/1611.05431

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm2D, Linear, ReLU
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D
from paddle.nn.initializer import KaimingNormal, Constant, Uniform

import math
#from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url


MODEL_URLS = {
    "VoVNet57": "",
	"VoVNet39": ""
}

__all__ = list(MODEL_URLS.keys())


class ConvBNLayer(nn.Layer):
    def __init__(self,
                input_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                name='conv_bn'):
        super(ConvBNLayer, self).__init__()

        self.conv2d = Conv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(name='{}/conv'.format(name), initializer=KaimingNormal(nonlinearity='leaky_relu')),
            bias_attr=False)
        self.batch_norm =BatchNorm2D(
            output_channels,
            momentum=0.1,
            weight_attr=ParamAttr(name='{}/norm_weight'.format(name), initializer=Constant(value=1.0)),
            bias_attr=ParamAttr(name='{}/norm_bias'.format(name), initializer=Constant(value=0.0)))
        self.relu = ReLU(name='{}/relu'.format(name))

    def forward(self, inputs):
        x = self.conv2d(inputs)
        x = self.batch_norm(x)
        return self.relu(x)


class OneShotAggLayer(nn.Layer):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 layer_per_block,
                 name,
                 identity=False):
        super(OneShotAggLayer, self).__init__()
        self.identity = identity
        layers = []
        in_channel = in_ch
        for i in range(layer_per_block):
            layers.append(ConvBNLayer(in_channel, stage_ch, 3, name='{}_{}'.format(name, i)))
            in_channel = stage_ch
        self.layers = nn.LayerList(layers)

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = ConvBNLayer(in_channel, concat_ch, 1, name='{}_{}'.format(name, 'concat'))

    def forward(self, inputs):
        identity_feat = inputs
        output = []
        output.append(inputs)

        for layer in self.layers:
            inputs = layer(inputs)
            output.append(inputs)

        x = paddle.concat(output, axis=1)

        xt = self.concat(x)

        if self.identity:
            xt = xt + identity_feat
        return xt


class OneShotAggBlock(nn.Layer):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 block_per_stage,
                 layer_per_block,
                 stage_num):
        super(OneShotAggBlock, self).__init__()
        layers = []
        if not stage_num == 2:
            layers.append(MaxPool2D(kernel_size=3, stride=2, ceil_mode=True))

        layers.append(OneShotAggLayer(in_ch, stage_ch, concat_ch, layer_per_block, 'osa{}_1'.format(stage_num)))
        for i in range(block_per_stage-1):
            layers.append(OneShotAggLayer(concat_ch, stage_ch, concat_ch, layer_per_block, 'osa{}_{}'.format(stage_num, i+2), identity=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.layers(inputs)


class VoVNet(nn.Layer):
    def __init__(self,
                 stage_ch,
                 concat_ch,
                 block_per_stage,
                 layer_per_block,
                 class_num=1000):
        super(VoVNet, self).__init__()
        self.stem = nn.Sequential(*[
            ConvBNLayer(3, 64, stride=2, name='stem_1'),
            ConvBNLayer(64, 64, stride=1, name='stem_2'),
            ConvBNLayer(64, 128, stride=2, name='stem_3')
        ])

        stem_out_ch = [128]
        in_ch_list = stem_out_ch + concat_ch[:-1]
        self.osa_stage = nn.Sequential(*[
            OneShotAggBlock(in_ch_list[i], stage_ch[i], concat_ch[i], block_per_stage[i], layer_per_block, i+2)
        for i in range(4)])
        self._pool = AdaptiveAvgPool2D([1,1])

        stdv = 1.0 / math.sqrt(concat_ch[-1])
        self.classifier = Linear(
                            concat_ch[-1],
                            class_num,
                            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
                            bias_attr=ParamAttr(initializer=Constant(value=0.0)))

    def forward(self, inputs):
        x = self.stem(inputs)
        x = self.osa_stage(x)
        x = self._pool(x).flatten(start_axis=1)
        x = self.classifier(x)
        return x


#def _load_pretrained(pretrained, model, model_url, use_ssld=False):
#    if pretrained is False:
#        pass
#    elif pretrained is True:
#        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
#    elif isinstance(pretrained, str):
#        load_dygraph_pretrain(model, pretrained)
#    else:
#        raise RuntimeError(
#            "pretrained type is not available. Please use `string` or `boolean` type."
#        )


def VoVNet57(pretrained=False, use_ssld=False, **kwargs):
    model = VoVNet(stage_ch=[128, 160, 192, 224], concat_ch=[256, 512, 768, 1024], block_per_stage=[1,1,4,3], layer_per_block=5, **kwargs)
#    _load_pretrained(
#        pretrained, model, MODEL_URLS["VoVNet57"], use_ssld=use_ssld)
    return model


def VoVNet39(pretrained=False, use_ssld=False, **kwargs):
    model = VoVNet(stage_ch=[128, 160, 192, 224], concat_ch=[256, 512, 768, 1024], block_per_stage= [1,1,2,2], layer_per_block=5, **kwargs)
#    _load_pretrained(
#        pretrained, model, MODEL_URLS["VoVNet39"], use_ssld=use_ssld
#    )
    return model


net = VoVNet39()
print(net)
print(net(paddle.rand(shape=[2,3,244,244])))