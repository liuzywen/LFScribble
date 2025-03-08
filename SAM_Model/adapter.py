import torch
import torch.nn as nn
from models.encoders.vmamba import SSM, FocalMambaBlock

class Adapter(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout=0.0,
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.input_dim)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.input_dim, self.output_dim)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.output_dim, self.input_dim)

        self.dropout = dropout
        # self.ssm = FocalMambaBlock(
        #     d_model=128,
        # )


    def forward(self, x):
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        #torch.Size([b, 16, 16, 1024])
        down = self.down_proj(x)
        # mamba
        # down = self.ssm(down)

        #torch.Size([24, 16, 16, 128])
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        output = up * self.scale

        if self.adapter_layernorm_option == 'out':
            output = self.adapter_layer_norm_before(output)

        return output


