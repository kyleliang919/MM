import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init="xavier"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.ttt_weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.ttt_bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('ttt_bias', None)

        # Initialize weights
        self.reset_parameters(init)
        self.delta_w = None
        self.time_dim = 1
        self.ttt = False

    def reset_parameters(self, init):
        if init == "xavier":
            nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.ttt_weight)
        elif init == "kaiming":
            nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
            nn.init.kaiming_uniform_(self.ttt_weight, nonlinearity='linear')
        else:
            nn.init.normal_(self.weight, std=0.02)
            nn.init.normal_(self.ttt_weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            nn.init.zeros_(self.ttt_bias)

    def forward(self, x):
        if self.training:
            if self.delta_w is None:
                x, ttt_x = torch.chunk(x, 2, dim =self.time_dim)
                out = F.linear(x, self.weight, self.bias)
                ttt_out = F.linear(x, self.ttt_weight, self.ttt_bias)
                self.delta_w = ttt_out.reshape(-1, self.out_features).T @ x.reshape(-1, self.in_features)
                out = torch.concat([out, ttt_out], dim = self.time_dim)
            else:
                out = F.linear(x, self.weight + 0.01 * self.delta_w, self.bias)
                self.delta_w = None
        else:
            out = F.linear(x, self.weight, self.bias)
        return out

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

if __name__=="__main__":
    model = Linear(64, 64).cuda()
    x = torch.randn(2, 128, 64).cuda()
    y = torch.randn(2, 128, 64).cuda()
    out1 = model(x)
    out2 = model(torch.cat([x, y], dim = 1))
    out3 = model(x)
    out3.sum().backward()
    ttt_grad = model.weight.grad.clone()
    print(model.weight.grad)
    print(out3)

    