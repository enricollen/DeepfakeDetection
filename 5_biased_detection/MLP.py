import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for i, (dim, dropout) in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            if i != len(hidden_layers) - 1:  # dropout except for the last layer
                layers.append(nn.Dropout(p=dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

"""import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)"""