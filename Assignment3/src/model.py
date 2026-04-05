import torch.nn as nn

"""
NN object
"""
class Net(nn.Module):
    def __init__(self, input_dim:int, hidden_dims:list[int], dropout=0.0):
        """
        makes the layers
        input_dim: number of nodes in input layer.
        hidden_dims: Sizes of hidden layers.
        dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_layers = nn.ModuleList()

        prev_dim = input_dim

        for h in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, h))
            prev_dim = h
        
        self.output_layer = nn.Linear(prev_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        forward pass throiugh the layers
        x: Input tensor
        return layer of raw model output
        """
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
            if self.dropout.p > 0: x = self.dropout(x)
        
        x = self.output_layer(x)
        return x
    
    def __str__(self):
        """
        string representation of the NN layer sizes
        """
        layers = []
        layers.append(str(self.hidden_layers[0].in_features))

        for layer in self.hidden_layers:
            layers.append(str(layer.out_features))

        layers.append(str(self.output_layer.out_features))
        
        return " -> ".join(layers)