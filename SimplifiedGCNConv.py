# the following code is just for demon the concept using a simplified implementation

import torch
from torch import nn

class SimplifiedGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # This is our learnable weight matrix W. It's just a standard Linear layer.
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [num_nodes, in_channels]
        # edge_index has shape [2, num_edges]


        # STEP 1: Feature Transformation (H' = H * W)
        x_transformed = self.linear(x) # Shape: [num_nodes, out_channels]
        
        # STEP 2a: Aggregate from neighbors ONLY
        # ... running the original loop to get the sum of neighbor features ...
        neighbor_features = torch.zeros_like(x_transformed)
        source_nodes, dest_nodes = edge_index[0], edge_index[1]
        for i in range(edge_index.shape[1]):
            source, dest = source_nodes[i], dest_nodes[i]
            neighbor_features[dest] += x_transformed[source]
        
        # STEP 2b: Combine with self-features (This is the corrected part)
        # The final output is the sum of the node's own transformed vector
        # and the aggregated vectors of its neighbors.
        final_output = neighbor_features + x_transformed
        
        # The normalization step would then be applied to this 'final_output'

return final_output
        
