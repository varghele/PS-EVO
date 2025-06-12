import torch
from torch_geometric.data import Data, Batch
from psevo.encoder.ml3 import GNNML3Model
from psevo.encoder.gin import GINEEncoder

# Simple compatibility test
def test_model_compatibility():
    # Create sample data
    x = torch.randn(10, 5)  # 10 nodes, 5 features
    edge_index = torch.randint(0, 10, (2, 20))  # 20 edges
    edge_attr = torch.randn(20, 3)  # 3 edge features

    batch = Batch.from_data_list([Data(x=x, edge_index=edge_index, edge_attr=edge_attr)])

    # Test GINE
    gine = GINEEncoder(dim_in=5, num_edge_type=3, dim_hidden=64, dim_out=32, t=4)
    gine_output = gine(batch)
    print(f"GINE output shape: {gine_output.shape}")

    # Test GNNML3
    gnnml3 = GNNML3Model(dim_in=5, dim_out=32, dim_hidden=64, num_layers=3, num_supports=5)
    gnnml3_output = gnnml3(batch)
    print(f"GNNML3 output shape: {gnnml3_output.shape}")

    return gine_output.shape == gnnml3_output.shape


compatible = test_model_compatibility()
print(f"Output shapes compatible: {compatible}")
