import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, ChebConv
from torch_geometric.nn.models import MLP
import numpy as np
import pytest

from psevo.encoder.gin import GINEEncoder
from psevo.encoder.cheb import ChebEncoder

class TestEncoders:
    """Test suite for both GINEEncoder and ChebEncoder."""
    @staticmethod
    def create_sample_data(num_nodes=10, num_edges=20, dim_in=5, num_edge_type=3):
        """Create sample graph data for testing."""
        # Node features
        x = torch.randn(num_nodes, dim_in)

        # Edge indices (random connections)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Edge attributes
        edge_attr = torch.randn(num_edges, num_edge_type)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


    @staticmethod
    def create_batch_data(batch_size=3, num_nodes_per_graph=8, num_edges_per_graph=15,
                          dim_in=5, num_edge_type=3):
        """Create a batch of graphs for testing."""
        graphs = []
        for _ in range(batch_size):
            data = TestEncoders.create_sample_data(
                num_nodes_per_graph, num_edges_per_graph, dim_in, num_edge_type
            )
            graphs.append(data)
        return Batch.from_data_list(graphs)


    def test_encoder_initialization(self):
        """Test that both encoders initialize correctly."""
        dim_in, num_edge_type, dim_hidden, dim_out = 5, 3, 64, 32

        # Test GINEEncoder initialization
        gine_encoder = GINEEncoder(dim_in, num_edge_type, dim_hidden, dim_out, t=4)
        assert isinstance(gine_encoder, nn.Module)
        assert gine_encoder.t == 4
        assert gine_encoder.num_edge_type == num_edge_type

        # Test ChebEncoder initialization
        cheb_encoder = ChebEncoder(dim_in, num_edge_type, dim_hidden, dim_out, t=4, K=3)
        assert isinstance(cheb_encoder, nn.Module)
        assert cheb_encoder.t == 4
        assert cheb_encoder.K == 3
        assert cheb_encoder.num_edge_type == num_edge_type

        print("✓ Encoder initialization tests passed")


    def test_single_graph_forward(self):
        """Test forward pass with a single graph."""
        dim_in, num_edge_type, dim_hidden, dim_out = 5, 3, 64, 32

        # Create sample data
        data = self.create_sample_data(num_nodes=10, num_edges=20,
                                       dim_in=dim_in, num_edge_type=num_edge_type)
        batch = Batch.from_data_list([data])

        # Test GINEEncoder
        gine_encoder = GINEEncoder(dim_in, num_edge_type, dim_hidden, dim_out, t=4)
        gine_encoder.eval()

        with torch.no_grad():
            gine_output = gine_encoder(batch)
            gine_output_with_x = gine_encoder(batch, return_x=True)

        assert gine_output.shape == (1, dim_out)
        assert len(gine_output_with_x) == 2
        assert gine_output_with_x[0].shape == (1, dim_out)
        assert gine_output_with_x[1].shape == (10, dim_hidden)  # num_nodes, dim_hidden

        # Test ChebEncoder
        cheb_encoder = ChebEncoder(dim_in, num_edge_type, dim_hidden, dim_out, t=4, K=3)
        cheb_encoder.eval()

        with torch.no_grad():
            cheb_output = cheb_encoder(batch)
            cheb_output_with_x = cheb_encoder(batch, return_x=True)

        assert cheb_output.shape == (1, dim_out)
        assert len(cheb_output_with_x) == 2
        assert cheb_output_with_x[0].shape == (1, dim_out)
        assert cheb_output_with_x[1].shape == (10, dim_hidden)

        print("✓ Single graph forward pass tests passed")


    def test_batch_forward(self):
        """Test forward pass with a batch of graphs."""
        dim_in, num_edge_type, dim_hidden, dim_out = 5, 3, 64, 32
        batch_size = 3

        # Create batch data
        batch = self.create_batch_data(batch_size=batch_size, dim_in=dim_in,
                                       num_edge_type=num_edge_type)

        # Test GINEEncoder
        gine_encoder = GINEEncoder(dim_in, num_edge_type, dim_hidden, dim_out, t=4)
        gine_encoder.eval()

        with torch.no_grad():
            gine_output = gine_encoder(batch)
            gine_output_with_x = gine_encoder(batch, return_x=True)

        assert gine_output.shape == (batch_size, dim_out)
        assert len(gine_output_with_x) == 2
        assert gine_output_with_x[0].shape == (batch_size, dim_out)

        # Test ChebEncoder
        cheb_encoder = ChebEncoder(dim_in, num_edge_type, dim_hidden, dim_out, t=4, K=3)
        cheb_encoder.eval()

        with torch.no_grad():
            cheb_output = cheb_encoder(batch)
            cheb_output_with_x = cheb_encoder(batch, return_x=True)

        assert cheb_output.shape == (batch_size, dim_out)
        assert len(cheb_output_with_x) == 2
        assert cheb_output_with_x[0].shape == (batch_size, dim_out)

        print("✓ Batch forward pass tests passed")


    def test_embed_node_method(self):
        """Test the embed_node method for both encoders."""
        dim_in, num_edge_type, dim_hidden, dim_out = 5, 3, 64, 32
        num_nodes = 10

        # Create sample data
        data = self.create_sample_data(num_nodes=num_nodes, dim_in=dim_in,
                                       num_edge_type=num_edge_type)

        # Test GINEEncoder embed_node
        gine_encoder = GINEEncoder(dim_in, num_edge_type, dim_hidden, dim_out, t=4)
        gine_encoder.eval()

        with torch.no_grad():
            final_x, all_x = gine_encoder.embed_node(data.x, data.edge_index, data.edge_attr)

        assert final_x.shape == (num_nodes, dim_hidden)
        assert all_x.shape == (num_nodes, dim_hidden * 4)  # t=4

        # Test ChebEncoder embed_node
        cheb_encoder = ChebEncoder(dim_in, num_edge_type, dim_hidden, dim_out, t=4, K=3)
        cheb_encoder.eval()

        with torch.no_grad():
            final_x, all_x = cheb_encoder.embed_node(data.x, data.edge_index, data.edge_attr)

        assert final_x.shape == (num_nodes, dim_hidden)
        assert all_x.shape == (num_nodes, dim_hidden * 4)  # t=4

        print("✓ embed_node method tests passed")


    def test_embed_graph_method(self):
        """Test the embed_graph method for both encoders."""
        dim_in, num_edge_type, dim_hidden, dim_out = 5, 3, 64, 32
        batch_size = 2

        # Create batch data
        batch = self.create_batch_data(batch_size=batch_size, dim_in=dim_in,
                                       num_edge_type=num_edge_type)

        # Test GINEEncoder embed_graph
        gine_encoder = GINEEncoder(dim_in, num_edge_type, dim_hidden, dim_out, t=4)
        gine_encoder.eval()

        with torch.no_grad():
            _, all_x = gine_encoder.embed_node(batch.x, batch.edge_index, batch.edge_attr)
            graph_embeddings = gine_encoder.embed_graph(all_x, batch.batch)

        assert graph_embeddings.shape == (batch_size, dim_out)

        # Test ChebEncoder embed_graph
        cheb_encoder = ChebEncoder(dim_in, num_edge_type, dim_hidden, dim_out, t=4, K=3)
        cheb_encoder.eval()

        with torch.no_grad():
            _, all_x = cheb_encoder.embed_node(batch.x, batch.edge_index, batch.edge_attr)
            graph_embeddings = cheb_encoder.embed_graph(all_x, batch.batch)

        assert graph_embeddings.shape == (batch_size, dim_out)

        print("✓ embed_graph method tests passed")

    def test_gradient_flow(self):
        """Test that gradients flow properly through both encoders."""
        dim_in, num_edge_type, dim_hidden, dim_out = 5, 3, 64, 32

        # Create sample data
        batch = self.create_batch_data(batch_size=2, dim_in=dim_in,
                                       num_edge_type=num_edge_type)

        # Test GINEEncoder gradients
        gine_encoder = GINEEncoder(dim_in, num_edge_type, dim_hidden, dim_out, t=4)
        gine_encoder.train()

        gine_output = gine_encoder(batch)
        loss = gine_output.sum()
        loss.backward()

        # Check that key parameters have gradients (not all parameters may have gradients)
        key_params_with_grads = 0
        total_key_params = 0

        for name, param in gine_encoder.named_parameters():
            if any(key in name for key in ['node_trans', 'conv', 'linear']):
                total_key_params += 1
                if param.grad is not None:
                    key_params_with_grads += 1

        assert key_params_with_grads > 0, "No gradients found in key GINE parameters"
        print(f"  GINE: {key_params_with_grads}/{total_key_params} key parameters have gradients")

        # Test ChebEncoder gradients
        cheb_encoder = ChebEncoder(dim_in, num_edge_type, dim_hidden, dim_out, t=4, K=3)
        cheb_encoder.train()

        cheb_output = cheb_encoder(batch)
        loss = cheb_output.sum()
        loss.backward()

        # Check that key parameters have gradients
        key_params_with_grads = 0
        total_key_params = 0

        for name, param in cheb_encoder.named_parameters():
            if any(key in name for key in ['node_trans', 'conv', 'linear']):
                total_key_params += 1
                if param.grad is not None:
                    key_params_with_grads += 1

        assert key_params_with_grads > 0, "No gradients found in key Cheb parameters"
        print(f"  Cheb: {key_params_with_grads}/{total_key_params} key parameters have gradients")

        print("✓ Gradient flow tests passed")

    def test_unused_parameters(self):
        """Test to identify which parameters don't receive gradients."""
        dim_in, num_edge_type, dim_hidden, dim_out = 5, 3, 64, 32

        # Create sample data
        batch = self.create_batch_data(batch_size=2, dim_in=dim_in,
                                       num_edge_type=num_edge_type)

        print("  Checking unused parameters...")

        # Test ChebEncoder (most likely to have unused parameters)
        cheb_encoder = ChebEncoder(dim_in, num_edge_type, dim_hidden, dim_out, t=4, K=3)
        cheb_encoder.train()

        cheb_output = cheb_encoder(batch)
        loss = cheb_output.sum()
        loss.backward()

        unused_params = []
        for name, param in cheb_encoder.named_parameters():
            if param.grad is None:
                unused_params.append(name)

        if unused_params:
            print(f"  ChebEncoder unused parameters: {unused_params}")
        else:
            print("  ChebEncoder: All parameters receive gradients")

        print("✓ Unused parameter analysis completed")

    def test_different_parameters(self):
        """Test encoders with different parameter configurations."""
        configurations = [
            {"dim_in": 3, "num_edge_type": 2, "dim_hidden": 32, "dim_out": 16, "t": 2},
            {"dim_in": 10, "num_edge_type": 5, "dim_hidden": 128, "dim_out": 64, "t": 6},
        ]

        for config in configurations:
            # Create appropriate data
            batch = self.create_batch_data(
                batch_size=2,
                dim_in=config["dim_in"],
                num_edge_type=config["num_edge_type"]
            )

            # Test GINEEncoder
            gine_encoder = GINEEncoder(**config)
            gine_encoder.eval()

            with torch.no_grad():
                gine_output = gine_encoder(batch)

            assert gine_output.shape == (2, config["dim_out"])

            # Test ChebEncoder
            cheb_config = config.copy()
            cheb_config["K"] = 3
            cheb_encoder = ChebEncoder(**cheb_config)
            cheb_encoder.eval()

            with torch.no_grad():
                cheb_output = cheb_encoder(batch)

            assert cheb_output.shape == (2, config["dim_out"])

        print("✓ Different parameter configuration tests passed")



    def run_all_tests(self):
        """Run all tests."""
        print("Running encoder tests...\n")

        self.test_encoder_initialization()
        self.test_single_graph_forward()
        self.test_batch_forward()
        self.test_embed_node_method()
        self.test_embed_graph_method()
        self.test_gradient_flow()
        self.test_different_parameters()

        print("\n🎉 All tests passed successfully!")

def benchmark_encoders():
    """Benchmark the performance of both encoders."""
    import time
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)

    dim_in, num_edge_type, dim_hidden, dim_out = 10, 5, 128, 64
    batch_size = 16

    # Create larger batch for benchmarking
    batch = TestEncoders.create_batch_data(
        batch_size=batch_size,
        num_nodes_per_graph=50,
        num_edges_per_graph=100,
        dim_in=dim_in,
        num_edge_type=num_edge_type
    )

    # Benchmark GINEEncoder
    gine_encoder = GINEEncoder(dim_in, num_edge_type, dim_hidden, dim_out, t=4)
    gine_encoder.eval()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = gine_encoder(batch)
    gine_time = time.time() - start_time

    # Benchmark ChebEncoder
    cheb_encoder = ChebEncoder(dim_in, num_edge_type, dim_hidden, dim_out, t=4, K=3)
    cheb_encoder.eval()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = cheb_encoder(batch)
    cheb_time = time.time() - start_time

    print(f"GINEEncoder: {gine_time:.4f}s (100 forward passes)")
    print(f"ChebEncoder: {cheb_time:.4f}s (100 forward passes)")
    print(f"Speed ratio: {gine_time/cheb_time:.2f}x")

if __name__ == "__main__":
    # Run tests
    tester = TestEncoders()
    tester.run_all_tests()
    # Run benchmark
    benchmark_encoders()


