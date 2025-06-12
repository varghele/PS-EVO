import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import numpy as np
import math
from typing import Optional
import pytest

# Import the cleaned GNNML3 classes (assuming they're in the same file or imported)
from psevo.encoder.ml3 import SpectralConvolution, GNNML3Layer, GNNML3Model


class TestGNNML3:
    """Comprehensive test suite for GNNML3."""

    @staticmethod
    def create_sample_graph(num_nodes=10, num_edges=20, dim_features=5):
        """Create a sample graph for testing."""
        # Node features
        x = torch.randn(num_nodes, dim_features)

        # Random edge connectivity
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Ensure some self-loops and bidirectional edges for better connectivity
        self_loops = torch.arange(num_nodes).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loops], dim=1)

        # Remove duplicates
        edge_index = torch.unique(edge_index, dim=1)

        return Data(x=x, edge_index=edge_index)

    @staticmethod
    def create_batch_graphs(batch_size=3, num_nodes=8, num_edges=15, dim_features=5):
        """Create a batch of graphs."""
        graphs = []
        for _ in range(batch_size):
            graph = TestGNNML3.create_sample_graph(num_nodes, num_edges, dim_features)
            graphs.append(graph)
        return Batch.from_data_list(graphs)

    def test_spectral_convolution_layer(self):
        """Test the SpectralConvolution layer."""
        print("Testing SpectralConvolution layer...")

        # Parameters
        in_channels, out_channels = 16, 32
        num_supports = 5
        num_nodes, num_edges = 10, 20

        # Create test data
        x = torch.randn(num_nodes, in_channels)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, num_supports)  # Precomputed spectral supports

        # Create layer
        conv = SpectralConvolution(in_channels, out_channels, K=num_supports)

        # Forward pass
        out = conv(x, edge_index, edge_attr)

        # Assertions
        assert out.shape == (num_nodes, out_channels)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        print("✓ SpectralConvolution layer test passed")

    def test_gnnml3_layer(self):
        """Test the GNNML3Layer."""
        print("Testing GNNML3Layer...")

        # Parameters
        nedgeinput, nedgeoutput = 5, 5
        ninp, nout1, nout2 = 16, 8, 8
        num_nodes, num_edges = 10, 20

        # Create test data
        x = torch.randn(num_nodes, ninp)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, nedgeinput)

        # Create layer
        layer = GNNML3Layer(
            learnedge=True,
            nedgeinput=nedgeinput,
            nedgeoutput=nedgeoutput,
            ninp=ninp,
            nout1=nout1,
            nout2=nout2
        )

        # Forward pass
        out = layer(x, edge_index, edge_attr)

        # Assertions
        expected_out_dim = nout1 + nout2  # Concatenation of spectral and element-wise branches
        assert out.shape == (num_nodes, expected_out_dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        print("✓ GNNML3Layer test passed")

    def test_full_model_single_graph(self):
        """Test the complete GNNML3 model on a single graph."""
        print("Testing full GNNML3 model on single graph...")

        # Parameters
        dim_in, dim_out = 5, 10
        dim_hidden = 32

        # Create model
        model = GNNML3Model(
            dim_in=dim_in,
            dim_out=dim_out,
            dim_hidden=dim_hidden,
            num_layers=2,
            num_supports=3
        )

        # Create test graph
        graph = self.create_sample_graph(num_nodes=8, dim_features=dim_in)

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(graph)

        # Assertions
        assert output.shape == (1, dim_out)  # Single graph output
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        print("✓ Full model single graph test passed")

    def test_full_model_batch(self):
        """Test the complete GNNML3 model on a batch of graphs."""
        print("Testing full GNNML3 model on batch...")

        # Parameters
        dim_in, dim_out = 5, 10
        dim_hidden = 32
        batch_size = 4

        # Create model
        model = GNNML3Model(
            dim_in=dim_in,
            dim_out=dim_out,
            dim_hidden=dim_hidden,
            num_layers=2,
            num_supports=3
        )

        # Create test batch
        batch = self.create_batch_graphs(batch_size=batch_size, dim_features=dim_in)

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(batch)

        # Assertions
        assert output.shape == (batch_size, dim_out)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        print("✓ Full model batch test passed")

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        print("Testing gradient flow...")

        # Parameters
        dim_in, dim_out = 5, 2
        dim_hidden = 16

        # Create model
        model = GNNML3Model(
            dim_in=dim_in,
            dim_out=dim_out,
            dim_hidden=dim_hidden,
            num_layers=2,
            num_supports=3
        )

        # Create test data
        graph = self.create_sample_graph(num_nodes=6, dim_features=dim_in)
        target = torch.randn(1, dim_out)

        # Forward pass
        model.train()
        output = model(graph)
        loss = F.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check gradients
        has_gradients = 0
        total_params = 0

        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None:
                has_gradients += 1
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

        assert has_gradients > 0, "No parameters received gradients"
        print(f"  {has_gradients}/{total_params} parameters have gradients")

        print("✓ Gradient flow test passed")

    def test_spectral_preprocessing(self):
        """Test the spectral preprocessing step."""
        print("Testing spectral preprocessing...")

        # Create model
        model = GNNML3Model(dim_in=5, dim_out=10, num_supports=4)

        # Create test graph
        graph = self.create_sample_graph(num_nodes=6, num_edges=12, dim_features=5)

        # Test preprocessing
        edge_attr = model.preprocess_graph(graph)

        # Assertions
        assert edge_attr.shape[0] == graph.edge_index.shape[1]  # num_edges
        assert edge_attr.shape[1] == model.num_supports
        assert not torch.isnan(edge_attr).any()
        assert not torch.isinf(edge_attr).any()

        print("✓ Spectral preprocessing test passed")

    def test_different_configurations(self):
        """Test different model configurations."""
        print("Testing different configurations...")

        configurations = [
            {"num_supports": 3, "num_layers": 1, "use_adjacency": False},
            {"num_supports": 5, "num_layers": 3, "use_adjacency": True},
            {"num_supports": 1, "num_layers": 2, "use_adjacency": False},
        ]

        for i, config in enumerate(configurations):
            print(f"  Testing configuration {i + 1}: {config}")

            model = GNNML3Model(
                dim_in=4,
                dim_out=6,
                dim_hidden=16,
                **config
            )

            graph = self.create_sample_graph(num_nodes=5, dim_features=4)

            model.eval()
            with torch.no_grad():
                output = model(graph)

            assert output.shape == (1, 6)
            assert not torch.isnan(output).any()

        print("✓ Different configurations test passed")

    def test_3wl_expressiveness_simple(self):
        """
        Simple test for 3-WL expressiveness using triangle counting.

        This tests if GNNML3 can distinguish graphs that differ in triangle count,
        which 1-WL equivalent methods cannot do.
        """
        print("Testing 3-WL expressiveness (triangle counting)...")

        # Create two graphs with different triangle counts
        # Graph 1: Triangle (3 nodes, 3 edges forming a cycle)
        edge_index_1 = torch.tensor([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]], dtype=torch.long)
        x_1 = torch.ones(3, 1)  # Same node features
        graph_1 = Data(x=x_1, edge_index=edge_index_1)

        # Graph 2: Path (3 nodes, 2 edges in a line)
        edge_index_2 = torch.tensor([[0, 1, 1, 0], [1, 2, 0, 2]], dtype=torch.long)  # Bidirectional
        x_2 = torch.ones(3, 1)  # Same node features
        graph_2 = Data(x=x_2, edge_index=edge_index_2)

        # Create model
        model = GNNML3Model(
            dim_in=1,
            dim_out=8,
            dim_hidden=32,
            num_layers=3,
            num_supports=5
        )

        # Get embeddings
        model.eval()
        with torch.no_grad():
            emb_1 = model(graph_1)
            emb_2 = model(graph_2)

        # Check if embeddings are different
        distance = torch.norm(emb_1 - emb_2).item()

        print(f"  Distance between triangle and path embeddings: {distance:.6f}")
        assert distance > 1e-4, "GNNML3 should distinguish triangle from path"

        print("✓ 3-WL expressiveness test passed")

    def test_spectral_ability(self):
        """Test spectral filtering abilities."""
        print("Testing spectral filtering abilities...")

        # Create a simple graph with known spectral properties
        # Complete graph K4 has known eigenvalues
        num_nodes = 4
        edge_list = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_list.extend([[i, j], [j, i]])  # Bidirectional

        edge_index = torch.tensor(edge_list, dtype=torch.long).T
        x = torch.randn(num_nodes, 3)
        graph = Data(x=x, edge_index=edge_index)

        # Test with different numbers of spectral supports
        for num_supports in [1, 3, 5]:
            model = GNNML3Model(
                dim_in=3,
                dim_out=4,
                num_supports=num_supports,
                bandwidth=1.0
            )

            model.eval()
            with torch.no_grad():
                output = model(graph)

            assert output.shape == (1, 4)
            assert not torch.isnan(output).any()

        print("✓ Spectral filtering test passed")

    def run_all_tests(self):
        """Run all tests."""
        print("=" * 60)
        print("RUNNING GNNML3 TESTS")
        print("=" * 60)

        try:
            self.test_spectral_convolution_layer()
            self.test_gnnml3_layer()
            self.test_full_model_single_graph()
            self.test_full_model_batch()
            self.test_gradient_flow()
            self.test_spectral_preprocessing()
            self.test_different_configurations()
            self.test_3wl_expressiveness_simple()
            self.test_spectral_ability()

            print("\n" + "=" * 60)
            print("🎉 ALL GNNML3 TESTS PASSED SUCCESSFULLY!")
            print("=" * 60)

        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            raise

def benchmark_gnnml3():
    """Benchmark GNNML3 performance."""
    import time
    print("\n" + "=" * 60)
    print("GNNML3 PERFORMANCE BENCHMARK")
    print("=" * 60)

    # Test parameters
    batch_size = 8
    num_nodes = 20
    num_edges = 60
    dim_features = 10

    # Create model
    model = GNNML3Model(
        dim_in=dim_features,
        dim_out=16,
        dim_hidden=64,
        num_layers=3,
        num_supports=5
    )

    # Create test batch
    batch = TestGNNML3.create_batch_graphs(
        batch_size=batch_size,
        num_nodes=num_nodes,
        num_edges=num_edges,
        dim_features=dim_features
    )

    # Warm up
    model.eval()
    with torch.no_grad():
        _ = model(batch)

    # Benchmark forward pass
    num_runs = 50
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(batch)

    forward_time = (time.time() - start_time) / num_runs

    # Benchmark backward pass
    model.train()
    start_time = time.time()

    for _ in range(num_runs):
        model.zero_grad()
        output = model(batch)
        loss = output.sum()
        loss.backward()

    backward_time = (time.time() - start_time) / num_runs

    print(f"Forward pass time: {forward_time * 1000:.2f} ms")
    print(f"Backward pass time: {backward_time * 1000:.2f} ms")
    print(f"Total time per iteration: {(forward_time + backward_time) * 1000:.2f} ms")

    # Memory usage
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

if __name__ == "__main__":
    # Run tests
    tester = TestGNNML3()
    tester.run_all_tests()
    # Run benchmark
    benchmark_gnnml3()
