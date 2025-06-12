#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem

from psevo.encoder.gin import GINEEncoder
from psevo.encoder.cheb import ChebEncoder
from psevo.encoder.ml3 import GNNML3Model as ML3Encoder

from torch_geometric.nn import MLP

from utils.chem_utils import smiles2molecule, valence_check, cnt_atom, cycle_check, molecule2smiles, rec
from utils.chem_utils import del_N_positive
from utils.logger import print_log

class VAEPieceDecoder(nn.Module):
    """ Variational Autoencoder Piece Decoder for molecular generation.
    This decoder generates molecules in two stages:
    1. Sequential piece generation using RNN (molecular fragments/pieces)
    2. Edge prediction between atoms using graph neural networks

    The VAE framework allows for:
    - Latent space interpolation between molecules
    - Conditional generation based on molecular properties
    - Stochastic sampling for diverse molecule generation

    Mathematical Background:
    - Uses reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)
    - KL divergence: D_KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    - Reconstruction loss: piece prediction + edge prediction losses
    """

    def __init__(self,
                 atom_embedding_dim,
                 piece_embedding_dim,
                 max_pos,
                 pos_embedding_dim,
                 piece_hidden_dim,
                 node_hidden_dim,
                 num_edge_type,
                 cond_dim,
                 latent_dim,
                 tokenizer,
                 t=4,
                 encoder_type="gin"):
        """
        Initialize the VAE Piece Decoder.

        Args:
            atom_embedding_dim (int): Dimension for atom type embeddings
            piece_embedding_dim (int): Dimension for molecular piece embeddings
            max_pos (int): Maximum position index for positional encoding
            pos_embedding_dim (int): Dimension for positional embeddings
            piece_hidden_dim (int): Hidden dimension for RNN piece generation
            node_hidden_dim (int): Hidden dimension for graph node representations
            num_edge_type (int): Number of different bond types (single, double, triple, etc.)
            cond_dim (int): Dimension of conditioning vector (molecular properties)
            latent_dim (int): Dimension of VAE latent space
            tokenizer: Molecular tokenizer for piece vocabulary
            t (int): Number of message passing iterations in graph encoder
            encoder_type (str): Type of encoder to use ("cheb", "gin", "ml3")
        """
        super(VAEPieceDecoder, self).__init__()

        self.tokenizer = tokenizer

        # =================================================================
        # PIECE PREDICTION COMPONENTS (Sequential Generation)
        # =================================================================

        # Embedding layers for different molecular components
        self.atom_embedding = nn.Embedding(
            tokenizer.num_atom_type(),
            atom_embedding_dim
        )  # Maps atom types (C, N, O, etc.) to dense vectors

        self.piece_embedding = nn.Embedding(
            tokenizer.num_piece_type(),
            piece_embedding_dim
        )  # Maps molecular pieces/fragments to dense vectors

        self.pos_embedding = nn.Embedding(
            max_pos,
            pos_embedding_dim
        )  # Positional encoding for sequence order (max position = 99, 0 = padding)

        # VAE latent to RNN hidden state transformation
        self.latent_to_rnn_hidden = nn.Linear(latent_dim, piece_hidden_dim)

        # Recurrent neural network for sequential piece generation
        # Takes piece embeddings as input, outputs hidden states for vocabulary prediction
        self.rnn = nn.GRU(
            piece_embedding_dim,
            piece_hidden_dim,
            batch_first=True
        )

        # Output layer: hidden states → piece vocabulary probabilities
        self.to_vocab = nn.Linear(piece_hidden_dim, tokenizer.num_piece_type())

        # =================================================================
        # GRAPH EMBEDDING COMPONENTS (Structural Representation)
        # =================================================================

        # Combined node feature dimension: atom + piece + position information
        node_dim = atom_embedding_dim + piece_embedding_dim + pos_embedding_dim

        # Graph neural network encoder for learning node representations
        # Uses message passing to capture local molecular structure
        if encoder_type == "gin":
            self.graph_embedding = GINEEncoder(
                dim_in=node_dim,  # feature dimension for nodes
                num_edge_type=num_edge_type,  # edge feature dimension
                dim_hidden=node_hidden_dim,
                dim_out=1,  # dim_out not used in this context, dimension for final graph embeddings
                t=t, # Number of message passing iterations
            )
        elif encoder_type == "cheb":
            self.graph_embedding = ChebEncoder(
                dim_in=node_dim,  # feature dimension for nodes
                num_edge_type=num_edge_type,  # edge feature dimension
                dim_hidden=node_hidden_dim,
                dim_out=1,  # dim_out not used in this context, dimension for final graph embeddings
                t=t,  # Number of message passing iterations
                K=3,  # Chebyshev filter size
            )
        elif encoder_type == "ml3":
            self.graph_embedding = ML3Encoder(
                dim_in=node_dim,
                dim_hidden=node_hidden_dim,
                dim_out=1,
                num_layers=t,
                num_supports=num_edge_type,
                bandwidth=5.0,
                use_adjacency=False,
            )

        # =================================================================
        # EDGE LINK PREDICTION COMPONENTS (Bond Formation)
        # =================================================================

        # Input dimension for edge predictor:
        # source_node + target_node + latent_context
        mlp_in = node_hidden_dim * 2 + latent_dim

        # Multi-layer perceptron for predicting bond types between atom pairs
        # Architecture: MLP → Linear → bond_type_probabilities
        self.edge_predictor = MLP(
            in_channels=mlp_in,
            hidden_channels=mlp_in // 2,  # Bottleneck architecture
            out_channels=num_edge_type,
            num_layers=4,
            act="relu",
            plain_last=True, # Final classification layer is without act
        )


        # =================================================================
        # VARIATIONAL AUTOENCODER COMPONENTS
        # =================================================================

        self.latent_dim = latent_dim

        # VAE encoder outputs: condition → (mean, log_variance)
        # Reparameterization trick: z = mean + exp(log_var/2) * epsilon
        self.W_mean = nn.Linear(cond_dim, latent_dim)      # μ(condition)
        self.W_log_var = nn.Linear(cond_dim, latent_dim)   # log(σ²(condition))

        # =================================================================
        # LOSS FUNCTIONS
        # =================================================================

        # Cross-entropy loss for piece sequence prediction
        # Ignores padding tokens in loss calculation
        self.piece_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx())

        # Cross-entropy loss for edge/bond type prediction
        self.edge_loss = nn.CrossEntropyLoss()






