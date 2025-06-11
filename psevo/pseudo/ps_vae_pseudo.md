#### Forward Method Pseudo Code

Inputs:
- batch: dict containing graph and molecule data (see data.bpe_dataset for details)
- return_accu: bool (default False), whether to return accuracy metrics

---

1. **Extract batch data**
    - `x`                : [batch_size, node_num]         # atom indices
    - `edge_index`       : [2, num_edges]                 # edge connections
    - `edge_attr`        : [num_edges, edge_dim]          # edge features
    - `x_pieces`         : [batch_size, node_num, ...]    # atom piece indices
    - `x_pos`            : [batch_size, node_num, ...]    # atom positions

2. **Atom Embedding**
    - `x = decoder.embed_atom(x, x_pieces, x_pos)`
        - Output: `x` : [batch_size, node_num, node_dim]  # embedded atom features

3. **Prepare graph ids for batching**
    - `graph_ids = repeat_interleave(arange(0, batch_size), node_num)`
        - Output: [batch_size * node_num]

4. **Node Embedding via Encoder**
    - `_, all_x = encoder.embed_node(x.view(-1, node_dim), edge_index, edge_attr)`
        - Input: [batch_size * node_num, node_dim], [2, num_edges], [num_edges, edge_dim]
        - Output: `all_x` : [batch_size * node_num, node_feature_dim]

5. **Graph Embedding**
    - `graph_embedding = encoder.embed_graph(all_x, graph_ids, atom_mask.flatten())`
        - Input: [batch_size * node_num, node_feature_dim], [batch_size * node_num], [batch_size * node_num]
        - Output: [batch_size, dim_graph_feature]

6. **Reconstruction via Decoder**
    - `in_piece_edge_idx = batch['in_piece_edge_idx']`
    - `z, res = decoder(x, x_pieces, x_pos, edge_index[:, in_piece_edge_idx], edge_attr[in_piece_edge_idx], pieces, conds=graph_embedding, edge_select, golden_edge, return_accu)`
        - Inputs:
            - `x`                : [batch_size, node_num, node_dim]
            - `x_pieces`         : [batch_size, node_num, ...]
            - `x_pos`            : [batch_size, node_num, ...]
            - `edge_index`       : [2, num_in_piece_edges]
            - `edge_attr`        : [num_in_piece_edges, edge_dim]
            - `pieces`           : [batch_size, ...]
            - `conds`            : [batch_size, dim_graph_feature]
            - `edge_select`      : [batch_size, ...]
            - `golden_edge`      : [batch_size, ...]
        - Outputs:
            - `z`    : [batch_size, latent_dim]
            - `res`  : tuple (losses, accus) or just losses

7. **Property Prediction**
    - `pred_prop = predictor(z)`
        - Input: [batch_size, latent_dim]
        - Output: [batch_size, num_properties]

8. **Prepare Ground Truth Properties**
    - `golden = batch['props'].reshape(batch_size, -1)[:, selected_properties]`
        - Output: [batch_size, len(selected_properties)]
    - `golden = golden.float()`

9. **Prediction Loss**
    - `pred_loss = MSELoss(pred_prop, golden)`
        - Output: scalar

10. **Return**
    - If `return_accu` is True:
        - Return: `(pred_loss, (losses, accus))`
    - Else:
        - Return: `(pred_loss, losses)`

---

#### **Tensor Dimensions Summary**

| Variable         | Shape                                      |
|------------------|--------------------------------------------|
| x                | [batch_size, node_num, node_dim]           |
| edge_index       | [2, num_edges]                             |
| edge_attr        | [num_edges, edge_dim]                      |
| all_x            | [batch_size * node_num, node_feature_dim]  |
| graph_embedding  | [batch_size, dim_graph_feature]            |
| z                | [batch_size, latent_dim]                   |
| pred_prop        | [batch_size, num_properties]               |
| golden           | [batch_size, len(selected_properties)]     |

---

#### **High-Level Flow**

1. **Embed atoms** → **Embed nodes** → **Aggregate to graph embedding**
2. **Decode/reconstruct** with graph embedding as condition
3. **Predict properties** from latent vector
4. **Compute loss** and return

