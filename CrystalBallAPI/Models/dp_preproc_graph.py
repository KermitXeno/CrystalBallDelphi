from dp_features import *


def process_graph(px_1h):
    print("Building graph edges...")
    r = np.log(px_1h["close"] / px_1h["close"].shift(1)).ffill().fillna(0)
    assets = r.columns.tolist()
    N = len(assets)

    granger_w = granger_pairwise_weights(r, window = GRANGER_WINDOW_1H, lag = GRANGER_LAG)
    granger_sparse = granger_w.copy()
    granger_sparse[granger_sparse < GRANGER_SPARSE_THRESHOLD] = 0.0

    e_stab = edge_stability(granger_w, window = GRANGER_STABILITY_WINDOW)
    in_deg, out_deg, eigvec, clust, between = graph_topology_features(granger_sparse)
    graph_dens, graph_ent = graph_level_features(granger_sparse)

    r_arr = r.values.astype(np.float32)
    ef = compute_edge_features(granger_sparse, r_arr)
    e_stab_4d = e_stab[:, :, :, np.newaxis].astype(np.float16)
    edge_features = np.concatenate([ef, e_stab_4d], axis = -1)

    times = r.index.asi8

    save_npz("graph_edges",
             edge_features = edge_features,
             node_in_degree = in_deg,
             node_out_degree = out_deg,
             node_eigvec = eigvec,
             node_clustering = clust,
             node_betweenness = between,
             graph_density = graph_dens,
             graph_entropy = graph_ent,
             times = times)

    print(f"Graph complete. Timesteps: {len(times)}, assets: {N}, F_edge: {edge_features.shape[-1]}")
    return r.index
