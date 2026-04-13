from dp_features import load_all_symbols, build_price_matrices
from dp_preproc_graph import process_graph
from dp_preproc_model1 import process_model1
from dp_preproc_model2 import process_model2


if __name__ == "__main__":
    px_15m = build_price_matrices(load_all_symbols("15m"))
    px_30m = build_price_matrices(load_all_symbols("30m"))
    px_1h = build_price_matrices(load_all_symbols("1h"))
    px_4h = build_price_matrices(load_all_symbols("4h"))

    process_graph(px_1h)
    process_model1(px_1h, px_4h)
    process_model2(px_15m, px_30m, px_1h)
