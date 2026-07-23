import logging

import numpy as np

from src.post_process.post_process import graph_to_cityjson

logger = logging.getLogger(__name__)


def draw_samples(model, num_batches, batch_size):
    """Run the full reverse chain num_batches times and convert each instance.

    Returns (records, stats). Records are only the reconstructable buildings;
    stats tracks attempted vs dropped so a rejection rate can be computed.
    """
    model.eval()
    records, attempted, dropped = [], 0, 0
    for _ in range(num_batches):
        pos, node_labels, edge_labels = model.sample(batch_size=batch_size)
        for i in range(pos.shape[0]):
            attempted += 1
            coords = model._denormalize_coords(pos[i])
            nl = node_labels[i].detach().cpu().numpy()
            el = edge_labels[i].detach().cpu().numpy()
            cj = graph_to_cityjson(coords, nl, el, building_id=f"gen_{attempted - 1}")
            if not cj:
                dropped += 1
                continue
            records.append({"coords": coords, "node_labels": nl,
                            "edge_labels": el, "cityjson": cj})
    return records, {"attempted": attempted, "dropped": dropped}
