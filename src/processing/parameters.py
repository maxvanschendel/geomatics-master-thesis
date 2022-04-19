from dataclasses import dataclass


@dataclass
class PipelineParameters:
    """Defines pipeline behaviour."""

    # Map extraction
    skip_extract: bool = True   # Skip map extraction step and go directly to map matching
    write_htmap: bool = True    # Write map extraction results to files

    partial_map_a: str = "../data/cslam_dataset/diningroom2kitchen.ply"
    partial_map_b: str = "../data/cslam_dataset/hall2oldkitchen.ply"
    htmap_a_fn: str = '../data/test/diningroom2kitchen_htmap.pickle'
    htmap_b_fn: str = '../data/test/hall2oldkitchen_htmap.pickle'

    # Map matching
    skip_match: bool = False    # Skip map matching and go directly to map merging

    # Map merging
    skip_merge: bool = False    # Skip map merging and go directly to evaluation

    def post_init(self):
        """Performs validation of pipeline configuration input,
        """

        pass

