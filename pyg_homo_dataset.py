import torch
import numpy as np
from sketchgraphs_models.graph.dataset import graph_info_from_sequence
from sketchgraphs.data import sketch as data_sketch, sequence as data_sequence
from torch_geometric.data import Data
from sketchgraphs_models.graph import dataset
from sketchgraphs_models.graph.train.data_loading import load_sequences_and_mappings

def load_homogeneous_dataset(dataset_file, auxiliary_file, quantization, seed=None,
                             force_entity_categorical_features=False):
    data = load_sequences_and_mappings(dataset_file, auxiliary_file, quantization, False, False)

    if data['entity_feature_mapping'] is None and force_entity_categorical_features:
        # Create an entity mapping which only computes the categorical features (i.e. isConstruction and clockwise)
        data['entity_feature_mapping'] = dataset.EntityFeatureMapping()

    return HomogeneousGraphDataset(
        data['sequences'], data['entity_feature_mapping'], data['edge_feature_mapping'], seed=seed), data['weights']



class HomogeneousGraphDataset(torch.utils.data.Dataset):

    def __init__(self, sequences, node_feature_mapping=None, edge_feature_mapping=None, seed=None):
        self.sequences = sequences
        self.rng = np.random.RandomState(seed)
        self.edge_feature_mapping = edge_feature_mapping
        self.node_feature_mapping = node_feature_mapping

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        idx = idx % len(self)  # allows using batch size larger than dataset
        seq = self.sequences[idx]

        # Exclude first step since we always start w/ external node
        # Exclude subnode edges since they can be inferred by the subnode op.
        step_indices = [i for i, op in enumerate(seq) if i > 0 and not _is_subnode_edge(op)]

        step_idx = self.rng.choice(step_indices)

        try:
            graph = graph_info_from_sequence(seq[:step_idx], self.node_feature_mapping, self.edge_feature_mapping)
        except Exception as e:
            raise ValueError('Failed to process sequence at index {0}'.format(idx)) from e

        target = seq[step_idx]

        return Data(x = graph.node_features, edge_index = graph.incidence, edge_attr= graph.edge_features), target


def _is_subnode_edge(op):
    return isinstance(op, data_sequence.EdgeOp) and op.label == data_sketch.ConstraintType.Subnode
