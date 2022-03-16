from pyg_dataset import GraphDataset
from sketchgraphs_models.graph import dataset
from sketchgraphs_models.graph.train.data_loading import load_sequences_and_mappings

dataset_train_path = 'sg_t16_train.npy'
quantization = {'angle': 127, 'length': 383}
seed = 7


def load_dataset_and_weights(dataset_file, auxiliary_file, quantization, seed=None,
                             entity_features=True, edge_features=True, force_entity_categorical_features=False):
    data = load_sequences_and_mappings(dataset_file, auxiliary_file, quantization, entity_features, edge_features)

    if data['entity_feature_mapping'] is None and force_entity_categorical_features:
        # Create an entity mapping which only computes the categorical features (i.e. isConstruction and clockwise)
        data['entity_feature_mapping'] = dataset.EntityFeatureMapping()

    return GraphDataset(
        data['sequences'], data['entity_feature_mapping'], data['edge_feature_mapping'], seed=seed), data['weights']


ds_train, weightsgr_train = load_dataset_and_weights(dataset_train_path, None, quantization, seed, entity_features=False, edge_features=False, force_entity_categorical_features=False)


ds_train