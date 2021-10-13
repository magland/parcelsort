import numpy as np

def compute_feature_components_from_snippets(snippets: np.ndarray, num_features: int):
    # snippets is L x T x M
    u, s, vt = np.linalg.svd(snippets.reshape((snippets.shape[0], snippets.shape[1] * snippets.shape[2])), full_matrices=False)
    # u: L x n, s: n, vt: n x n
    # features = u[:, :num_features]
    components = vt[:num_features, :].reshape((num_features, snippets.shape[1], snippets.shape[2])) # K x T x M
    return components # K x T x M

def apply_feature_components_to_snippets(components: np.ndarray, snippets: np.ndarray):
    # components: K x T x M
    # snippets: L x T x M
    # returned features: L x K
    snippets_matrix = snippets.reshape((snippets.shape[0], snippets.shape[1] * snippets.shape[2]))
    components_matrix = components.reshape((components.shape[0], components.shape[1] * components.shape[2]))
    features = snippets_matrix @ components_matrix.T # L x K
    return features