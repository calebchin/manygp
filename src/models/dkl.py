import gpytorch

class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, gp_layer, per_feature=False):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = gp_layer
        self.per_feature = per_feature

    def forward(self, x):
        features = self.feature_extractor(x)
        if self.per_feature:
            features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res
