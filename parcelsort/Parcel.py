import numpy as np

class Parcel:
    def __init__(self,
        timestamps: np.ndarray, # L
        features: np.ndarray # L x K
    ) -> None:
        self._timestamps = timestamps
        self._features = features
    @property
    def timestamps(self):
        return self._timestamps
    @property
    def features(self):
        return self._features
    def to_dict(self):
        return {
            'timestamps': self._timestamps.astype(np.int32),
            'features': self._features.astype(np.float32)
        }
    @staticmethod
    def from_dict(x: dict):
        return Parcel(
            timestamps=x['timestamps'],
            features=x['features']
        )