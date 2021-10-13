from typing import List
import json
import numpy as np
from .ParcelSortingSegment import ParcelSortingSegment
from .Parcel import Parcel
from .serialize_wrapper import _serialize, _deserialize

class ParcelSorting:
    def __init__(self, feature_components: np.ndarray) -> None:
        self._feature_components = feature_components
        self._segments: List[ParcelSortingSegment] = []
    @property
    def num_segments(self):
        return len(self._segments)
    @property
    def feature_components(self):
        return self._feature_components
    def segment(self, i: int):
        return self._segments[i]
    def add_segment(self, s: ParcelSortingSegment):
        self._segments.append(s)
    def to_dict(self, *, serialize=False):
        ret = {
            'feature_components': self._feature_components,
            'segments': [S.to_dict() for S in self._segments]
        }
        if serialize:
            ret = _serialize(ret)
        return ret
    def save(self, path: str):
        x = self.to_dict()
        with open(path, 'w') as f:
            json.dump(_serialize(x), f)
    def figurl(self):
        import figurl as fig
        return fig.Figure(
            type='sortingview.parcelexplorer.1',
            data = {
                'parcelSorting': self.to_dict(serialize=True)
            }
        )
    @staticmethod
    def from_dict(x: dict):
        x = _deserialize(x)
        P = ParcelSorting(feature_components=x['feature_components'])
        for s in x['segments']:
            P.add_segment(ParcelSortingSegment.from_dict(s))
        return P
    @staticmethod
    def load(path: str):
        with open(path, 'r') as f:
            x = _deserialize(json.load(f))
        return ParcelSorting.from_dict(x)
        

        