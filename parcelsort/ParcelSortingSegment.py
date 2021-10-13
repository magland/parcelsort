from typing import List
import numpy as np
from .Parcel import Parcel


class ParcelSortingSegment:
    def __init__(self,
        start_frame: int,
        end_frame: int
    ) -> None:
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._parcels: List[Parcel] = []
    @property
    def num_parcels(self):
        return len(self._parcels)
    @property
    def start_frame(self):
        return self._start_frame
    @property
    def end_frame(self):
        return self._end_frame
    def parcel(self, i: int):
        return self._parcels[i]
    def add_parcel(self, p: Parcel):
        self._parcels.append(p)
    def to_dict(self):
        return {
            'start_frame': int(self._start_frame),
            'end_frame': int(self._end_frame),
            'parcels': [pp.to_dict() for pp in self._parcels]
        }
    @staticmethod
    def from_dict(x: dict):
        S = ParcelSortingSegment(start_frame=x['start_frame'], end_frame=x['end_frame'])
        for pp in x['parcels']:
            S.add_parcel(Parcel.from_dict(pp))
        return S