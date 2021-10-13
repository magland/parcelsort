import numpy as np
import spikeextractors as se
from .ParcelSorting import ParcelSorting
from .ParcelSortingSegment import ParcelSortingSegment
from .Parcel import Parcel

from .overcluster import overcluster
from ._detect_on_channel import detect_on_channel
from ._extract_snippets import extract_snippets
from ._estimate_noise_level import estimate_noise_level
from ._snippet_pca_features import compute_feature_components_from_snippets, apply_feature_components_to_snippets
from sklearn.neighbors import NearestNeighbors

def parcelsort(*,
    recording: se.RecordingExtractor,
    segment_duration: float,
    detect_threshold: float=6,
    detect_interval: int=20,
    parcel_size_factor: float=1,
    snippet_size: int=40,
    num_pca=10
):
    N = recording.get_num_frames() # num. timepoints
    M = recording.get_num_channels() # num. channels
    T = snippet_size # snippet size
    samplerate = recording.get_sampling_frequency()
    segment_size = int(segment_duration * samplerate)

    traces0 = recording.get_traces(start_frame=0, end_frame=segment_size).T
    noise_level = estimate_noise_level(traces0)
    
    num_segments = int(np.ceil(N / segment_size))

    max_radius = parcel_size_factor * np.sqrt(2 * M * T * (noise_level ** 2))

    for i in range(num_segments):
        print(f'Segment {i + 1} of {num_segments}')
        start_frame = i * segment_size
        end_frame = (i + 1) * segment_size
        S = ParcelSortingSegment(start_frame=start_frame, end_frame=end_frame)
        traces0 = recording.get_traces(start_frame=start_frame, end_frame=end_frame).T
        # traces0_maxabs = np.max(np.abs(traces0), axis=1)
        traces0_min = np.min(traces0, axis=1)
        times0 = detect_on_channel(
            traces0_min,
            detect_threshold=detect_threshold * noise_level,
            detect_sign=-1,
            detect_interval=detect_interval,
            margin=T
        )
        snippets0 = extract_snippets(traces0, times=times0, snippet_size=snippet_size)
        if i == 0:
            feature_components = compute_feature_components_from_snippets(snippets0, num_features=num_pca)
            P = ParcelSorting(feature_components=feature_components)
        features0 = apply_feature_components_to_snippets(feature_components, snippets0)
        labels0 = overcluster(features0, max_radius=max_radius)
        for a in range(1, np.max(labels0) + 1):
            times1 = start_frame + times0[np.where(labels0 == a)]
            pp = Parcel(timestamps=times1, features=features0[np.where(labels0 == a)])
            S.add_parcel(pp)
        print(f'{S.num_parcels} parcels')
        P.add_segment(S)
    return P