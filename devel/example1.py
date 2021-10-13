from parcelsort import parcelsort, ParcelSorting
import kachery_client as kc
import sortingview as sv
import os

def main():
    if not os.getenv('FIGURL_CHANNEL'): raise Exception('You must set the FIGURL_CHANNEL environment variable')

    # The URI of the .h5v1 recording file (in this example, channel ids are 192-223)
    recording_h5v1_uri = 'sha1://1e2ec04b531ff5211febecf8691df3d141586f67/J1620210602_.nwb_raw_data_valid_times_first_hour_25_franklab_default_cortex_recording.h5v1?manifest=618095f0b0db08aa3c33842d2b6b32a249f812e6'
    subrec_duration_min = 60 # Process only the first portion of the recording
    segment_duration_min = 20 # This is the duration of individual segments
    channel_ids = [201, 202, 203, 204] # Let's focus on only 4 channels to start
    label = 'Parcelsort example'

    R = load_recording_from_h5v1(recording_h5v1_uri)
    R = sv.subrecording(recording=R, channel_ids=channel_ids, start_frame=0, end_frame=R.get_sampling_frequency() * 60 * subrec_duration_min)

    # Run parcelsort -- not quite spike sorting, just splitting events into manageable parcels
    psorting = parcelsort(recording=R, segment_duration=segment_duration_min * 60, parcel_size_factor=5)

    parcel_sorting_uri = kc.store_json(psorting.to_dict(serialize=True))
    # Later this can be loaded via:
    #     from parcelsort import ParcelSorting
    #     psorting = ParcelSorting.from_dict(kc.load_json(uri))
    print(f'Parcel sorting: {parcel_sorting_uri}')
    
    F = psorting.figurl()
    url = F.url(label=label)
    print(f'FigURL: {url}')

def load_recording_from_h5v1(uri: str):
    recording_object = {"recording_format":"h5_v1","data":{"h5_uri":uri}}
    R = sv.LabboxEphysRecordingExtractor(recording_object)
    return R

if __name__ == '__main__':
    main()