import dask.array
import dask
import itertools
import os
import zarr
from skimage.util import map_array
from skimage import measure, transform
import btrack
from btrack.constants import BayesianUpdates
import pandas as pd
import numpy as np
from multiprocessing import Pool

DATA_CHANNEL = 0

def makeBtrackObjects(zarr_group: zarr.hierarchy.Group, group_component: str = '1' ,segment_channel = 0):
    da = dask.array.from_zarr(zarr_group[group_component], component='1')  # Ch 0 is labeled, Ch 1 is labeled + watershed. New Ch 2 is labeled by track

    objects = btrack.utils.segmentation_to_objects(
        da[:,segment_channel,:, :, :],
        properties=('centroid', 'area')
    )
    return objects

def runBtrackOnZarrStore(array_shape, objects):
    #!#!#!#!#!#!#!#!#!#!#!#!#!#
    ##### BTrack Segment ######
    #!#!#!#!#!#!#!#!#!#!#!#!#!#
    # 'axis_major_length', 'axis_minor_length',

    n_t, n_c, n_z, n_y, n_x = array_shape

    def filterPyTrackObject(objects):
        objects_filt = [obj for obj in objects if float(obj.properties['area']) > 400 and float(obj.properties['area']) < 3000]
        return objects_filt

    # initialise a tracker session using a context manager
    with btrack.BayesianTracker() as tracker:
        # configure the tracker using a config file
        tracker.configure_from_file('/home/nick/postdoc/code/sopi_analysis/ome_ngff_workflow/cell_config.json')

        # append the objects to be tracked
        tracker.append(filterPyTrackObject(objects))
        tracker.update_method = BayesianUpdates.EXACT
        tracker.max_search_radius = 50
        tracker.volume = ((0, n_x), (0, n_y), (0, n_z)) # X, Y, Z for no clear reason
        tracker.track()
        _ = tracker.optimize()

        print("Writing file: " + IN_FILENAME[:-5] + '_tracks.h5')
        tracker.export(IN_FILENAME[:-5] + '_tracks.h5', obj_type='obj_type_1')
        # get the tracks as a python list
        tracks = tracker.tracks

        # optional: get the data in a format for napari
        data, properties, graph = tracker.to_napari(ndim=3)
        refs = tracker.refs

        return tracks, refs, (data, properties, graph)



#!#!#!#!#!#!#!#!#!#!#!#!#!#
##### BTrack Segment ######
#!#!#!#!#!#!#!#!#!#!#!#!#!#

def makeMetadataDf(objects, tracks):

    # Construct metadata dataframe from object data
    obj_metadata_df = pd.DataFrame([obj.to_dict() for obj in objects])


    # Make a dict of object id as the key, and the track ID as the value
    # Negative values are dummy objects and don't need to be included
    d = {}
    for track_id, refs_list in [(o.ID, o.refs) for o in tracks]:
        for obj_id in refs_list:
            if obj_id > 0:
                d[obj_id] = track_id

    # Add the 'track_id' to the DF that has the object info
    obj_metadata_df['track_id'] = [d.get(id, 0) for id in np.array(obj_metadata_df['ID'])]

    return obj_metadata_df

def daskMapTrackValues(block, block_id=None, data_df = None):
    t_idx = block_id[1]
    return map_array(block,
                     np.array(data_df[data_df['t'] == t_idx]['class_id']),
                     np.array(data_df[data_df['t'] == t_idx]['track_id']).astype(np.uint16)
                     )


def getPropsMap(t, IN_FILENAME, DATA_CHANNEL):
    print(f"Getting props for {t}, reading from channel {DATA_CHANNEL + 1}")
    store_segment_forprops = zarr.NestedDirectoryStore(IN_FILENAME)
    #data_array = dask.array.from_zarr(store_segment, component='0')  # Ch 0 is labeled, Ch 1 is labeled + watershed. New Ch 2 is labeled by track
    g_forprops = zarr.open_group(store_segment_forprops, mode = 'r')
    intensity_fname = IN_FILENAME[:-14] + ".zarr"
    if os.path.exists(intensity_fname):
        intensity_store = zarr.NestedDirectoryStore(intensity_fname)
        g_intensity = zarr.open_group(intensity_store, mode = 'r')
    def calcSurfArea(masked_object):
        if any(s < 3 for s in masked_object.shape):
            return 0
        if len(masked_object.shape) != 3:
            return 0
        calc_mask = np.full(shape = masked_object.shape, fill_value=True, dtype = bool)
        calc_mask = np.pad(calc_mask, 2, mode = 'constant', constant_values=False)

        masked_object = np.pad(masked_object, 2, mode = 'constant', constant_values=False)

        from skimage.measure import marching_cubes, mesh_surface_area
        verts, faces, _, _ = marching_cubes(masked_object, mask = calc_mask)
        return mesh_surface_area(verts, faces)

    if os.path.exists(intensity_fname):
        prop_res = measure.regionprops_table(np.squeeze(g_forprops['1'][t,DATA_CHANNEL + 1,:, :, :]),
                                             properties=('centroid', 'area', 'label', 'axis_major_length', 'bbox',
                                                         'intensity_min', 'intensity_mean', 'intensity_max'),
                                             extra_properties=(calcSurfArea,),
                                             intensity_image=g_intensity['1'][t,-1,:, :, :]),
    else:
        prop_res = measure.regionprops_table(np.squeeze(g_forprops['1'][t,DATA_CHANNEL + 1,:, :, :]),
                                             properties=('centroid', 'area', 'label', 'axis_major_length', 'bbox'),
                                             extra_properties=(calcSurfArea,)),
    prop_res_df = pd.DataFrame(prop_res[0])
    prop_res_df['t'] = t
    return prop_res_df

def calcAndWriteTrackedLabels(t, class_ids, track_ids, IN_FILENAME):
    print(f"Beginning timept {t} color_channel_matching")
    store_segment = zarr.NestedDirectoryStore(IN_FILENAME)
    g = zarr.open_group(store_segment, mode = 'a')
    if np.sum(g['1'][t, DATA_CHANNEL + 1, :, :, :]) != 0:
        print(f"Color matching at {t} already done, skipping")
        return

    # print("Calculating new values")
    new_mapped_vals = map_array(g['1'][t,DATA_CHANNEL,:, :, :],
                                np.array(class_ids, dtype = np.uint16),
                                np.array(track_ids, dtype = np.uint16)).astype(np.uint16)
    # print(f"Old new calc shape: {new_mapped_vals.shape}")
    # print("Values calculated, writing")
    # print(f"Zarr shape: {g['1'].shape}")
    # Write the values from the resolution that ran the calculation
    g['1'][t,DATA_CHANNEL + 1,:, :, :] = new_mapped_vals

    # print("Beginning scaleup and scaledown process")
    # Scale up the values and write those
    fullres_shape = g['0'].shape
    scaleup_vals = transform.rescale(new_mapped_vals, (2, 2, 2), order = 0, anti_aliasing=False, preserve_range=True).astype(np.uint16)
    g['0'][t, DATA_CHANNEL + 1, :, :, :] = scaleup_vals[0:fullres_shape[-3], 0:fullres_shape[-2], 0:fullres_shape[-1]]  # One px crop

    # Resize without messing up the labels
    for downsample_key in g.keys():
        if downsample_key in ['0', '1']:
            continue
        new_size = g[downsample_key][t, DATA_CHANNEL, :, :, :].shape[-3:]
        downsampled_data = transform.resize(new_mapped_vals,
                                            new_size, preserve_range=True,
                                            anti_aliasing=False).astype(np.uint16)
        g[downsample_key][t, DATA_CHANNEL + 1, :, :, :] = downsampled_data

    print(f"Finished timept {t} color_channel_matching")
    return

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Run btrack on segmented zarrs")
    group = parser.add_argument_group('Required')
    group.add_argument('--in_path', help="Input directory, loaction of the *.nd2.zst files", required=True)
    group_opt = parser.add_argument_group('Optional')
    group_opt.add_argument('--cores', help="number cores", default = os.cpu_count())
    args = parser.parse_args()

    IN_FILENAME = args.in_path
    n_cpu = int(args.cores)


    store_segment = zarr.NestedDirectoryStore(IN_FILENAME)
    if IN_FILENAME.endswith("gfp_segment.zarr"):
        DATA_CHANNEL = 1
    else:
        DATA_CHANNEL = 0

    g = zarr.open_group(store_segment, mode = 'a')
    da = dask.array.from_zarr(g['0'])
    # For GFP images Ch 0 is labeled, Ch 1 is labeled + watershed. New Ch 2 is labeled by track
    # For cellpose images, Ch0 is labeled - nothing in Ch1 and Ch2. Still used Ch2 for writing labels by track
    n_t, n_c, n_z, n_y, n_x = da.shape

    # For cellpose output, must extend the array along channel 2
    if g['0'].shape[1] < 2:
        print("Expanding axis for cellpose image")
        for group_key in g.keys():
            current_shape = list(g[group_key].shape)
            current_shape[1] = current_shape[1] + 1
            new_shape = tuple(current_shape)
            g[group_key].resize(new_shape)

    if os.path.exists(IN_FILENAME[:-5] + '_tracks.h5'):
        print("Loading from previous run")
        with btrack.dataio.HDF5FileHandler(IN_FILENAME[:-5] + '_tracks.h5') as btrack_obj:
            tracks = btrack_obj.tracks
            refs = [tracklist.refs for tracklist in tracks]
            napari_data = btrack.utils.tracks_to_napari(tracks, ndim=3)
            objects = btrack_obj.objects
    else:
        print("Making btrack objects")
        objects = makeBtrackObjects(g, 1, segment_channel=DATA_CHANNEL)
        print("Starting btrack")
        tracks, refs, napari_data = runBtrackOnZarrStore(da.shape, objects)

    print("Making metadata")
    obj_metadata_df = makeMetadataDf(objects, tracks)

    ## Write the corrected (tracked) output to the zarr
    ## This is far too slow... should just use a multiprocessing Pool approach
    # print("Writing tracked output colors")
    # g['0'][:,2,:, :, :] = da[:,1,:, :, :].map_blocks(daskMapTrackValues, data_df = obj_metadata_df, dtype = np.uint16)

    #def writeTrackedLabels():
    ## Create the args headed to the starmap
    # Need to make list of lists of pixel values to map
    class_ids = [list(np.array(obj_metadata_df[obj_metadata_df['t'] == t_idx]['class_id'], dtype = np.uint16)) for t_idx in range(n_t)]
    track_ids = [list(np.array(obj_metadata_df[obj_metadata_df['t'] == t_idx]['track_id'], dtype = np.uint16)) for t_idx in range(n_t)]

    remap_colors_args = list(zip(range(0, n_t),
                                  class_ids,
                                  track_ids,
                                  itertools.repeat(IN_FILENAME, n_t)))

    print("Beginning multithreaded run to write output colors")
    with Pool(processes=n_cpu) as pool:
        pool.starmap(calcAndWriteTrackedLabels, remap_colors_args, chunksize=4)

    map_props_args = list(zip(range(0, n_t),
                              itertools.repeat(IN_FILENAME, n_t),
                              itertools.repeat(DATA_CHANNEL, n_t)
                              ))
    ## Calculate object properties
    print("Beginning mulithreaded object property calculation")
    with Pool(processes=n_cpu) as pool:
        prop_res_df_list = pool.starmap(getPropsMap, map_props_args)


    prop_res_df = pd.concat(prop_res_df_list, ignore_index=True)
    print("Calculating sphericity")
    # Sphericity is defined as the area of a sphere with the same volume as the object of interest,
    # divided by the actual surface area
    # Equation from: https://www.calculatorsoup.com/calculators/geometry-solids/sphere.php
    # note here that 'area' is actually 'volume' when applied in 3D
    obj_metadata_df.to_csv(IN_FILENAME[:-5] + '_track_metadata.csv')

    prop_res_df['sphericity'] = ((np.pi ** (1/3)) * ((6 * prop_res_df['area']) ** (2/3))) / prop_res_df['calcSurfArea']

    prop_res_df.to_csv(IN_FILENAME[:-5] + '_track_properties.csv')


