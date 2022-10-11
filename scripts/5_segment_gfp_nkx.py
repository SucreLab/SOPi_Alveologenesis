import sys

# == Custom helper functions == #
from helpers import readND2Img, affineImage, createOMEzarrND2, extendOMEzarr, calcZIsoFactorND2, writeToOMEzarr, createAffineMask, roundToOdd
from config import ROT_90, MAX_DOWNSAMPLE_FACTOR, COMPRESSION_LEVEL
import zarr
import numcodecs
import os
import numpy as np
from multiprocessing import Pool, shared_memory
from skimage.morphology import cube, erosion
import time

import itertools
THREADS_PER_PROCESS = 1
COMPRESS_THREADS = 1
RUN_THREADS = 14
# For each view (run in parallel)
# First round special - Read the i image - read from original file, write to new file

# Read the i image - from new file
# Read the i + 1 (second image) - from original file
# Find the drift offset, and correct the i + 1 image - write to new file
# Iterate i
# Repeat block above
if len(sys.argv) < 2:
    raise ValueError("Must supply in zarr name")

in_zarr_filename = sys.argv[1]
if not in_zarr_filename.endswith("driftfix.zarr"):
    raise RuntimeError(f"Input path must end in driftfix.zarr: {in_zarr_filename}")
out_zarr_filename = in_zarr_filename[:-5] + "_gfp_segment.zarr"
complete_filename = in_zarr_filename[:-5] + "_gfp_segment_complete.txt"

# This script takes a single OME zarr and does a drift correction
# Multithreading on this is bad because it needs to be sequential by nature
# The number of threads that this script takes is configurable by COMPRESS_THREADS above
# Can be with more efficient multiprocess with something like snakemake

def createCopyOMEZarr(in_zarr_filename, out_zarr_filename, COMPRESS_THREADS = THREADS_PER_PROCESS):
    old_store = zarr.NestedDirectoryStore(in_zarr_filename)
    old_g = zarr.open_group(old_store, mode = 'r')

    # Create the new mirror zarr
    store = zarr.NestedDirectoryStore(out_zarr_filename)
    g = zarr.open_group(store, mode = 'a')

    for attr_key in old_g.attrs.keys():
        g.attrs[attr_key] = old_g.attrs[attr_key]

    for group_name in list(old_g.array_keys()):
        g.create(name=group_name,
                 shape=old_g[group_name].shape,
                 chunks=old_g[group_name].chunks,
                 dtype=old_g[group_name].dtype,
                 compressor=numcodecs.Blosc(cname='zstd', clevel=COMPRESSION_LEVEL))


def createAffineMaskSingleFrame(img_z, img_y, img_x):
    ## == Mask out the afined edges == ##

    pad_clip_px = 1

    img_mask = np.tri(img_y)
    img_mask_3d = np.broadcast_to(img_mask, (img_x, img_mask.shape[0], img_mask.shape[0])) # x, y, z
    img_mask_3d_roll = np.transpose(img_mask_3d) # z y x
    z_len_to_pad = img_z - img_mask_3d_roll.shape[0]
    img_mask_3d_padded = np.lib.pad(img_mask_3d_roll, ((0, z_len_to_pad - (pad_clip_px * 2)), (0, 0), (0, 0)))

    # Make the mask happen on both sides
    start_array = np.flip(img_mask_3d_padded, axis = 1)
    end_array = np.flip(img_mask_3d_padded, axis = 0)
    affine_mask = np.logical_or(start_array, end_array)

    # Extend the mask two px in
    affine_mask = np.pad(affine_mask, ((pad_clip_px, pad_clip_px), (0, 0), (0, 0)), constant_values=1).astype(bool)
    affine_mask = np.invert(affine_mask)  # Inverted mask = True is places to keep
    return affine_mask


def thresholdSingleChannel(in_zarr_filename, out_zarr_filename, t, COMPRESS_THREADS=1):
    # from skimage import exposure
    # from skimage import filters
    from skimage import morphology
    from skimage import segmentation
    from skimage import measure
    from skimage import feature
    from skimage import transform
    from scipy import ndimage as ndi
    ### Start here
    start_time = time.time()

    print(f"Starting timepoint {t}")

    in_store = zarr.NestedDirectoryStore(in_zarr_filename)
    g = zarr.open_group(in_store, mode = 'r')
    fullres = g['0']  # T C Z Y X
    halfres = g['1']  # T C Z Y X
    frame_for_thresh = np.squeeze(halfres[t, 2, :]).astype(np.uint16)



    thresh_pct = frame_for_thresh > np.percentile(frame_for_thresh, 99.5)

    del frame_for_thresh
    eroded_thresh_pct = erosion(thresh_pct, footprint = cube(4))

    eroded_thresh_pct_closed = morphology.closing(eroded_thresh_pct, cube(8))  # Close the holes in the thresholded image

    affine_mask = createAffineMaskSingleFrame(halfres.shape[-3], halfres.shape[-2], halfres.shape[-1])
    noedges_thresh = segmentation.clear_border(eroded_thresh_pct_closed, mask = affine_mask)  # Todo: !!! Apply the SOPi mask here

    label_image = measure.label(noedges_thresh).astype(np.uint16)
    label_image_resize = transform.resize(label_image, fullres.shape[-3:], order = 0, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=0).astype(label_image.dtype)

    writeToOMEzarr(label_image_resize, zarr_filename= out_zarr_filename, t = t, c = 0)

    del eroded_thresh_pct
    del eroded_thresh_pct_closed
    del affine_mask
    del label_image

    ## Watershed to segment two close things
    distance = ndi.distance_transform_edt(noedges_thresh)

    local_max_coords = feature.peak_local_max(distance, min_distance=25)  # Lower distance = more labels separated
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True

    markers = measure.label(local_max_mask)

    segmented_cells = segmentation.watershed(-distance, markers, mask=noedges_thresh).astype(np.uint16)
    del distance
    del markers
    del noedges_thresh
    segmented_cells_resize = transform.resize(segmented_cells, fullres.shape[-3:], order = 0, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=0).astype(segmented_cells.dtype)
    writeToOMEzarr(segmented_cells_resize, zarr_filename= out_zarr_filename, t = t, c = 1, COMPRESS_THREADS=COMPRESS_THREADS)

    print(f"--- One image in {time.time() - start_time} seconds ---")

if __name__ == '__main__':
    # Reading first frame of initial data
    in_store = zarr.NestedDirectoryStore(in_zarr_filename)
    g = zarr.open_group(in_store, mode = 'r')
    fullres = g['0']  # T C Z Y X
    n_timepoint = fullres.shape[0]

    if fullres.shape[1] != 3:
        print("Not a three channel iamge - skipping, no output produced")
    else:
        print("Creating output zarr")
        if not os.path.exists(out_zarr_filename):
            createCopyOMEZarr(in_zarr_filename, out_zarr_filename)


        args = list(itertools.product(itertools.repeat(in_zarr_filename, 1),
                                      itertools.repeat(out_zarr_filename, 1),
                                      range(0, n_timepoint),
                                      itertools.repeat(COMPRESS_THREADS, 1)
                                      ))
        print("Beginning multithreaded run")
        with Pool(processes=RUN_THREADS) as pool:
            pool.starmap(thresholdSingleChannel, args, chunksize=2)

        with open(complete_filename, "w") as f:
            pass