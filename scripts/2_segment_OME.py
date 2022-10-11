import sys

# == Custom helper functions == #
from helpers import readND2Img, affineImage, createOMEzarrND2, extendOMEzarr, calcZIsoFactorND2, writeToOMEzarr, createAffineMask
from config import ROT_90, MAX_DOWNSAMPLE_FACTOR
import zarr
import numcodecs
import os
import numpy as np
from multiprocessing import Pool, shared_memory
from skimage.exposure import rescale_intensity, match_histograms
from skimage.transform import downscale_local_mean
import cv2 as cv
from skimage.filters import threshold_local
from skimage.morphology import cube, erosion
from skimage import transform
import time
from pathlib import Path

import itertools
THREADS_PER_PROCESS = 2
COMPRESS_THREADS = 2
RUN_THREADS = 28
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
if not in_zarr_filename.endswith(".zarr"):
    raise RuntimeError(f"Input path must end in .zarr: {in_zarr_filename}")
out_zarr_filename = in_zarr_filename[:-5] + "_segment.zarr"

out_complete_filename = in_zarr_filename[:-5] + "_segment_complete.txt"

# This script takes a single OME zarr and does a drift correction
# Multithreading on this is bad because it needs to be sequential by nature
# The number of threads that this script takes is configurable by COMPRESS_THREADS above
# Can be with more efficient multiprocess with something like snakemake
def _NNmatch_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    src_unique_indices = src_unique_indices.astype(np.uint16)

    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values).astype(np.uint16)
    return interp_a_values[src_unique_indices].reshape(source.shape)

def roundToOdd(f):
    return int(np.ceil(f) // 2 * 2 + 1)

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
                 compressor=old_g[group_name].compressor)


def thresholdAndOutput(in_zarr_filename, out_zarr_filename, t, COMPRESS_THREADS=1):
    CALC_ON_FULLRES = False
    out_zarr_filename_Path = Path(out_zarr_filename)
    checkpoint_folder = out_zarr_filename_Path.parent / "checkpoints" / out_zarr_filename_Path.stem
    checkpoint_file = f"{checkpoint_folder}/{t}_done.checkpoint"
    if os.path.exists(checkpoint_file):
        print(f"{t} already done, skipping")
        return
    ### Start here
    start_time = time.time()

    print(f"Starting timepoint {t}")
    in_store = zarr.NestedDirectoryStore(in_zarr_filename)
    g = zarr.open_group(in_store, mode = 'r')
    fullres = g['0']  # T C Z Y X
    halfres = g['1']  # T C Z Y X
    if CALC_ON_FULLRES == True:
        if fullres.shape[1] > 2:
            frame_c1 = np.squeeze(fullres[t, 1, :]).astype(np.uint16)
            frame_c2 = np.squeeze(fullres[t, 2, :]).astype(np.uint16)
            # cv2_match_histograms used here because it is faster and more memory efficient
            # Use the one from skimage for normal operations
            matched_c2 = _NNmatch_cumulative_cdf(frame_c2, frame_c1).astype(np.uint16)
            max_2c = np.maximum(frame_c1, matched_c2)
        else:
            frame_c1 = np.squeeze(fullres[t, 1, :]).astype(np.uint16)
            print("Only two channels")
            max_2c = frame_c1
        threshold = threshold_local(max_2c, roundToOdd(frame_c1.shape[-1] / 2), offset = 0)  # Greater negative number here causes more 'True' pixels
    else:
        #print("Calculating halfres threshold")
        if halfres.shape[1] > 2:
            frame_c1 = np.squeeze(halfres[t, 1, :]).astype(np.uint16)
            frame_c2 = np.squeeze(halfres[t, 2, :]).astype(np.uint16)
            # cv2_match_histograms used here because it is faster and more memory efficient
            # Use the one from skimage for normal operations
            matched_c2 = _NNmatch_cumulative_cdf(frame_c2, frame_c1).astype(np.uint16)
            max_2c = np.maximum(frame_c1, matched_c2)
        else:
            frame_c1 = np.squeeze(halfres[t, 1, :]).astype(np.uint16)
            print("Only two channels")
            max_2c = frame_c1
        threshold = threshold_local(max_2c, roundToOdd(frame_c1.shape[-1] / 2), offset = 0)  # Greater negative number here causes more 'True' pixels



    #print(threshold.shape)
    # threshold = transform.resize(threshold, fullres.shape[-3:])
    # applied_threshold = _NNmatch_cumulative_cdf(fullres[t, 2, :], fullres[t, 1, :]).astype(np.uint16) > threshold

    applied_threshold = max_2c > threshold
    applied_threshold = transform.resize(applied_threshold, fullres.shape[-3:], preserve_range=True, anti_aliasing=False)
    eroded_thresh = erosion(applied_threshold, footprint = cube(4))
    writeToOMEzarr(eroded_thresh, out_zarr_filename, t, c = 0, COMPRESS_THREADS=COMPRESS_THREADS)

    file_sum = np.sum(eroded_thresh)
    with open(checkpoint_file, "w") as f:
        print(f"Writing checkpoint file: {checkpoint_file}, filesum = {file_sum}")
        f.write(str(file_sum))
    print(f"--- One image in {time.time() - start_time} seconds ---")
    return

if __name__ == '__main__':
    # Reading first frame of initial data
    in_store = zarr.NestedDirectoryStore(in_zarr_filename)
    g = zarr.open_group(in_store, mode = 'r')
    fullres = g['0']  # T C Z Y X
    n_timepoint = fullres.shape[0]

    print("Creating output zarr")
    if not os.path.exists(out_zarr_filename):
        createCopyOMEZarr(in_zarr_filename, out_zarr_filename)

    out_zarr_filename_Path = Path(out_zarr_filename)
    checkpoint_folder = out_zarr_filename_Path.parent / "checkpoints" / out_zarr_filename_Path.stem
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    args = list(itertools.product(itertools.repeat(in_zarr_filename, 1),
                                  itertools.repeat(out_zarr_filename, 1),
                                  range(0, n_timepoint),
                                  itertools.repeat(COMPRESS_THREADS, 1)
                                  ))
    print("Beginning multithreaded run")
    with Pool(processes=RUN_THREADS) as pool:
        pool.starmap(thresholdAndOutput, args, chunksize=2)
        #for _ in tqdm.tqdm(pool.istarmap(thresholdAndOutput, args), total=len(args)):
            #pass

    # Reshape the output zarr and adjust metadata
    with open(out_complete_filename, "w") as f:
        pass

