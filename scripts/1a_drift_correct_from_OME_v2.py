import sys

# == Custom helper functions == #
from helpers import readND2Img, affineImage, createOMEzarrND2, extendOMEzarr, calcZIsoFactorND2, writeToOMEzarr, createAffineMask, multiShiftAlongAxis
from config import ROT_90, MAX_DOWNSAMPLE_FACTOR, COMPRESSION_LEVEL
import zarr
import numcodecs
import os
import numpy as np
from multiprocessing import Pool, shared_memory
from skimage.exposure import rescale_intensity
from skimage.transform import downscale_local_mean
import cv2 as cv
import itertools
import dask.array
import csv
import time
from skimage.registration import phase_cross_correlation
from skimage.exposure import match_histograms, equalize_adapthist, rescale_intensity
from pathlib import Path
COMPRESS_THREADS = 4
RUN_THREADS = 4
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
#in_zarr_filename = "/run/user/1000/gvfs/smb-share:server=i10file.vumc.org,share=ped/NEO/Sucre Lab/Negretti/Microscopy/SOPi/2021.01.28 SOPi_P5_72h/processed/2021.01.28_SOPi_mtngjfho_p5003 - Denoised.nd2_xy2_OME.zarr"
#in_zarr_filename = "/scratch/sopi/2021.01.28_SOPi_mtngjfho_p5003 - Denoised.nd2_xy2_OME.zarr"
if not in_zarr_filename.endswith(".zarr"):
    raise RuntimeError(f"Input path must end in .zarr: {in_zarr_filename}")
out_zarr_filename = in_zarr_filename[:-5] + "_driftfix.zarr"
out_driftcsv_filename = in_zarr_filename[:-5] + "_drifts.csv"
out_complete_filename = in_zarr_filename[:-5] + "_drifts_complete.txt"

# Set this here for the parallel run
numcodecs.blosc.set_nthreads(COMPRESS_THREADS)

def createCopyOMEZarr(in_zarr_filename, out_zarr_filename):
    from numcodecs import Blosc
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
                 compressor=Blosc(cname='zstd', clevel=COMPRESSION_LEVEL)
                 #compressor=old_g[group_name].compressor
                 )

def copyFirstFrametoNewZarr(in_zarr_filename, out_zarr_filename, COMPRESS_THREADS = 1, RESCALE_RANGE = True):
    numcodecs.blosc.set_nthreads(COMPRESS_THREADS)
    old_store = zarr.NestedDirectoryStore(in_zarr_filename)
    old_g = zarr.open_group(old_store, mode = 'r')

    # Create the new mirror zarr
    store = zarr.NestedDirectoryStore(out_zarr_filename)
    g = zarr.open_group(store, mode = 'a')

    if len(list(g.array_keys())) != len(list(old_g.array_keys())):
        raise ValueError(f"Problem with the new zarr group: new {list(g.array_keys())}; old {list(old_g.array_keys())}")

    for group_name in list(g.array_keys()):
        if RESCALE_RANGE:
            for c in range(0, g[group_name].shape[1]):
                #print(f"Rescaling range c{c}")
                min_val = np.min(old_g[group_name][0, c, :][np.nonzero(old_g[group_name][0, c, :])]) # Find 'background' minimum
                tmp = cv.subtract(old_g[group_name][0, c, :], np.full(old_g[group_name][0, c, :].shape, min_val)) # Do background subtraction
                g[group_name][0, c, :] = rescale_intensity(tmp, out_range=(0, 2 ** 12 - 1)).astype(np.uint16)
        else:
            g[group_name][0, :] = old_g[group_name][0, :]



def rescaleRange(data):
    '''
    Takes a 3D data (Z, Y X) and returns values scaled from 0 to max (12 bit)
    :param data:
    :return:
    '''
    nonzero_array = data[np.nonzero(data)]
    if len(nonzero_array) >= 1:
        min_val = np.min(nonzero_array) # Find 'background' minimum
        data = cv.subtract(data, np.full(data.shape, min_val)) # Do background subtraction

    data = rescale_intensity(data, out_range=(0, 2 ** 12 - 1)).astype(np.uint16)
    return data


# Read i from new image, and i + 1 from old image (through .shape(i)), transform, write
def readFramesAndTransform(in_zarr_filename, out_zarr_filename, timepoint, ref_shm_name, ref_shape, offsets_list, COMPRESS_THREADS, HISTOGRAM_MATCHING = True, RESCALE_RANGE = True):
    '''
    Takes a shared memory array of a reference image, transforms a new image, and writes

    :param out_zarr_filename:
    :param timepoint:
    :param ref_shm_name:
    :param ref_shape:
    :param COMPRESS_THREADS:
    :param HISTOGRAM_MATCHING:
    :param RESCALE_RANGE:
    :return:
    '''

    out_zarr_filename_Path = Path(out_zarr_filename)
    checkpoint_folder = out_zarr_filename_Path.parent / "checkpoints" / out_zarr_filename_Path.stem
    checkpoint_file = f"{checkpoint_folder}/{timepoint}_done.checkpoint"
    if os.path.exists(checkpoint_file):
        print(f"{timepoint} already done, skipping")
        return


    numcodecs.blosc.set_nthreads(COMPRESS_THREADS)
    if timepoint == 0:
        raise ValueError("Do not process timepoint == 0 with this function")
    offsets = offsets_list[timepoint]
    import time
    from skimage.registration import phase_cross_correlation
    from skimage.exposure import match_histograms, equalize_adapthist, rescale_intensity

    ### Start here
    start_time = time.time()

    # Pull refdata from the shared memory
    existing_shm = shared_memory.SharedMemory(name=ref_shm_name, create=False)
    img_ref = np.ndarray(ref_shape, dtype=np.uint16, buffer=existing_shm.buf)


    # Load in the image to transform
    g = zarr.open_group(zarr.NestedDirectoryStore(in_zarr_filename), mode = 'r')
    new_fullres = g['0']  # T C Z Y X
    timepoint_data = np.squeeze(new_fullres[timepoint, :])  # results in C Z Y X

    c_len, z_len, y_len, x_len = ref_shape

    # Collect the middle third of the image, excluding the dark areas on the edge
    # Run the alignment on this - cleaner and less memory needed
    # Potential problem: The drift is bigger than this crop - however,
    # if that's the case.. probably too big to compensate for
    # Note, b/c the angle is a 45deg triangle, y_len describes the triangle
    z_min = y_len
    z_max = z_len - y_len
    z_slice = int(round(y_len - (y_len * 0.15)))
    y_slice = int(round(y_len * 0.15))

    z_len_10 = int(round(z_len * 0.1)) # for cropping out the outer 10% for memory savings and speed
    y_len_10 = int(round(y_len * 0.1))
    x_len_10 = int(round(x_len * 0.1))

    # Grab all channels from the same view - the transformation is the same
    # Returns a numpy array for everything with ':'. This means c, z, y, x

    if RESCALE_RANGE:
        print(f"Rescaling range timepoint {timepoint}")
        for c in range(c_len):
            #print(f"Rescaling range c{c}")
            timepoint_data[c, :] = rescaleRange(timepoint_data[c, :])

    if HISTOGRAM_MATCHING:
        print(f"Histogram matching timepoint {timepoint}")
        #print("Starting histogram match")
        for c in range(c_len):
            timepoint_data[c,:] = match_histograms(np.squeeze(timepoint_data[c, :]), np.squeeze(img_ref[c, :])).astype(np.uint16)

    print(f"Offsets: {offsets} at timepoint {timepoint}")
    z_off = int(round(float(offsets[0])))
    y_off = int(round(float(offsets[1])))
    x_off = int(round(float(offsets[2])))
    #print(f"Offsets numeric: {z_off}, {y_off}, {x_off} at timepoint {timepoint}")

    print("Rolling dimensions")
    timepoint_data = multiShiftAlongAxis(timepoint_data, (0, z_off, y_off, x_off))
    # timepoint_data = np.roll(timepoint_data,z_off,axis=1)  # z
    # timepoint_data = np.roll(timepoint_data,y_off,axis=2)  # y
    # timepoint_data = np.roll(timepoint_data,x_off,axis=3)  # x
    #
    # # Black out areas that were rolled over gvbh468JkLVUM!

    # if z_off > 0:
    #     timepoint_data[:,0:z_off,:,:] = 0
    # elif z_off < 0:
    #     timepoint_data[:,z_off:,:,:] = 0
    #
    # if y_off > 0:
    #     timepoint_data[:,:,0:y_off,:] = 0
    # elif y_off < 0:
    #     timepoint_data[:,:,y_off:,:] = 0
    #
    # if x_off > 0:
    #     timepoint_data[:,:,:,0:x_off] = 0
    # elif x_off < 0:
    #     timepoint_data[:,:,:,x_off:] = 0
    #print("About to write")
    for c in range(c_len):
        writeToOMEzarr(timepoint_data[c, :], out_zarr_filename, t = timepoint, c = c, COMPRESS_THREADS = COMPRESS_THREADS)

    with open(checkpoint_file, "w") as f:
        print(f"Writing checkpoint file: {checkpoint_file}")
        f.write(str(time.time()))
    print(f"--- One image in {time.time() - start_time} seconds ---")
    return


def getOffsetPair(in_zarr, i = 0):
    ref_frame = np.squeeze(in_zarr[int(i), :].astype(np.uint16))
    adjust_frame = np.squeeze(in_zarr[int(i + 1), :].astype(np.uint16))

    print(f"Timepoint {i}")

    c_len, z_len, y_len, x_len = ref_frame.shape

    # Collect the middle third of the image, excluding the dark areas on the edge
    # Run the alignment on this - cleaner and less memory needed
    # Potential problem: The drift is bigger than this crop - however,
    # if that's the case.. probably too big to compensate for
    # Note, b/c the angle is a 45deg triangle, y_len describes the triangle
    z_min = y_len
    z_max = z_len - y_len
    z_slice = int(round(y_len - (y_len * 0.15)))
    y_slice = int(round(y_len * 0.15))

    z_len_10 = int(round(z_len * 0.1)) # for cropping out the outer 10% for memory savings and speed
    y_len_10 = int(round(y_len * 0.1))
    x_len_10 = int(round(x_len * 0.1))

    ref_projection = np.maximum.reduce([ref_frame[c, z_slice:-z_slice, y_slice:-y_slice, x_len_10:-x_len_10] for c in range(c_len)])
    timepoint_projection = np.maximum.reduce([adjust_frame[c, z_slice:-z_slice, y_slice:-y_slice, x_len_10:-x_len_10] for c in range(c_len)])

    #timepoint_projection_matched = match_histograms(timepoint_projection, ref_projection).astype(np.uint16)
    detected_shift = phase_cross_correlation(ref_projection,  # Ref data
                                             timepoint_projection,
                                             overlap_ratio=0.3)  # New data
    detected_shift_scaled = [i * 2 for i in detected_shift[0]]
    return(detected_shift_scaled)

if __name__ == '__main__':

    overall_start_time = time.time()
    # Reading first frame of initial data
    old_store = zarr.NestedDirectoryStore(in_zarr_filename)
    #old_fullres = dask.array.from_zarr(old_store, component='3')
    old_g = zarr.open_group(old_store, mode = 'r')
    old_fullres = old_g['0']  # T C Z Y X
    old_halfres = old_g['1']  # T C Z Y X
    n_timepoint = old_fullres.shape[0]
    firstframe = np.squeeze(old_fullres[0, :]).astype(np.uint16)

    ## First, find the drift offsets for all i and i+1 images

    #offsets_single = getOffsetPair(old_fullres, 4)
    #print(offsets_single)
    beginning_image = 0
    final_image = n_timepoint
    #final_image = 20

    if not os.path.exists(out_driftcsv_filename):
        args = list(itertools.product(itertools.repeat(old_halfres, 1),
                                      range(beginning_image, final_image - 1)
                                      ))
        print("Calculating drift")

        start_time = time.time()
        with Pool(processes=int(round(RUN_THREADS))) as pool:
            offsets = pool.starmap(getOffsetPair, args, chunksize=2)
        print(f"--- Drifts found in {time.time() - start_time} seconds ---")
        # Add the offsets for the first image - 0, 0, 0
        offsets.insert(0, [0, 0, 0])

        cumulitive_sums = []
        for idx in range(len(offsets)):
            if len(cumulitive_sums) < 1:
                cumulitive_sums.append(list(offsets[idx]))
                continue
            z_prev, y_prev, x_prev = cumulitive_sums[idx - 1]
            z_new, y_new, x_new = offsets[idx]

            cumulitive_sums.append([z_prev + z_new, y_prev + y_new, x_prev + x_new])

        with open(out_driftcsv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(cumulitive_sums)
    else:
        print("Reading previously found drift")
        #Read the drift that has been previously been calculated
        with open(out_driftcsv_filename) as file_obj:
            reader_obj = csv.reader(file_obj)
            cumulitive_sums = list(reader_obj)


    print("Creating reference for histogram matching")
    # Rescale the first frame and store it in a shared memory array
    shm = shared_memory.SharedMemory(create=True, size=firstframe.nbytes)
    shm_array = np.ndarray(firstframe.shape, dtype=firstframe.dtype, buffer=shm.buf)
    for c in range(firstframe.shape[0]):
        firstframe[c, :] = rescaleRange(firstframe[c, :])

    shm_array[:,:,:,:] = firstframe[:,:,:,:]

    print("Creating output zarr")
    if not os.path.exists(out_zarr_filename):
        createCopyOMEZarr(in_zarr_filename, out_zarr_filename)
    print("Writing first frame to output zarr")
    for c in range(shm_array.shape[0]):
        writeToOMEzarr(shm_array[c,:], out_zarr_filename, t = 0, c = c, COMPRESS_THREADS=RUN_THREADS * COMPRESS_THREADS)
    numcodecs.blosc.set_nthreads(COMPRESS_THREADS)
    # Scale data??
    # out_zarr_filename, timepoint, ref_shm_name, ref_shape, COMPRESS_THREADS = THREADS_PER_PROCESS, HISTOGRAM_MATCHING = True, RESCALE_RANGE = True
    args = list(itertools.product(itertools.repeat(in_zarr_filename, 1),
                                  itertools.repeat(out_zarr_filename, 1),
                                  #range(260, 262),
                                  range(beginning_image + 1, final_image),
                                  itertools.repeat(shm.name, 1),
                                  itertools.repeat(firstframe.shape, 1),
                                  itertools.repeat(cumulitive_sums, 1),
                                  itertools.repeat(COMPRESS_THREADS, 1),
                                  itertools.repeat(True, 1),
                                  itertools.repeat(True, 1),
                                  ))
    # Create checkpoint folder if it doesn't exist
    out_zarr_filename_Path = Path(out_zarr_filename)
    checkpoint_folder = out_zarr_filename_Path.parent / "checkpoints" / out_zarr_filename_Path.stem
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    print("Beginning multithreaded run")
    with Pool(processes=RUN_THREADS) as pool:
        pool.starmap(readFramesAndTransform, args, chunksize=2)


    shm.unlink()

    # Write a .txt file when complete - helps with automated workflows
    with open(out_complete_filename, "w") as f:
        print("Complete")

    print(f"Total file done in {time.time() - overall_start_time}")