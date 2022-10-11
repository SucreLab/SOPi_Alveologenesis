import sys

# == Custom helper functions == #
import zarr
import os
import time
from skimage import exposure
from skimage import transform
from cellpose import models
import numpy as np

COMPRESSION_LEVEL = 4
COMPRESS_THREADS = int(os.cpu_count())
RUN_THREADS = 1 # Limited by n-gpu
BATCH_SIZE = 8  # 8 is the default. Expect ~1gb / batch.

import logging
transforms_logger = logging.getLogger(__name__)
transforms_logger.setLevel(logging.DEBUG)

def createNonOMEZarrLabels(in_zarr_filename, out_zarr_filename):
    from numcodecs import Blosc
    old_store = zarr.NestedDirectoryStore(in_zarr_filename)
    old_g = zarr.open_group(old_store, mode = 'r')

    # Create the new mirror zarr
    store = zarr.NestedDirectoryStore(out_zarr_filename)
    g = zarr.open_group(store, mode = 'a')

    for group_name in list(old_g.array_keys()):
        new_shape = list(old_g[group_name].shape)
        new_shape[1] = 1  # One channel
        print(new_shape)
        g.create(name=group_name,
                 shape=new_shape,
                 chunks=old_g[group_name].chunks,
                 dtype=old_g[group_name].dtype,
                 compressor=Blosc(cname='zstd', clevel=COMPRESSION_LEVEL)
                 )

def processWriteNuclei(zarr_data, out_zarr_filename, model, t = 0, nuclei_channel = 0):
    out_store = zarr.NestedDirectoryStore(out_zarr_filename)
    g_out = zarr.open_group(out_store, mode = 'a')

    single_frame = zarr_data[t, nuclei_channel, :]
    if np.sum(single_frame) == 0:
        raise ValueError(f"Encountered a blank frame at timepoint {t} - check source data")
    kernel_size = (single_frame.shape[0] // 2, # z
                   single_frame.shape[1] // 2, # y
                   single_frame.shape[2] // 2) # x
    kernel_size = np.array(kernel_size)


    # Need to use adaptive equalization to make the illumination at the edges more or less equal
    # to the rest of the volume
    single_frame_8bit = exposure.rescale_intensity(single_frame,out_range=(0, 2**8 - 1)).astype(np.uint8)
    equalized_8bit = exposure.equalize_adapthist(single_frame_8bit, kernel_size = kernel_size, clip_limit = 0.02, nbins = 256)

    print("Evaluating model")
    # for group '1' diameter = 10
    masks = model.eval(equalized_8bit, channels=[0, 0], diameter = 9, do_3D=True, resample = False, batch_size = BATCH_SIZE)[0]

    print("Writing outputs")
    g_out['1'][t, 0, :, :, :] = masks

    fullres_shape = g_out['0'].shape
    scaleup_masks = transform.rescale(masks, (2, 2, 2), order = 0, anti_aliasing=False, preserve_range=True).astype(np.uint16)
    # Pad if not exact size
    pad_size = [(0, 0), (0, 0), (0, 0)]
    for ax in [-1, -2, -3]:
        if scaleup_masks.shape[ax] < fullres_shape[ax]:
            pad_size[ax] = (fullres_shape[ax] - scaleup_masks.shape[ax], 0)
    if sum([a + b for a, b in pad_size]) != 0:
        print(f"Padding: {pad_size}")
        scaleup_masks = np.pad(scaleup_masks, pad_size)


    g_out['0'][t, 0, :, :, :] = scaleup_masks[0:fullres_shape[-3], 0:fullres_shape[-2], 0:fullres_shape[-1]]  # One px crop

    # Resize without messing up the labels
    for downsample_key in g_out.keys():
        if downsample_key in ['0', '1']:
            continue
        new_size = g_out[downsample_key][t, 0, :, :, :].shape[-3:]
        downsampled_data = transform.resize(masks,
                                            new_size, preserve_range=True,
                                            anti_aliasing=False, order = 0).astype(np.uint16)
        g_out[downsample_key][t, 0, :, :, :] = downsampled_data

    # Old code that messed up the local means
    # g_out['2'][t, 0, :, :, :] = transform.downscale_local_mean(masks, (2, 2, 2)).astype(np.uint16)
    # g_out['3'][t, 0, :, :, :] = transform.downscale_local_mean(masks, (4, 4, 4)).astype(np.uint16)

    return True

if __name__ == '__main__':
    # Setup argument parsing on initialization

    import argparse
    parser = argparse.ArgumentParser(description="Run cellpose on drift corrected image")
    group = parser.add_argument_group('Required')
    group.add_argument('--in_path', help="Input zarr file to run", required=True)
    args = parser.parse_args()

    in_path = args.in_path
    if not args.in_path.endswith(".zarr"):
        raise RuntimeError(f"Input path must end in .zarr: {args.in_path}")
    out_zarr_filename = args.in_path[:-5] + "_cellpose.zarr"


    print(f"Out dir will be {out_zarr_filename}")


    # 1 - read the input to get metadata
    in_store = zarr.NestedDirectoryStore(in_path)
    g = zarr.open_group(in_store, mode = 'r')
    fullres = g['0']  # T C Z Y X
    halfres = g['1']  # T C Z Y X
    n_timepoint = fullres.shape[0]
    if not os.path.exists(out_zarr_filename):
        createNonOMEZarrLabels(in_path, out_zarr_filename)


    print("Initializng model")
    model = models.Cellpose(gpu=True, model_type='nuclei', net_avg=False)

    for t in range(0, n_timepoint):
        print (f"Starting timepoint {t}")
        start_time = time.time()

        processWriteNuclei(halfres, out_zarr_filename, model, t)

        print(f"--- One image in {time.time() - start_time} seconds ---")

    with open(str(in_path)[:-5] + "_cellpose_complete.txt", "w") as f:
        pass


    # 2 - make output zarr with same x, y z, but only one c

    # Start processing one by one

    # Scale up 2x - and write all of the other resolutions
