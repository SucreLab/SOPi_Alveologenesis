import os
import zarr
# == Custom helper functions == #
from helpers import readND2Img, affineImage, createOMEzarrND2, extendOMEzarr, calcZIsoFactorND2, writeToOMEzarr
from config import ROT_90, MAX_DOWNSAMPLE_FACTOR, COMPRESSION_LEVEL

import time
import glob
import os
import subprocess
from natsort import natsorted
USE_ND2 = True
# This script takes a path full of nd2 files and turns them into OME zarr files in the out_dir
# the tmp_dir needs to have > 1tb of space available for the temporary non-compressed files

def processAndWrite(nd2_img_filepath, z_iso_factor, zarr_filename, timept, channel, view, time_offset):
    print(f"Running time point {timept}, channel {channel}, view {view}")
    time_start = time.time()
    ## Affine each frame and save
    data = readND2Img(nd2_img_filepath, timepoint=timept, channel=channel, view=view)
    affined_data = affineImage(data, z_iso_factor, shape_only = False)
    writeToOMEzarr(affined_data, zarr_filename, t = timept, c = channel, time_offset=time_offset)
    print(f"One timepoint and channel in {time.time() - time_start}s")


if __name__ == '__main__':
    # Setup argument parsing on initialization

    import argparse
    parser = argparse.ArgumentParser(description="Affine .nd2.zst files")
    group = parser.add_argument_group('Required')
    group.add_argument('--tmp_dir', help="Location for temporary files, needs > 1tb", required=True)
    group.add_argument('--in_path', help="Input directory, loaction of the *.nd2.zst files", required=True)
    group_opt = parser.add_argument_group('Optional')
    group_opt.add_argument('--out_dir', help="Output directory for affined output files, "
                                          "will be created if does not exist", default="./{in_path}/processed")
    group_opt.add_argument('--cores', help="number cores", default = os.cpu_count())
    args = parser.parse_args()


    tmp_dir = args.tmp_dir
    in_path = args.in_path
    if args.out_dir == "./{in_path}/processed":  # This is intentionall not an F-string!!!
        out_dir = in_path.rstrip("/") + "/processed"
    else:
        out_dir = args.out_dir
    n_cpu = args.cores

    # Todo: Checkpointing by specific timepoint
    in_path_clean = in_path.rstrip("/")
    in_folder_name = os.path.basename(in_path_clean)
    print(f"In path: {in_path_clean}")
    input_zst_list = natsorted([f for f in glob.glob(f"{in_path_clean}/*.zst")])
    if USE_ND2:
        input_nd2_list = natsorted([f for f in glob.glob(f"{in_path_clean}/*.nd2")])
        input_zst_list = input_nd2_list
    print(f"Input files: {input_zst_list}")
    out_filename = os.path.basename(input_zst_list[0]).rstrip(".zst")
    for zst_file in input_zst_list:
        if USE_ND2:
            # == Decompress the file to temporary dir == #
            nd2_img_filepath = zst_file  # In this case.. this is an nd2
            # Skip file if already done
            completion_filename = f"{zst_file.rstrip('.nd2')}_complete.txt"
        else:
            # == Decompress the file to temporary dir == #
            nd2_img_filepath = os.path.join(tmp_dir, os.path.basename(zst_file).rstrip(".zst"))
            # Skip file if already done
            completion_filename = f"{zst_file.rstrip('.zst')}_complete.txt"

        if os.path.exists(completion_filename):
            print(f"Skipping {zst_file}")
        else:
            if not os.path.exists(nd2_img_filepath):
                print(f"Decompressing {zst_file} into {tmp_dir}")
                print(f"zstd -T0 -d '{zst_file}' -o '{nd2_img_filepath}'")
                subprocess.run(f'zstd -T0 -d "{zst_file}" -o "{nd2_img_filepath}"', shell=True, check=True)
            try:
                import nd2
                import itertools
                from multiprocessing import Pool
                from functools import partial

                z_iso_factor = calcZIsoFactorND2(nd2_img_filepath)
                # Get some metadata from the image
                nd2_info = nd2.ND2File(nd2_img_filepath)
                n_scene = nd2_info.sizes.get('P', 1)
                n_t = nd2_info.sizes.get('T', 1)
                n_c = nd2_info.sizes.get('C', 1)

                # Process each view sequentially
                for view in range(0, n_scene):
                    print(f"Starting view {view}")
                    zarr_filename = f'{out_dir}/{out_filename}_xy{view}_OME.zarr'
                    print(f"Output zarr will be {zarr_filename}")
                    # Flow: If the file doesn't exist, create it and start working on the first image
                    # If the file does exist, reshape based on the new timepoint needs
                    if not os.path.exists(zarr_filename):
                        createOMEzarrND2(zarr_filename, view, nd2_img_filepath, MAX_DOWNSAMPLE_FACTOR)
                        time_offset = 0
                    else:
                        store = zarr.NestedDirectoryStore(zarr_filename)
                        g = zarr.group(store=store)
                        time_offset = int(g['0'].shape[0])
                        extendOMEzarr(zarr_filename, view, nd2_img_filepath, MAX_DOWNSAMPLE_FACTOR)

                    # nd2_img_filepath, z_iso_factor, zarr_filename, timept, channel, view, time_offset
                    args = list(itertools.product(itertools.repeat(nd2_img_filepath, 1),
                                                  itertools.repeat(z_iso_factor, 1),
                                                  itertools.repeat(zarr_filename, 1),
                                                  range(0, n_t),
                                                  range(0, n_c),
                                                  itertools.repeat(view, 1),
                                                  itertools.repeat(time_offset, 1)))
                    with Pool(processes=int(n_cpu)) as pool:
                        pool.starmap(processAndWrite, args)

                # Create file to indicate that this nd2 has been processed - this is in the tmp dir
                f = open(completion_filename, "w")
                f.close()
            except:
                print("Something has gone wrong, cleaning up!")
                raise
            finally:
                time.sleep(10)  # Wait 10 seconds for filesystem operations - otherwise the delete doesn't always work
                # == Clean up the temporary decompressed nd2 == #
                if not USE_ND2:
                    if os.path.exists(nd2_img_filepath) and os.path.exists(completion_filename):
                        os.remove(nd2_img_filepath)

    # == Create file to say the directory is done == #
    f = open(f"{in_path}/complete.txt", "w")
    f.close()

