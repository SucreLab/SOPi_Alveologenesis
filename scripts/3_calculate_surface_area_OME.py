import sys
# == Custom helper functions == #
import zarr
import os
from multiprocessing import Pool
import time
import dask.array
from skimage import measure

import pandas as pd
import numpy as np

import itertools
THREADS_PER_PROCESS = 1
COMPRESS_THREADS = 1
RUN_THREADS = 4



RUN_MAX_REDUCT = 1 # What is the maximum scaledown to run, 1 is a good number
if len(sys.argv) < 2:
    raise ValueError("Must supply in zarr name")

in_zarr_filename = sys.argv[1]
if not in_zarr_filename.endswith("_segment.zarr"):
    raise RuntimeError(f"Input path must end in _segment.zarr: {in_zarr_filename}")

output_dir = in_zarr_filename[:-5] + "surface_area/"
begin_timept = 0
end_timept = False


print(f"Out dir will be {output_dir}")

def createSurface(in_zarr_filename, out_foldername, t):
    start_time = time.time()
    print(f"Starting calculation on {t}")

    for down in range(0, RUN_MAX_REDUCT + 1):
        # if sfactor > down:  # If we need to make the numbers bigger - processing a higher res than locations picked at
        #     sfactor_adj = 2 ** sfactor
        # elif sfactor < down:  # If we need to make the numbers smaller - processing a higher res than locations picked at
        #     sfactor_adj = 1 / (2 ** down - sfactor + 1)
        # else:  # Running on the same downsample factor as locations chose
        #     sfactor_adj = 1
        #
        global image_slice
        # image_slice = np.index_exp[round((z_center - size_halfwidth) * sfactor_adj):round((z_center + size_halfwidth) * sfactor_adj),
        #               round((y_center - size_halfwidth) * sfactor_adj): round((y_center + size_halfwidth) * sfactor_adj),
        #               round((x_center - size_halfwidth) * sfactor_adj): round((x_center + size_halfwidth) * sfactor_adj)]
        # print(f"Sample used to find images: {sfactor}. Running {down}. Adjusting locations by {sfactor_adj}x -- slice {image_slice}")
        image_slice = np.index_exp[:, :, :]
        print(f"Running {down} downsample.")
        createSurfaceDownsample(in_zarr_filename, out_foldername, t, down)

    print(f"--- One image {t} in {time.time() - start_time} seconds ---")

def createSurfaceDownsample(in_zarr_filename, out_foldername, t, downsample_group = 0):
    ### Start here
    store = zarr.NestedDirectoryStore(in_zarr_filename)
    g = zarr.open_group(store, mode = 'r')
    da = dask.array.from_zarr(g[str(downsample_group)]) # T C Z Y X
    single_frame = da[t, 0, :].compute()

    ## == Mask out the afined edges == ##
    img_t, img_c, img_z, img_y, img_x = da.shape

    pad_clip_px = 4

    # Make a triangle mask
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

    # Mask off the top and bottom for non-solid top and bottom
    affine_mask[:, :pad_clip_px, :] = False # Z Y X
    affine_mask[:,-pad_clip_px:, :] = False

    # Mask off sides for non-solid sides
    affine_mask[:, :, :pad_clip_px] = False # Z Y X
    affine_mask[:, :, -pad_clip_px:] = False

    #single_frame[affine_mask] = 0
    # Slice the Y
    sliced_mask = affine_mask[image_slice]
    sliced_da = single_frame[image_slice]

    # Surround everything by one px of darkness for the edge finding to work
    # If the goal is to not have a solid border, remove this.
    # sliced_mask = np.pad(sliced_mask, ((2, 2), (2, 2), (2, 2)), constant_values=1).astype(bool)
    # sliced_da = np.pad(sliced_da, ((2, 2), (2, 2), (2, 2)), constant_values=0)

    verts, faces, normals, values = measure.marching_cubes(sliced_da.astype(bool), mask= sliced_mask)
    surface_area = measure.mesh_surface_area(verts, faces)

    print(f"Surface found on {t}")
    outfolder_timept = str(out_foldername + str(t))
    if not os.path.exists(outfolder_timept):
        print(f"Creating {outfolder_timept}")
        os.makedirs(outfolder_timept)
    if downsample_group == 0:
        np.save(str(outfolder_timept + "/" + "verts.npy"), verts)
        np.save(str(outfolder_timept + "/" + "faces.npy"), faces)
        np.save(str(outfolder_timept + "/" + "normals.npy"), normals)
        np.save(str(outfolder_timept + "/" + "values.npy"), values)
        with open(str(outfolder_timept + "/" + "surface_area.txt"), "w") as f:
            f.write(str(surface_area))
    else:
        np.save(str(outfolder_timept + "/" + f"verts_{downsample_group}x_reduce.npy"), verts)
        np.save(str(outfolder_timept + "/" + f"faces_{downsample_group}x_reduce.npy"), faces)
        np.save(str(outfolder_timept + "/" + f"normals_{downsample_group}x_reduce.npy"), normals)
        np.save(str(outfolder_timept + "/" + f"values_{downsample_group}x_reduce.npy"), values)
        with open(str(outfolder_timept + "/" + f"surface_area_{downsample_group}x_reduce.txt"), "w") as f:
            f.write(str(surface_area))


def readNpySurfData(t):
    outfolder_timept = str(output_dir + str(t))
    verts = np.load(str(outfolder_timept + "/" + "verts.npy"))
    faces = np.load(str(outfolder_timept + "/" + "faces.npy"))
    return (verts, faces)

def readNpySurfDataReduce(t, reduct):
    outfolder_timept = str(output_dir + str(t))
    verts = np.load(str(outfolder_timept + "/" + "verts.npy"))
    faces = np.load(str(outfolder_timept + "/" + "faces.npy"))
    return (verts, faces)


if __name__ == '__main__':
    # Reading first frame of initial data
    in_store = zarr.NestedDirectoryStore(in_zarr_filename)
    g = zarr.open_group(in_store, mode = 'r')
    fullres = g['1']  # T C Z Y X
    n_timepoint = fullres.shape[0]
    if end_timept == False:
        end_timept = n_timepoint

    args = list(itertools.product(itertools.repeat(in_zarr_filename, 1),
                                  itertools.repeat(output_dir, 1),
                                  range(begin_timept, end_timept)
                                  ))
    print("Beginning multithreaded run")
    with Pool(processes=RUN_THREADS) as pool:
        pool.starmap(createSurface, args)

    import glob
    from natsort import natsorted

    timepoint_dirs = natsorted(glob.glob(output_dir + "/[0-9]*"))

    i = 0
    with open(output_dir + "/compiled_surface_area.txt", "w") as f_out:
        for dir in timepoint_dirs:
            with open(dir + "/surface_area.txt", "r") as f_in:
                f_out.write(str(i) + "," + str(f_in.read()) + "\n")
            i += 1

    # Compile verticies and faces

    ### Save verts and triangles csv
    print("Compiling fullres verts and faces")



    print("Reading verts and faces")
    with Pool(processes=RUN_THREADS) as pool:
        out = pool.map(readNpySurfData, range(begin_timept, end_timept))

    with open(str(output_dir + "/vertices.csv"), "w") as f_out:
        f_out.write("time,z,y,x\n")

    for timept in range(len(out)):
        print(f"Verts timept {timept}")
        vert_ndarray = out[timept][0]
        time_array = np.full((vert_ndarray.shape[0]), timept)
        out_array = np.insert(vert_ndarray, 0, time_array, axis=1)  # Add time column
        pd.DataFrame(out_array).to_csv(str(output_dir + "/vertices.csv"), header=None, index=None, mode='a')

    with open(str(output_dir + "/triangles.csv"), "w") as f_out:
        f_out.write("time,triangle_id,z,y,x\n")

    print("Compiling faces")

    timept_adjust = 0
    for timept in range(len(out)):
        print(f"Faces timept {timept}")
        vert_ndarray, faces_ndarray = out[timept][0], out[timept][1]

        vert_by_faces = vert_ndarray[faces_ndarray]
        vert_long_array = np.reshape(vert_by_faces, (vert_by_faces.shape[0] * 3, 3), order='C')

        idx_array = np.repeat(list(range(timept_adjust, timept_adjust + faces_ndarray.shape[0])), 3)

        out_array_vert = np.insert(vert_long_array, 0, idx_array, axis=1)  # Add vertex IDs


        time_array = np.full((vert_long_array.shape[0]), timept)
        out_array = np.concatenate((time_array[:,np.newaxis], out_array_vert), axis = 1)  # Add time column
        timept_adjust = timept_adjust + out_array.shape[0]
        pd.DataFrame(out_array).to_csv(str(output_dir + "/triangles.csv"), header=None, index=None, mode='a')



    print("Compiling downsampled verts and faces")

    for down in range(1, RUN_MAX_REDUCT + 1):
        out = []
        for t in range(begin_timept, end_timept):
            outfolder_timept = str(output_dir + str(t))
            verts = np.load(str(outfolder_timept + "/" + f"verts_{down}x_reduce.npy"))
            faces = np.load(str(outfolder_timept + "/" + f"faces_{down}x_reduce.npy"))
            out.append((verts, faces))

        with open(str(output_dir + f"/vertices_{down}x_reduce.csv"), "w") as f_out:
            f_out.write("time,z,y,x\n")
        for timept in range(len(out)):
            print(f"Verticies timept {timept} down {down}")
            vert_ndarray = out[timept][0]
            time_array = np.full((vert_ndarray.shape[0]), timept)
            out_array = np.insert(vert_ndarray, 0, time_array, axis=1)  # Add time column
            pd.DataFrame(out_array).to_csv(str(output_dir + f"/vertices_{down}x_reduce.csv"), header=None, index=None, mode='a')


        with open(str(output_dir + f"/triangles_{down}x_reduce.csv"), "w") as f_out:
            f_out.write("time,triangle_id,z,y,x\n")
        timept_adjust = 0
        for timept in range(len(out)):
            print(f"Faces timept {timept} down {down}")
            vert_ndarray, faces_ndarray = out[timept][0], out[timept][1]

            vert_by_faces = vert_ndarray[faces_ndarray]
            vert_long_array = np.reshape(vert_by_faces, (vert_by_faces.shape[0] * 3, 3), order='C')

            idx_array = np.repeat(list(range(timept_adjust, timept_adjust + faces_ndarray.shape[0])), 3)

            out_array_vert = np.insert(vert_long_array, 0, idx_array, axis=1)  # Add vertex IDs


            time_array = np.full((vert_long_array.shape[0]), timept)
            out_array = np.concatenate((time_array[:,np.newaxis], out_array_vert), axis = 1)  # Add time column
            timept_adjust = timept_adjust + out_array.shape[0]
            pd.DataFrame(out_array).to_csv(str(output_dir + f"/triangles_{down}x_reduce.csv"), header=None, index=None, mode='a')

