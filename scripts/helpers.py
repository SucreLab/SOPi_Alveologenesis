from nd2reader import ND2Reader
from scipy.ndimage import affine_transform
import numpy as np
import zarr
from config import ROT_90, MAX_DOWNSAMPLE_FACTOR, COMPRESSION_LEVEL

# For v2 sopi: ROT_90 = False, REVERSE_ORDER = True
# For v1 sopi: ROT_90 = True, REVERSE_ORDER = False
ROT_90 = True
REVERSE_ORDER = False


_DEBUG_START_DOWNSAMPLE = 3

def roundToOdd(f):
    return int(np.ceil(f) // 2 * 2 + 1)


def multiShiftAlongAxis(array: np.ndarray, shift_tuple: tuple = (0), constant_values: int = 0) -> np.ndarray:
    '''
    Faster way to roll multiple axes - may not return memory contiguous arrays
    :param array:
    :param shift_tuple:
    :param constant_values:
    :return:
    '''
    if len(shift_tuple) != array.ndim:
        raise ValueError(f"Must supply same number of shift axes as array dims: {len(shift_tuple)} {array.shape}")

    npad = [(0, 0)] * array.ndim
    nslice = [slice(0, dim) for dim in array.shape]

    for axis, nshift in enumerate(shift_tuple):
        nshift = int(round(float(nshift)))
        if nshift >= array.shape[axis]:
            npad[axis] = (array.shape[axis], 0)
            nslice[axis] = slice(array.shape[axis], array.shape[axis])
        elif nshift == 0:
            npad[axis] = (0, 0)
            nslice[axis] = slice(0, array.shape[axis])
        elif nshift > 0:
            npad[axis] = (nshift, 0)
            nslice[axis] = slice(0, -nshift)
        elif nshift < 0:
            npad[axis] = (0, -nshift)
            nslice[axis] = slice(-nshift, array.shape[axis])

    shifted_arr = array[tuple(nslice)]
    return np.pad(shifted_arr, pad_width=npad, mode='constant', constant_values=constant_values)

def createAffineMask(shape_z, shape_y, shape_x):
    '''
    Creates a 3D bool (1 or 0) array where 1 = image area and 0 = no image area
    :param shape_z:
    :param shape_y:
    :param shape_x:
    :return:
    '''
    img_mask = np.tri(shape_y)
    img_mask_3d = np.broadcast_to(img_mask, (shape_x, img_mask.shape[0], img_mask.shape[0])) # x, y, z
    img_mask_3d_roll = np.transpose(img_mask_3d) # z y x
    z_len_to_pad = shape_z - img_mask_3d_roll.shape[0]
    img_mask_3d_padded = np.lib.pad(img_mask_3d_roll, ((0, z_len_to_pad), (0, 0), (0, 0)))

    start_array = np.flip(img_mask_3d_padded, axis = 1)
    end_array = np.flip(img_mask_3d_padded, axis = 0)
    affine_mask = np.invert(np.logical_or(start_array, end_array))
    return(affine_mask)


def calcZIsoFactorND2(img_filename, heuristics = True):
    '''
    Given a filename, calculate the Z factor needed to make an isotropic image
    :param img_filename:
    :return:
    '''
    import nd2
    in_img_metadata = nd2.ND2File(img_filename).metadata

    z_scale = in_img_metadata.channels[0].volume.axesCalibration[2]  # X Y Z
    n_z =  in_img_metadata.channels[0].volume.voxelCount[2] # X Y Z
    z_dist_um = z_scale * (n_z - 1)

    x_scale = in_img_metadata.channels[0].volume.axesCalibration[0]
    y_scale = in_img_metadata.channels[0].volume.axesCalibration[1]
    if x_scale != y_scale:
        raise ValueError("Image is not isotropic in X and Y")

    # Heuristics for incorrectly configured microscope:
    if heuristics == True:
        if n_z == 669 and x_scale == 0.37:
            x_scale = 0.22
    z_isotropic_factor = (z_dist_um / x_scale) / n_z  # (300 um / 0.37 px/um) / 335 nz
    return z_isotropic_factor


def affineImage(data: np.ndarray, z_isotropic_factor: float, shape_only: bool = False, affine_threads = 1) -> np.ndarray:
    '''
    Take an image as a numpy array and return an affined image from SOPi microscope

    :param data: np.ndarray of image
    :param z_isotropic_factor: the ratio between the number of pixels in the z-axis, and the number of pixels
        needed in the z-axis to make an isotropic image. For example, if the scanned area is 300 um, and the um/px
        conversion is 0.37: 300 um / 0.37 = 811 px. 335 images / 811 px = 2.42.
        Basically, z_len / (z real distance / scale factor).
    :param shape_only: bool (return only shape of output array given input array)
    :return: affined nd.ndarray, z, y, x
    '''
    # Skip calculations and return a blank frame for gap frames - eliminate later
    if (np.isnan(np.sum(data)) or np.sum(data) == 0) and not shape_only:
        print("Potentially blank image blank image")
        z_len = data.shape[0]
        y_len = data.shape[1]
        x_len = data.shape[2]

        # Z dimension is scaled to make it isotropic
        # This is because 0.37 um / px on that scope
        # Target number of px is 300 um / 0.37 = 811 px
        # Scale to go from 335 slices to 811 px = 2.42
        rad_2 = np.sqrt(2)

        new_y_len = round(y_len * (1 / rad_2))
        new_x_len = int(x_len) # no change
        if z_isotropic_factor is None:
            raise ValueError("z_isotropic_factor was never set")
        shear_mat = np.array(((z_isotropic_factor,-1,0,new_y_len), # Z dimension
                              (0,1,0,0), # Y dimension
                              (0,0,1,0), # X
                              (0,0,0,1))) # ?

        scale_y_mat = np.array(((1,0,0,0), # Z dimension
                                (0,1 / rad_2,0,0), # Y dimension
                                (0,0,1,0), # X
                                (0,0,0,1)))

        xform_mat = np.dot(shear_mat, scale_y_mat)

        new_z_len = round(np.dot(xform_mat, ((z_len), (0), (0), (1)))[0])

        return np.zeros((new_z_len, new_y_len,  new_x_len))

    z_len = data.shape[0]
    y_len = data.shape[1]
    x_len = data.shape[2]

    # Z dimension is scaled to make it isotropic
    # This is because 0.37 um / px on that scope
    # Target number of px is 300 um / 0.37 = 811 px
    # Scale to go from 335 slices to 811 px = 2.42
    rad_2 = np.sqrt(2)

    new_y_len = round(y_len * (1 / rad_2))
    new_x_len = int(x_len) # no change
    if z_isotropic_factor is None:
        raise ValueError("z_isotropic_factor was never set")
    shear_mat = np.array(((z_isotropic_factor,-1,0,new_y_len), # Z dimension
                          (0,1,0,0), # Y dimension
                          (0,0,1,0), # X
                          (0,0,0,1))) # ?

    scale_y_mat = np.array(((1,0,0,0), # Z dimension
                            (0,1 / rad_2,0,0), # Y dimension
                            (0,0,1,0), # X
                            (0,0,0,1)))

    xform_mat = np.dot(shear_mat, scale_y_mat)

    new_z_len = round(np.dot(xform_mat, ((z_len), (0), (0), (1)))[0])
    if shape_only:
        return (new_z_len, new_y_len, new_x_len)

    # z, y, x
    data_affine = affine_transform(data.astype(np.uint16), np.linalg.inv(xform_mat),
                                   output_shape=(new_z_len, new_y_len, new_x_len),
                                   order = 0, prefilter = False)

    return data_affine



def readND2Img(nd2_file: str, timepoint: int = 0, channel: int = 0, view: int = 0, slice_start: int = 0, slice_end: int = None):
    '''
    Reads an nd2 file, or slice of an nd2 file, and returns a numpy array as Z, Y, X
    :param nd2_file: str, filename
    :param timepoint: int, timepoint
    :param channel: int, channel
    :param view: int, view (or scene)
    :param slice_start: int, where to start slice along X axis (for reducing memory on affine) whole image if undefined.
    :param slice_end: int, where to stop slice along X axis (for reducing memory on affine) whole image if undefined.
    :return: numpy array, Z, Y, X
    '''
    with ND2Reader(nd2_file) as images:
        n_t = images.sizes.get('t', 1)
        n_c = images.sizes.get('c', 1)
        n_v = images.sizes.get('v', 1)
        n_y = images.sizes.get('y', 1)

        images.bundle_axes = 'zyx'
        if n_c > 1 and n_v > 1:
            images.iter_axes = 'vct'
        elif n_c > 1 and n_v == 1:
            images.iter_axes = 'ct'
        elif n_c == 1 and n_v > 1:
            images.iter_axes = 'vt'
        elif n_c == 1 and n_v == 1:
            images.iter_axes = 't'

        v_offset = (n_t * n_c)
        c_offset = n_t

        idx = timepoint + (v_offset * view) + (c_offset * channel)
        print(f"Reading idx {idx}, view: {view}, channel: {channel}, timepoint, {timepoint}, slice_start: {slice_start}, slice_end: {slice_end}")

        if slice_end == None:
            slice_end = n_y
        if slice_end > (n_y):
            slice_end = n_y

        if slice_start == 0 and (slice_end == None or slice_end == n_y):
            img_data = images[idx]
            # y axis is mirrored with numpy coordinates for some reason
            img_data = np.flip(img_data, axis = 1)
        else:
            import gc
            try:
                full_image = images[idx]  # This mess is to quickly release the memory that was used
            except (KeyError, IndexError):  # This can happen when the file is improperly truncated
                full_image = np.zeros((images.sizes.get('z', 1), images.sizes.get('y', 1), images.sizes.get('x', 1)))

            full_image = np.flip(full_image, axis = 1)
            img_data = full_image[:,slice_start:slice_end, :].copy()
            del full_image
            gc.collect()
        if REVERSE_ORDER == True:
            img_data = np.flip(img_data, 0)
        if ROT_90 == True:
            # For v1 SOPi need to rotate 90deg. For transform, horizontal must be x axis.
            img_data = np.swapaxes(img_data,1,2)

        if np.isnan(np.sum(img_data)):
            print("Invalid data in image")
            img_data[np.isnan(img_data)] = 0

        return img_data.astype(np.uint16)

def extendOMEzarr(zarr_filename, view, in_ND2_filename, MAX_DOWNSAMPLE_FACTOR = 3):
    '''
    Add addititional timepoints to existing OME zarr
    :return:
    '''
    import nd2

    # Open zarr store
    store = zarr.NestedDirectoryStore(zarr_filename)
    g = zarr.group(store=store)

    # Get some metadata from the image
    nd2_info = nd2.ND2File(in_ND2_filename)
    #in_img_metadata = nd2_info.metadata
    n_t = nd2_info.sizes.get('T', 1)
    #n_c = nd2_info.sizes.get('C', 1)
    #z_iso_factor = calcZIsoFactorND2(in_ND2_filename)


    for sample in list(g.array_keys()):
        oldshape = g[sample].shape  # t c z y x
        newshape = (oldshape[0] + int(n_t), *oldshape[1:])  # Add 1 to fix the 'fencepost' problem
        print(f"Resizing array to {newshape}")
        g[sample].resize(newshape)


def createOMEzarrND2(zarr_filename: str, view: int, in_ND2_filename: str, MAX_DOWNSAMPLE_FACTOR: int = 3, COMPRESS_THREADS: int = 1, chunk_size: int = 300):
    '''
    Create a new OME zarr from nd2 filespec (this is used if one doesn't already exist)

    Zarr has groups for each downsample. Requires
    :return: True
    '''
    import nd2
    from numcodecs import Blosc
    import numcodecs
    import os
    numcodecs.blosc.set_nthreads(COMPRESS_THREADS)
    def makeOMEmetadataND2(basename: str, in_img_metadata, view: int = 0):
        '''
        Create the OMERO metadata dictionary
        :param basename: the base filename for the input (series)
        :param in_img_metadata: nd2 metadata object from the 'nd2' package
        :param view: which view is this for?
        :return:
        '''
        channel_dict_list = []
        for channel in in_img_metadata.channels:
            out_dict = {"active": "true",
                        #"color": str(hex(channel.channel.colorRGB))[2:],
                        "color": str('{:06x}'.format(channel.channel.colorRGB)),
                        "label": str(channel.channel.name),
                        "window": {
                            "end": 2 ** int(channel.volume.bitsPerComponentSignificant),
                            "max": 2 ** int(channel.volume.bitsPerComponentSignificant),
                            "min": 0,
                            "start": 0
                        }
                        }
            channel_dict_list.append(out_dict)

        omero_dict = {
            "name": f"{basename}_xy{view}",
            "version": 0.3,
            "channels": channel_dict_list
        }
        return omero_dict

    # Open zarr store
    store = zarr.NestedDirectoryStore(zarr_filename)
    g = zarr.group(store=store)


    # Get some metadata from the image
    nd2_info = nd2.ND2File(in_ND2_filename)
    in_img_metadata = nd2_info.metadata
    n_t = nd2_info.sizes.get('T', 1)
    n_c = nd2_info.sizes.get('C', 1)
    n_z = nd2_info.sizes.get('Z', 1)
    if ROT_90:
        n_y = nd2_info.sizes.get('X', 1)
        n_x = nd2_info.sizes.get('Y', 1)
    else:
        n_y = nd2_info.sizes.get('Y', 1)
        n_x = nd2_info.sizes.get('X', 1)
    z_iso_factor = calcZIsoFactorND2(in_ND2_filename)
    out_shape_zyx = affineImage(np.empty((n_z, n_y, n_x)), z_iso_factor, shape_only = True)


    shape = (n_t, n_c, *out_shape_zyx) # t, c, z, y ,x
    chunks = (1, 1, chunk_size * 2, chunk_size, chunk_size) # t, c, z, y ,x

    # Setup OME metadata
    g.attrs["omero"] = makeOMEmetadataND2(os.path.basename(zarr_filename), in_img_metadata, view)
    g.attrs["multiscales"] = [
        {
            "datasets": [{"path": dn} for dn in range(0,MAX_DOWNSAMPLE_FACTOR + 1)],
            "axes": ["t", "c", "z", "y", "x"],
            "version": 0.3,
            "type": "bicubic",
            "window": {
                "end": g.attrs["omero"]["channels"][0]["window"]["end"],
                "start": 0
            }
        }
    ]

    # Create the groups
    for sample in range(0,MAX_DOWNSAMPLE_FACTOR + 1):
        downscale = 2 ** sample
        if not str(sample) in list(g.array_keys()):
            if sample == 0:
                shape = shape
                chunks = chunks
            else:
                print(f"Creating zarr groups for {sample}x reduction")
                from skimage import transform
                downscaled_data = transform.downscale_local_mean(np.empty(out_shape_zyx), (downscale, downscale, downscale)).astype(np.uint16)
                shape = shape[:2] + tuple(downscaled_data.shape)
                chunks = chunks[:2] + tuple(downscaled_data.shape)

            g.create(name=sample,
                     shape=shape,
                     chunks=chunks,
                     dtype=np.uint16,
                     compressor=Blosc(cname='zstd', clevel=COMPRESSION_LEVEL))



    return True




def createOMEzarr(zarr_filename: str, out_shape: tuple, copy_metadata_fname: str = None, MAX_DOWNSAMPLE_FACTOR: int = 3, COMPRESS_THREADS: int = 1, chunk_size: int = 300):
    '''
    Create a new OME zarr from specified shape

    Zarr has groups for each downsample. Requires
    :return: True
    '''
    from numcodecs import Blosc
    import numcodecs
    numcodecs.blosc.set_nthreads(COMPRESS_THREADS)

    store = zarr.NestedDirectoryStore(zarr_filename)
    g = zarr.group(store=store)

    # Setup OME metadata
    g.attrs["multiscales"] = [
        {
            "datasets": [{"path": dn} for dn in range(0,MAX_DOWNSAMPLE_FACTOR + 1)],
            "axes": ["t", "c", "z", "y", "x"],
            "version": 0.3,
            "type": "bicubic"
        }
    ]

    if len(out_shape) != 5:
        raise RuntimeError(f"out_shape must have five dimensions: t, c, z, y, x")
    shape = list(out_shape) # t, c, z, y ,x
    chunks = (1, 1, chunk_size * 2, chunk_size, chunk_size) # t, c, z, y ,x

    for sample in range(0,MAX_DOWNSAMPLE_FACTOR + 1):
        downscale = 2 ** sample
        if not str(sample) in list(g.array_keys()):
            if sample == 0:
                group_shape = shape
                chunks = chunks
            else:
                print(f"Creating zarr groups for {sample}x reduction")
                group_shape = shape[:2] + [i // downscale for i in shape[-3:]]
                chunks = list(chunks[:2]) + group_shape[-3:]

            g.create(name=sample,
                     shape=group_shape,
                     chunks=chunks,
                     dtype=np.uint16,
                     compressor=Blosc(cname='zstd', clevel=COMPRESSION_LEVEL))

    if copy_metadata_fname is not None:
        print("Copying metadata")
        old_store = zarr.NestedDirectoryStore(copy_metadata_fname)
        old_g = zarr.group(store=old_store)
        for attr_key in old_g.attrs.keys():
            g.attrs[attr_key] = old_g.attrs[attr_key]

    return True

def writeToOMEzarr(data, zarr_filename, t, c, time_offset = 0, COMPRESS_THREADS = 1, exact_shape = True):
    '''
    Write a 3D (Z Y X) array to a zarr file, automatically downsamples
    :param data: 3d ndarray
    :param zarr_filename: filename to write to
    :param t: what timepoint
    :param c: what channel
    :param time_offset: what should the offset be for the image time
    :return:
    '''
    from skimage import transform
    import numcodecs
    numcodecs.blosc.set_nthreads(COMPRESS_THREADS)
    store = zarr.NestedDirectoryStore(zarr_filename)
    g = zarr.open_group(store, mode = 'a')
    print(f"Writing timepoint {time_offset + t} channel {c}")
    if exact_shape == False:
        pad_z, pad_y, pad_x = [a - b for a, b, in zip(g[0].shape[-3:], data.shape)]
        if any([pad_z < 0, pad_y < 0, pad_x < 0]):
            raise ValueError(f"image is bigger than container - need to re-evaluate")
        print(f"Padding {pad_z}, {pad_y}, {pad_x}")
        data = np.pad(data, ((0, pad_z), (0, pad_y), (0,pad_x)))

    for sample in list(g.array_keys()):
        sample = int(sample)
        downscale = 2 ** sample
        if sample == 0:
            g[sample][time_offset + t, c, :, :, :] = data
        else:
            #new_size = tuple([i // downscale for i in data.shape])
            downsample_size = g[sample].shape[-3:]
            downscaled_data = transform.resize(data,
                                               downsample_size, order = 0, preserve_range=True,
                                               anti_aliasing=False, anti_aliasing_sigma = 0).astype(np.uint16)


            g[sample][time_offset + t, c, :, :, :] = downscaled_data
