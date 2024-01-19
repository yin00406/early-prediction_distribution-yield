import os
from osgeo import gdal, osr
import glob
import numpy as np
import config

def mosaic_images(input_dir, file_prefix, output_dir):
    '''
    mosaic 5-day images into one image
    '''
    # Collect all the files for a given prefix
    files = glob.glob(f'{input_dir}/{file_prefix}*')
    # Mosaic the images together
    images = [gdal.Open(file) for file in files]
    mosaic = gdal.Warp('', images, format='MEM') # mosaic multiple images together in memory

    # Save the result
    output_filename = os.path.join(output_dir, f'{file_prefix}.tif')
    gdal.Translate(output_filename, mosaic, creationOptions = ["BIGTIFF=YES", "COMPRESS=DEFLATE"])

    mosaic = None
    for img in images:
        img = None

def fill_nan_values(input_dir, fileName, output_dir):
    '''
    interpolation
    '''

    mosaic = gdal.Open(os.path.join(input_dir, fileName))
    mosaic_array = mosaic.ReadAsArray()

    for pre_image_dir in sorted(glob.glob(os.path.join(input_dir, f'{state}_{year}_*.tif')), reverse=True):
        if int(pre_image_dir.split("/")[-1].split(".")[0].split("_")[-1]) < doy:
            mask = np.isnan(mosaic_array)
            pre_image = gdal.Open(pre_image_dir)
            pre_image_array = pre_image.ReadAsArray()
            mosaic_array[mask] = pre_image_array[mask]

            if np.count_nonzero(np.isnan(mosaic_array)) == 0:
                break

    # Save the result
    gt = mosaic.GetGeoTransform()
    dest = gdal.GetDriverByName('GTiff').Create(os.path.join(output_dir, fileName), mosaic.RasterXSize, mosaic.RasterYSize, 9,
                      gdal.GDT_Float32, options = ["BIGTIFF=YES", "COMPRESS=DEFLATE"])
    for band in range(9):
        dest.GetRasterBand(band + 1).WriteArray(mosaic_array[band])
    dest.SetGeoTransform(gt)
    wkt = mosaic.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dest.SetProjection(srs.ExportToWkt())
    dest = None

if __name__ == '__main__':
    # print("################################ MOSAIC")
    # Directory that contains the images
    input_dir = '/scratch.global/yin00406/early-prediction/DATA'
    # Directory to save the output images
    output_dir = f'{input_dir}/mosaic'

    # Create mosaic for each state and doy
    grids = config.grids_NE[3:4] #c
    years = config.year_list[:] #c
    print("################################ MOSAIC")
    for state in grids:
        for year in years:
            if year == 2020:
                for extended_doy in config.extended_doys_CS_leap:
                    mosaic_images(os.path.join(input_dir, state), f"{state}_{year}_{extended_doy}", os.path.join(output_dir, state))
                    print(f"done : {state}_{str(year)}_{extended_doy}")
            else:
                for extended_doy in config.extended_doys_CS:
                    mosaic_images(os.path.join(input_dir, state), f"{state}_{year}_{extended_doy}", os.path.join(output_dir, state))
                    print(f"done : {state}_{str(year)}_{extended_doy}")

    print("################################ INTERPOLATION")
    for state in grids:
        for year in years:
            if year == 2020:
                for doy in config.doys_CS_leap:
                    fill_nan_values(os.path.join(input_dir, "mosaic", state), f"{state}_{year}_{doy}.tif", os.path.join(input_dir, "interpolation", state))
                    print(f"done : {state}_{year}_{doy}")
            else:
                for doy in config.doys_CS:
                    fill_nan_values(os.path.join(input_dir, "mosaic", state), f"{state}_{year}_{doy}.tif", os.path.join(input_dir, "interpolation", state))
                    print(f"done : {state}_{year}_{doy}")
