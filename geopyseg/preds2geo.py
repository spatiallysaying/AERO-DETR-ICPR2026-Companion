#!/usr/bin/env python3

###############################################################################
#  $Id$
#
# Purpose:  Georeferenced Yolov8 predictions for Geotiffs
# Author:   Durga Prasad Dhulipudi, dgplinux@yahoo.com
#
###############################################################################
import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#os.environ['USE_PYGEOS'] = '0'
#os.environ['PROJ_LIB'] = 'C:\\ProgramData\\anaconda3\\envs\\gis\\Library\\share\\proj'
#os.environ['GDAL_DATA'] = 'C:\\ProgramData\\anaconda3\\envs\\gis\\Library\\share'
#os.environ["vipshome"]='C:\\SAI\\IIIT_Thesis\\vips-dev-8.14\\bin'


from PIL import Image
import pandas as pd
import geopandas as gpd
from shapely import geometry
from osgeo import gdal
from tqdm import tqdm
import numpy as np


import pyvips
import logging
logging.basicConfig(level=logging.WARNING)

def is_tile_almost_empty(tile, threshold=0.05):
    """
    Determine if the tile is almost empty based on the threshold percentage of non-black pixels.
    """
    # Convert the tile to a numpy array and check if the number of non-black pixels is less than the threshold.
    array = np.array(tile)
    non_black_pixels = np.count_nonzero(array > 0)  # Count non-black (non-zero) pixels
    total_pixels = array.size
    return (non_black_pixels / total_pixels) < threshold

def predict_geo_batch(model,input_geotiff,minfo,df_tiles,output_geojson):

    pyvips_image = pyvips.Image.new_from_file(input_geotiff)
    if pyvips_image.bands>3:
        pyvips_image = pyvips_image[0:3] #3 bands are allowed

    forward_transform=(minfo.ulx, minfo.scaleX, 0, minfo.uly, 0, minfo.scaleY)
    detections_list=[]
    images=[]
    tileOffests=[]
    
    for index, row in tqdm(df_tiles.iterrows()):     
        offsetX, offsetY       = row['OffsetX'], row['OffsetY']
        current_tile           = pyvips_image.crop(row['OffsetX'], row['OffsetY'], row['Width'], row['Height'])
        pil_image              = Image.fromarray(current_tile.numpy()) #Numpy array is not working in yolov8 prediction

        #if not is_tile_almost_empty(pil_image):        
        images.append(pil_image)
        tileOffests.append((offsetX, offsetY))
            
        del pil_image
        del current_tile
    
    #Predict 
    results=model.predict(images, save_conf=True, conf=0.8,verbose=False,max_det=10,half=True)
                
    #Convert predcition coordinates to Geographic coordinates    
    for result,tileOffest in zip(results,tileOffests):
        if result.masks:#Proceed only if masks are detected
            offsetX, offsetY=tileOffest
            for cls, conf,seg in zip(result.boxes.cls, result.boxes.conf,result.masks.xy):                

                pixels_x,pixels_y = map(list,zip(*seg))

                pixels_x_global=[x+offsetX for x in pixels_x] #Convert tile pixel_x to global image pixel
                pixels_y_global=[y+offsetY for y in pixels_y] #Convert tile pixel_y to global image pixel

                latlong_list=[]
                #Convert Pixel to Latlong
                for x,y in zip(pixels_x_global,pixels_y_global):
                    pixel_coord = gdal.ApplyGeoTransform(forward_transform, x, y)
                    latlong_list.append(pixel_coord)
                poly=geometry.Polygon([[lon,lat] for lon,lat in latlong_list])#Create geometry

                result_dict={}
                result_dict['class_id']=int(cls.item()) #1-D IntTensor to an integer
                result_dict['conf']=conf.item() #1-D IntTensor to an integer
                result_dict['input_image']=row['Tile']
                result_dict['geometry']=poly
                detections_list.append(result_dict)
   
    del pyvips_image
    del df_tiles
    
    df_preds=pd.DataFrame.from_dict(detections_list)
    #print(df_preds.head())
    gdf_preds = gpd.GeoDataFrame(df_preds)

    gdf_preds.to_file(output_geojson,driver='GeoJSON')          
    
    del df_preds
    del gdf_preds   
    

# Function to convert predictions to GeoDataFrame
def predictions_to_geodf(src_raster,results):

    src = gdal.Open(src_raster)
    ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
    # Explicitly convert each value to float
    ulx = float(ulx)
    xres = float(xres)
    xskew = float(xskew)
    uly = float(uly)
    yskew = float(yskew)
    yres = float(yres)
    forward_transform=(ulx, xres, xskew, uly, yskew, yres)
    del src
    result = results[0]
    detections_list=[]
    #Convert predcition coordinates to Geographic coordinates    
    if result.masks:#Proceed only if masks are detected
        for cls, conf,seg in zip(result.boxes.cls, result.boxes.conf,result.masks.xy):  
            pixels_x,pixels_y = map(list,zip(*seg))
            latlong_list=[]
            #Convert Pixel to Latlong
            for x,y in zip(pixels_x,pixels_y ):
                pixel_coord = gdal.ApplyGeoTransform(forward_transform, float(x),float(y))
                latlong_list.append(pixel_coord)
            poly=geometry.Polygon([[lon,lat] for lon,lat in latlong_list])#Create geometry
            result_dict={}
            result_dict['class_id']=int(cls.item()) #1-D IntTensor to an integer
            result_dict['conf']=conf.item() #1-D IntTensor to an integer
            result_dict['geometry']=poly
            detections_list.append(result_dict)
    df_preds=pd.DataFrame.from_dict(detections_list)
    #print(df_preds.head())
    gdf_preds = gpd.GeoDataFrame(df_preds)
    return gdf_preds    