import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal 
#import fiona

import rasterio
import rasterio.plot
from rasterio.mask import mask
from rasterio.windows import Window
from rasterio.windows import from_bounds

from shapely import geometry
from shapely.affinity import scale,rotate
from shapely.geometry import MultiPolygon,Polygon, LineString,box,Point
from shapely.ops import unary_union, linemerge

from affine import Affine  # For easly manipulation of affine matrix

import matplotlib.pyplot as plt

from geopyseg.retile_processor import *


""" def save_geopandas(gdf, filepath):
    try:
        gdf.to_file(filepath) 
    except:
        with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
            gdf.to_file(filepath)   """

def get_obb_rotation(xywhr):
    # Extract the angle in radians and convert it to degrees
    angle_radians = xywhr[-1]  # Assuming the last item is the rotation angle in radians
    angle_degrees = torch.rad2deg(angle_radians).item()  # Convert radians to degrees and get as Python number

    return angle_degrees
    
def pixel_to_geo(input_raster, pixels_list):
    row_list = []
    
    with rasterio.open(input_raster) as src:
        pX_List = [pt[0] for pt in pixels_list]
        pY_List = [pt[1] for pt in pixels_list]
        lons, lats = rasterio.transform.xy(src.transform, pY_List, pX_List)
        lons, lats = np.array(lons), np.array(lats)

        vertices_list = [(lon, lat) for lon, lat in zip(lons, lats)]
        vertices_list.append(vertices_list[0])  # Close the polygon
        polygon_geom = Polygon(vertices_list)

        geom_dict = {'geometry': polygon_geom}
        row_list.append(geom_dict)

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(row_list, crs=src.crs)
    
    return gdf
    
def get_bbox(ds):
    """Return list of corner coordinates from a GDAL Dataset."""
    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel

    return xmin,ymin,xmax,ymax
    
def get_center(dataset):
    """This function return the CRS coordinates of the raster center 
    """
    info = gdal.Info(dataset, format='json')
    return  info['cornerCoordinates']['center']
    
# Some functions declaration for clarify the code
def raster_center_in_pixels(raster):
    """This function return the pixel coordinates of the raster center 
    """
    # We get the size (in pixels) of the raster
    # using gdal
    width, height = raster.RasterXSize, raster.RasterYSize

    # We calculate the middle of raster
    xmed = width / 2
    ymed = height / 2

    return (xmed, ymed)

def rotate_gt(affine_matrix, angle, pivot=None):
    """This function generate a rotated affine matrix
    """
    # The gdal affine matrix format is not the same
    # of the Affine format, so we use a bullit-in function
    # to change it
    # see : https://github.com/sgillies/affine/blob/master/affine/__init__.py#L178
    affine_src = Affine.from_gdal(*affine_matrix)
    # We made the rotation. For this we calculate a rotation matrix,
    # with the rotation method and we combine it with the original affine matrix
    # Be carful, the star operator (*) is surcharged by Affine package. He make
    # a matrix multiplication, not a basic multiplication
    affine_dst = affine_src * affine_src.rotation(angle, pivot)
    # We retrun the rotated matrix in gdal format
    return affine_dst.to_gdal()
        
    
def gdal_clip_raster_with_vector(input_raster, vector_mask, output_raster):
    """
    Clips a raster file to the extents of a vector mask file using GDAL.

    Args:
        input_raster (str): The path to the input raster file.
        vector_mask (str): The path to the vector file used as a mask.
        output_raster (str): The path where the output clipped raster will be saved.
    """
    # Open the input raster
    src_ds = gdal.Open(input_raster)
    if src_ds is None:
        print("Unable to open input raster file.")
        return

    # Open the mask layer
    vector_ds = ogr.Open(vector_mask)

    if vector_ds is None:
        print("Unable to open mask vector file.")
        return

    # Use gdal.Warp to perform the clipping
    
    #warp_options = gdal.WarpOptions(cutlineDSName=vector_ds.GetName(), 
    #                                cropToCutline=True,
    #                                dstNodata=src_ds.GetRasterBand(1).GetNoDataValue())
    
    warp_options = gdal.WarpOptions(cutlineDSName=str(vector_mask), 
                                    cropToCutline=True,
                                    dstNodata=src_ds.GetRasterBand(1).GetNoDataValue())
    result = gdal.Warp(output_raster, src_ds, options=warp_options)
    
    # Check if clipping was successful
    if result is None:
        print("Clipping failed")
    # else:
    #     print("Clipping successful, output saved to", output_raster)

    # Clean up
    del src_ds, vector_ds, result


def rotate_and_store_gdf(gdf, angle, pivot):
    """Rotates all geometries in the given GeoDataFrame by the specified angle around a pivot and returns a new GeoDataFrame with rotated geometries."""
    rotated_geometries = [rotate(geom, angle, origin=pivot) for geom in gdf.geometry]
    new_gdf = gpd.GeoDataFrame(gdf.copy(), geometry=rotated_geometries, crs=gdf.crs)
    return new_gdf
    
def rotate_image(input_path, output_path, angle):
    """
    Rotates the raster image at the given pivot point by the specified angle.
    Parameters:
        input_path (str): Path to the input raster file.
        output_path (str): Path to save the rotated raster file.
        angle (float): Rotation angle in degrees, positive counter-clockwise.
        pivot (tuple): Pivot point (x, y) for the rotation in georeferenced coordinates.
                       If None, the center is used.
    """
    # Open the source dataset
    src_ds = gdal.Open(input_path)
    if src_ds is None:
        print("The specified file does not exist or could not be opened.")
        raise FileNotFoundError("The specified file does not exist or could not be opened.")

    # Get the original geo-transform and projection
    gt = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    center = raster_center_in_pixels(src_ds)
   
    # Warp options
    warp_options = gdal.WarpOptions(format='GTiff',
                                    outputType=gdal.GDT_Byte,
                                    resampleAlg=gdal.GRA_Lanczos ,
                                    srcSRS=projection,
                                    dstSRS=projection,
                                    dstNodata=src_ds.GetRasterBand(1).GetNoDataValue(),
                                    )

    # Apply the rotation using gdal.Warp
    dst_ds = gdal.Warp(output_path, src_ds, options=warp_options)
    
    if dst_ds is None:
        print("Failed to create the output file.",output_path)
        raise RuntimeError("Failed to create the output file.")
        
    dst_ds.SetGeoTransform(rotate_gt(gt, angle, center))

    dst_ds.FlushCache()
    # Clean up
    del src_ds, dst_ds
 
def view_vector_overlay(gdf1,gdf2):
    fig, ax = plt.subplots(figsize=(5, 5))
    gdf1.plot(ax=ax, facecolor='none', edgecolor='red')
    gdf2.plot(ax=ax, facecolor='none', edgecolor='blue')

def view_raster_vector_overlay(raster_path,vector_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    df = gpd.read_file(vector_path)
    raster = rasterio.open(raster_path)
    rasterio.plot.show(raster, ax=ax)
    df.plot(ax=ax, facecolor='none', edgecolor='red')
    ax.axis('off')  # Turn off axes
    
    del raster
    del df

def view_raster_gdf_overlay(raster, gdf, figsize=(4, 4), edgecolor='red', facecolor='none', title=None):
    """
    Display a raster with a GeoDataFrame overlay.
    
    Parameters:
    - raster: An open rasterio dataset or a path string to the raster file
    - gdf: A GeoDataFrame containing vector geometries to overlay
    - figsize: Tuple specifying figure size (width, height)
    - edgecolor: Color of the geometry edges
    - facecolor: Fill color of the geometries
    - title: Optional title for the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle both rasterio dataset and path string
    if isinstance(raster, str):
        raster_ds = rasterio.open(raster)
        close_raster = True
    else:
        raster_ds = raster
        close_raster = False
    
    rasterio.plot.show(raster_ds, ax=ax)
    gdf.plot(ax=ax, facecolor=facecolor, edgecolor=edgecolor)
    if title:
        ax.set_title(title)
    ax.axis('off')  # Turn off axes
    
    if close_raster:
        raster_ds.close()
        
def view_raster(raster_path):
    raster = rasterio.open(raster_path)
    rasterio.plot.show(raster)
    del raster

# Function to calculate azimuth between two points
def calculate_azimuth(linestring):
    if len(linestring.coords) > 1:
        start, end = linestring.coords[0], linestring.coords[-1]
        angle_rad = np.arctan2(end[0] - start[0], end[1] - start[1])
        angle_deg = np.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        return angle_deg
    return None
    
def normalize_angle(angle):
    if angle is None:
        return 0.0  # Default angle for None values
    return angle % 360


def compute_rotation_angle(azimuth):
    if azimuth is None:
        return 0.0  # Default rotation angle for invalid geometries
    azimuth = normalize_angle(azimuth)
    rotation_angle = 90 - azimuth
    return normalize_angle(rotation_angle)    


def split_multipolygon(row):
    # Splits a MultiPolygon geometry into individual Polygon geometries.
    # Parameters:
    #   row (GeoSeries/GeoDataFrame Row): A row containing a MultiPolygon geometry.
    # Returns:
    #   list: A list of dictionaries, each representing a row with a single Polygon geometry.

    geom = row.geometry
    if isinstance(geom, MultiPolygon):
        # Iterate over each Polygon in the MultiPolygon
        return [row.drop('geometry').to_dict() | {'geometry': poly} for poly in geom.geoms]
    else:
        return [row.to_dict()]

        
def midpoint(p1, p2):
    # Computes the midpoint between two points.
    # Parameters:
    #   p1 (tuple): Coordinates of the first point (x, y).
    #   p2 (tuple): Coordinates of the second point (x, y).
    # Returns:
    #   tuple: Coordinates of the midpoint (x, y).

    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
    
def calculate_shorter_side_lengths(obb_gdf):
    # Calculates the length of the shorter side of the oriented bounding box (OBB)
    # for each geometry in a GeoDataFrame.
    # Parameters:
    #   obb_gdf (GeoDataFrame): A GeoDataFrame containing OBB geometries.
    # Returns:
    #   list: A list of shorter side lengths for each geometry.

    shorter_sides = []
    for geom in obb_gdf.geometry:
        # Extract the coordinates of the oriented bounding box (OBB)
        rect_coords = list(geom.exterior.coords)

        # Calculate the lengths of the sides
        side_lengths = [LineString([rect_coords[i], rect_coords[(i + 1) % len(rect_coords)]]).length for i in range(len(rect_coords) - 1)]

        # Find the length of the shorter side
        shorter_side_length = min(side_lengths)
        shorter_sides.append(shorter_side_length)

    return shorter_sides
        
def make_geometries_valid_trick(gdf):
    """
    Takes a GeoDataFrame and returns a new GeoDataFrame with valid geometries.
    
    Parameters:
    gdf (GeoDataFrame): A GeoDataFrame with potentially invalid geometries.

    Returns:
    GeoDataFrame: A new GeoDataFrame with all geometries made valid.
    """
    # Apply 'buffer(0)' to each geometry to make it valid
    valid_geometries = gdf.geometry.apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)

    # Create a new GeoDataFrame with these valid geometries
    valid_gdf = gpd.GeoDataFrame(gdf.copy(), geometry=valid_geometries)
    #valid_gdf.set_crs(epsg=gdf.crs.to_epsg(), inplace=True)
    return valid_gdf

def make_geometries_valid(gdf):
    """
    Takes a GeoDataFrame and returns a new GeoDataFrame with valid geometries.
    
    Parameters:
    gdf (GeoDataFrame): A GeoDataFrame with potentially invalid geometries.

    Returns:
    GeoDataFrame: A new GeoDataFrame with all geometries made valid.
    """
    # Apply 'make_valid' to each geometry in the GeoDataFrame
    valid_geometries = gdf.geometry.apply(lambda geom: geom.make_valid() if not geom.is_valid else geom)

    # Create a new GeoDataFrame with these valid geometries
    valid_gdf = gpd.GeoDataFrame(gdf.copy(), geometry=valid_geometries)
    #valid_gdf.set_crs(epsg=gdf.crs.to_epsg(), inplace=True)
    return valid_gdf

def explode_multiparts(gdf):
    """
    Converts any MultiPart geometries in a GeoDataFrame to SinglePart geometries.

    Parameters:
    gdf (GeoDataFrame): A GeoDataFrame that may contain MultiPart geometries.

    Returns:
    GeoDataFrame: A GeoDataFrame where all MultiPart geometries have been converted to SinglePart.
    """
    # Exploding the geometries
    singleparts = gdf.explode(index_parts=True)

    # Resetting the index to avoid duplication
    singleparts.reset_index(drop=True, inplace=True)
    #singleparts.set_crs(epsg=gdf.crs.to_epsg(), inplace=True)
    return singleparts

def get_obb_for_all(gdf):
    """
    For each polygon in a GeoDataFrame, calculates the Oriented Bounding Box (OBB).
    
    Parameters:
    gdf (GeoDataFrame): A GeoDataFrame with valid geometries.

    Returns:
    GeoDataFrame: A new GeoDataFrame containing the OBB for each polygon.
    """
    # Initialize an empty list to store OBB geometries
    obb_geometries = []

    # Iterate over each geometry in the GeoDataFrame
    for polygon in gdf.geometry:
        # Calculate the OBB for each polygon
        obb = polygon.minimum_rotated_rectangle
        obb_geometries.append(obb)

    # Create a new GeoDataFrame with the OBB geometries
    obb_gdf = gpd.GeoDataFrame(geometry=obb_geometries)
    #obb_gdf.set_crs(epsg=gdf.crs.to_epsg(), inplace=True)
    return obb_gdf

def calculate_medial_axis(gdf):
    """
    Calculates the medial axis for each polygon in a GeoDataFrame.
    
    Parameters:
    gdf (GeoDataFrame): A GeoDataFrame with valid geometries.

    Returns:
    GeoDataFrame: A new GeoDataFrame containing the medial axis for each polygon.
    """
    medial_lines = []

    for polygon in gdf.geometry:
        # Extract the coordinates of the oriented bounding box (OBB)
        obb = polygon.minimum_rotated_rectangle
        rect_coords = list(obb.exterior.coords)

        # Calculate the lengths of the sides and identify the shorter sides
        side_lengths = [LineString([rect_coords[i], rect_coords[i+1]]).length for i in range(4)]
        short_sides_indices = sorted(range(len(side_lengths)), key=lambda i: side_lengths[i])[:2]

        # Calculate midpoints of the shorter sides
        midpoints = [midpoint(rect_coords[short_sides_indices[i]], rect_coords[short_sides_indices[i]+1]) for i in range(2)]

        # Create a line connecting these midpoints
        medial_line = LineString(midpoints)
        medial_lines.append(medial_line)

    # Create a new GeoDataFrame with the medial lines
    #medial_gdf = gpd.GeoDataFrame(geometry=medial_lines)
    #Create a new GeoDataFrame copying the original data
    medial_gdf = gdf.copy()
    # Replace the geometry column with the new medial axis geometries
    medial_gdf['geometry'] = medial_lines
    return medial_gdf

def extend_medial_axis_to_bounds_old(gdf, raster_path):
    """
    Extends the medial axis of each line in a GeoDataFrame to the bounds of a given raster.

    Parameters:
    gdf (GeoDataFrame): A GeoDataFrame containing medial axis lines.
    raster_path (str): Path to the raster file.

    Returns:
    GeoDataFrame: A new GeoDataFrame containing extended medial axis lines.
    """
    # Load raster to get bounds
    raster = rasterio.open(raster_path)
    bounds = raster.bounds
    raster_box = box(*bounds)

    extended_lines = []

    for medial_line in gdf.geometry:
        # Calculate delta x and y for the line extension
        dx, dy = medial_line.coords[1][0] - medial_line.coords[0][0], medial_line.coords[1][1] - medial_line.coords[0][1]
        extended_line = LineString([
            (medial_line.coords[0][0] - 10*dx, medial_line.coords[0][1] - 10*dy),  # Extend start point
            (medial_line.coords[1][0] + 10*dx, medial_line.coords[1][1] + 10*dy)   # Extend end point
        ])

        # Intersect the extended line with the raster bounds
        intersected_line = extended_line.intersection(raster_box)
        extended_lines.append(intersected_line)

    # Create a new GeoDataFrame with the extended lines
    #extended_gdf = gpd.GeoDataFrame(geometry=extended_lines)
    #Create a new GeoDataFrame copying the original data
    extended_gdf = gdf.copy()
    # Replace the geometry column with the new medial axis geometries
    extended_gdf['geometry'] = extended_lines
    return extended_gdf 


def extend_medial_axis_to_bounds(gdf: gpd.GeoDataFrame, raster_path: str) -> gpd.GeoDataFrame:
    """
    Extend each line's medial axis to the raster bounds and clip to the box.

    Notes:
    - Assumes line-like geometries.
    - Reprojects GeoDataFrame to raster CRS if needed.
    - Returns line-like intersections; non-line results become None (NaN geometry).
    """
    if gdf.empty:
        return gdf.copy()

    # Open raster and get bounds + CRS
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        bounds = src.bounds
        raster_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    # Reproject to raster CRS if needed
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS. Set gdf.crs before calling.")
    if raster_crs is None:
        raise ValueError("Raster has no CRS. Cannot align coordinates.")
    work_gdf = gdf.to_crs(raster_crs) if gdf.crs != raster_crs else gdf.copy()

    # Compute a large extension length (diagonal of raster bounds * factor)
    diag = math.hypot(bounds.right - bounds.left, bounds.top - bounds.bottom)
    L = max(diag * 5.0, 1.0)  # extend generously beyond the raster box

    extended_geoms = []

    for geom in work_gdf.geometry:
        if geom is None or geom.is_empty:
            extended_geoms.append(None)
            continue

        # Normalize to a single LineString if possible
        line = geom
        if geom.geom_type == "MultiLineString":
            try:
                line = linemerge(geom)
                if line.geom_type == "MultiLineString":  # still multi, pick the longest component
                    line = max(list(line.geoms), key=lambda g: g.length)
            except Exception:
                # Fallback: pick the longest part
                parts = list(geom.geoms)
                line = max(parts, key=lambda g: g.length)

        if line.geom_type != "LineString" or line.length == 0:
            extended_geoms.append(None)
            continue

        # Use endpoints to define overall direction
        x0, y0 = line.coords[0]
        x1, y1 = line.coords[-1]

        # Direction unit vector
        dx = x1 - x0
        dy = y1 - y0
        norm = math.hypot(dx, dy)
        if norm == 0:
            extended_geoms.append(None)
            continue
        ux, uy = dx / norm, dy / norm

        # Extend both directions by large distance L
        start_ext = (x0 - ux * L, y0 - uy * L)
        end_ext   = (x1 + ux * L, y1 + uy * L)
        extended_line = LineString([start_ext, end_ext])

        # Clip with raster bounds
        cut = extended_line.intersection(raster_box)

        # Keep only line-like outputs; others become None
        if cut.is_empty:
            extended_geoms.append(None)
        elif cut.geom_type in ("LineString", "MultiLineString"):
            extended_geoms.append(cut)
        else:
            # Could be points at the box boundary; treat as no usable line
            extended_geoms.append(None)

    # Set the new geometry (keeps only one active geometry column)
    out = work_gdf.set_geometry(extended_geoms)
    # Preserve original CRS of raster (already ensured)
    out.crs = raster_crs
    return out


def extend_medial_axis_to_bounds_v2(gdf: gpd.GeoDataFrame, raster_path: str, extension_factor: float = 0.25) -> gpd.GeoDataFrame:
    """
    Conservative extension: Extend each line's medial axis by a percentage of its original length
    on both sides, then clip to raster bounds. This reduces unwanted non-runway areas in tiles.

    Args:
        gdf: GeoDataFrame containing runway centerlines
        raster_path: Path to the raster file for bounds and CRS
        extension_factor: Factor by which to extend (0.25 = 25% extension on each side)

    Notes:
        - More conservative than extend_medial_axis_to_bounds() 
        - Better for temporal mismatches between FAA data and satellite imagery
        - Reduces inclusion of grass/pavement areas around runways
    """
    if gdf.empty:
        return gdf.copy()

    # Open raster and get bounds + CRS
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        bounds = src.bounds
        raster_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    # Reproject to raster CRS if needed
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS. Set gdf.crs before calling.")
    if raster_crs is None:
        raise ValueError("Raster has no CRS. Cannot align coordinates.")
    work_gdf = gdf.to_crs(raster_crs) if gdf.crs != raster_crs else gdf.copy()

    extended_geoms = []

    for geom in work_gdf.geometry:
        if geom is None or geom.is_empty:
            extended_geoms.append(None)
            continue

        # Normalize to a single LineString if possible
        line = geom
        if geom.geom_type == "MultiLineString":
            try:
                line = linemerge(geom)
                if line.geom_type == "MultiLineString":  # still multi, pick the longest component
                    line = max(list(line.geoms), key=lambda g: g.length)
            except Exception:
                # Fallback: pick the longest part
                parts = list(geom.geoms)
                line = max(parts, key=lambda g: g.length)

        if line.geom_type != "LineString" or line.length == 0:
            extended_geoms.append(None)
            continue

        # Use endpoints to define overall direction
        x0, y0 = line.coords[0]
        x1, y1 = line.coords[-1]

        # Direction unit vector
        dx = x1 - x0
        dy = y1 - y0
        norm = math.hypot(dx, dy)
        if norm == 0:
            extended_geoms.append(None)
            continue
        ux, uy = dx / norm, dy / norm

        # Conservative extension: use percentage of original line length
        original_length = line.length
        extension_distance = original_length * extension_factor

        # Extend both directions by the calculated distance
        start_ext = (x0 - ux * extension_distance, y0 - uy * extension_distance)
        end_ext   = (x1 + ux * extension_distance, y1 + uy * extension_distance)
        extended_line = LineString([start_ext, end_ext])

        # Clip with raster bounds to ensure we stay within image
        cut = extended_line.intersection(raster_box)

        # Keep only line-like outputs; others become None
        if cut.is_empty:
            extended_geoms.append(None)
        elif cut.geom_type in ("LineString", "MultiLineString"):
            extended_geoms.append(cut)
        else:
            # Could be points at the box boundary; treat as no usable line
            extended_geoms.append(None)

    # Set the new geometry (keeps only one active geometry column)
    out = work_gdf.set_geometry(extended_geoms)
    # Preserve original CRS of raster (already ensured)
    out.crs = raster_crs
    return out


def extend_medial_axis_to_bounds_v3(pcl_gdf: gpd.GeoDataFrame, markings_gdf: gpd.GeoDataFrame, 
                                     raster_path: str, margin: float = 50.0) -> gpd.GeoDataFrame:
    """
    Extend each PCL (painted centerline) to cover the bounding box of the markings,
    with an optional margin. This ensures the corridor fully contains all runway markings.

    Args:
        pcl_gdf: GeoDataFrame containing runway centerlines
        markings_gdf: GeoDataFrame containing runway markings (used to compute extent)
        raster_path: Path to the raster file for CRS reference and clipping bounds
        margin: Additional margin in meters to add beyond the markings extent (default 50m)

    Returns:
        GeoDataFrame with extended centerlines that cover markings extent
    """
    if pcl_gdf.empty:
        return pcl_gdf.copy()

    # Open raster and get bounds + CRS
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        bounds = src.bounds
        raster_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    # Reproject to raster CRS if needed
    if pcl_gdf.crs is None:
        raise ValueError("Input pcl_gdf has no CRS. Set gdf.crs before calling.")
    work_pcl = pcl_gdf.to_crs(raster_crs) if pcl_gdf.crs != raster_crs else pcl_gdf.copy()
    
    # Handle markings - if empty, fall back to extension factor method
    if markings_gdf is None or markings_gdf.empty:
        # Fallback: use 25% extension factor
        return extend_medial_axis_to_bounds_v2(pcl_gdf, raster_path, extension_factor=0.25)
    
    work_markings = markings_gdf.to_crs(raster_crs) if markings_gdf.crs != raster_crs else markings_gdf.copy()

    extended_geoms = []

    for geom in work_pcl.geometry:
        if geom is None or geom.is_empty:
            extended_geoms.append(None)
            continue

        # Normalize to a single LineString if possible
        line = geom
        if geom.geom_type == "MultiLineString":
            try:
                line = linemerge(geom)
                if line.geom_type == "MultiLineString":
                    line = max(list(line.geoms), key=lambda g: g.length)
            except Exception:
                parts = list(geom.geoms)
                line = max(parts, key=lambda g: g.length)

        if line.geom_type != "LineString" or line.length == 0:
            extended_geoms.append(None)
            continue

        # Get endpoints and direction
        x0, y0 = line.coords[0]
        x1, y1 = line.coords[-1]

        dx = x1 - x0
        dy = y1 - y0
        norm = math.hypot(dx, dy)
        if norm == 0:
            extended_geoms.append(None)
            continue
        ux, uy = dx / norm, dy / norm

        # Get the total bounds of all markings
        markings_bounds = work_markings.total_bounds  # [minx, miny, maxx, maxy]
        minx, miny, maxx, maxy = markings_bounds

        # Add margin
        minx -= margin
        miny -= margin
        maxx += margin
        maxy += margin

        # Project markings bounds onto the line direction to find required extension
        # For each corner of the markings bounding box, find the projection distance
        corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
        
        # Project each corner onto the line direction relative to line start
        projections = []
        for cx, cy in corners:
            # Vector from line start to corner
            vx, vy = cx - x0, cy - y0
            # Projection onto line direction
            proj = vx * ux + vy * uy
            projections.append(proj)

        min_proj = min(projections)
        max_proj = max(projections)

        # Current line endpoints in projection space
        line_start_proj = 0
        line_end_proj = norm

        # Calculate required extension distances
        start_extension = max(0, line_start_proj - min_proj)
        end_extension = max(0, max_proj - line_end_proj)

        # Extend the line
        start_ext = (x0 - ux * start_extension, y0 - uy * start_extension)
        end_ext = (x1 + ux * end_extension, y1 + uy * end_extension)
        extended_line = LineString([start_ext, end_ext])

        # Clip with raster bounds
        cut = extended_line.intersection(raster_box)

        if cut.is_empty:
            extended_geoms.append(None)
        elif cut.geom_type in ("LineString", "MultiLineString"):
            extended_geoms.append(cut)
        else:
            extended_geoms.append(None)

    out = work_pcl.set_geometry(extended_geoms)
    out.crs = raster_crs
    return out


def _to_single_linestring(geom):
    """Return a LineString (prefer longest) or None."""
    if geom is None or geom.is_empty:
        return None

    if geom.geom_type == "LineString":
        return geom if geom.length > 0 else None

    if geom.geom_type == "MultiLineString":
        try:
            merged = linemerge(geom)
            if merged.geom_type == "LineString":
                return merged if merged.length > 0 else None
            # Still MultiLineString -> choose longest
            return max(list(merged.geoms), key=lambda g: g.length) if len(merged.geoms) else None
        except Exception:
            parts = list(geom.geoms)
            return max(parts, key=lambda g: g.length) if parts else None

    return None


def _extract_longest_line(geom):
    """From LineString/MultiLineString/GeometryCollection, pick the longest LineString."""
    if geom is None or geom.is_empty:
        return None

    if geom.geom_type == "LineString":
        return geom if geom.length > 0 else None

    if geom.geom_type == "MultiLineString":
        lines = list(geom.geoms)
        return max(lines, key=lambda g: g.length) if lines else None

    if geom.geom_type == "GeometryCollection":
        lines = [g for g in geom.geoms if g.geom_type in ("LineString", "MultiLineString")]
        # Flatten possible MultiLineString members
        flat = []
        for g in lines:
            if g.geom_type == "LineString":
                flat.append(g)
            else:
                flat.extend(list(g.geoms))
        return max(flat, key=lambda g: g.length) if flat else None

    return None


def extend_medial_axis_to_markings_obb(
    pcl_gdf: gpd.GeoDataFrame,
    markings_gdf: gpd.GeoDataFrame,
    raster_path: str,
    margin: float = 50.0
) -> gpd.GeoDataFrame:
    """
    Extend each PCL (painted centerline) to cover the Oriented Bounding Box (OBB)
    of the markings (minimum rotated rectangle), with an optional margin.
    Clipped to raster bounds.

    Args:
        pcl_gdf: GeoDataFrame containing runway centerlines
        markings_gdf: GeoDataFrame containing runway markings (used to compute OBB)
        raster_path: Path to raster file for CRS reference and clipping bounds
        margin: Additional margin (same unit as CRS; meters if projected CRS)

    Returns:
        GeoDataFrame with extended centerlines.
    """
    if pcl_gdf is None or pcl_gdf.empty:
        return pcl_gdf.copy()

    if pcl_gdf.crs is None:
        raise ValueError("Input pcl_gdf has no CRS. Set gdf.crs before calling.")

    if markings_gdf is None or markings_gdf.empty:
        raise ValueError("markings_gdf is empty; cannot compute OBB. Provide markings or use your fallback method.")

    if markings_gdf.crs is None:
        raise ValueError("Input markings_gdf has no CRS. Set markings_gdf.crs before calling.")

    # --- Raster bounds + CRS ---
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        bounds = src.bounds
        raster_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    # --- Reproject to raster CRS ---
    work_pcl = pcl_gdf.to_crs(raster_crs) if pcl_gdf.crs != raster_crs else pcl_gdf.copy()
    work_markings = markings_gdf.to_crs(raster_crs) if markings_gdf.crs != raster_crs else markings_gdf.copy()

    # --- Build OBB from markings ---
    markings_union = unary_union([g for g in work_markings.geometry if g is not None and not g.is_empty])
    if markings_union.is_empty:
        raise ValueError("Markings union is empty after cleaning geometries; cannot compute OBB.")

    obb = markings_union.minimum_rotated_rectangle  # Oriented bbox
    if margin and margin > 0:
        # Square-ish expansion around the OBB to keep corners reasonable
        obb = obb.buffer(margin, cap_style=3, join_style=2)

    # OBB corners: use exterior coords (last == first)
    obb_coords = list(obb.exterior.coords)[:-1]
    if len(obb_coords) < 4:
        raise ValueError("Computed OBB does not have expected polygon corners.")

    extended_geoms = []

    for geom in work_pcl.geometry:
        line = _to_single_linestring(geom)
        if line is None:
            extended_geoms.append(None)
            continue

        # Endpoints and direction
        x0, y0 = line.coords[0]
        x1, y1 = line.coords[-1]

        dx = x1 - x0
        dy = y1 - y0
        norm = math.hypot(dx, dy)
        if norm == 0:
            extended_geoms.append(None)
            continue

        ux, uy = dx / norm, dy / norm

        # Project OBB corners onto line direction relative to line start
        projections = []
        for cx, cy in obb_coords:
            vx, vy = cx - x0, cy - y0
            proj = vx * ux + vy * uy
            projections.append(proj)

        min_proj = min(projections)
        max_proj = max(projections)

        # Current line endpoints in projection space
        line_start_proj = 0.0
        line_end_proj = norm

        # Required extension
        start_extension = max(0.0, line_start_proj - min_proj)
        end_extension   = max(0.0, max_proj - line_end_proj)

        # Extend
        start_ext = (x0 - ux * start_extension, y0 - uy * start_extension)
        end_ext   = (x1 + ux * end_extension, y1 + uy * end_extension)
        extended_line = LineString([start_ext, end_ext])

        # Clip to raster bounds and pick the longest line if fragmented
        clipped = extended_line.intersection(raster_box)
        clipped_line = _extract_longest_line(clipped)

        extended_geoms.append(clipped_line)

    out = work_pcl.copy()
    out = out.set_geometry(extended_geoms)
    out.crs = raster_crs
    return out


def create_polygon_from_corners(ulx, uly, lrx, lry):
    # Creates a polygon from specified corner coordinates.
    # Parameters:
    #   ulx, uly, lrx, lry (float): Coordinates for upper left and lower right corners.
    # Returns:
    #   Polygon: A shapely Polygon object.

    llx, lly = ulx, lry  # lower left x,y - same x as upper left, same y as lower right
    urx, ury = lrx, uly  # upper right x,y - same x as lower right, same y as upper left
    # Create a polygon using the four corners
    return Polygon([(ulx, uly), (urx, ury), (lrx, lry), (llx, lly), (ulx, uly)])


def buffer_extended_gdf_ori(extended_gdf, obb_gdf, input_raster, width_scale=3.0):
    # 1) Ensure both GDFs have correct CRS set before any operation
    if extended_gdf.crs is None:
        raise ValueError("extended_gdf.crs is None. Set the correct CRS before buffering.")
    if obb_gdf.crs is None:
        raise ValueError("obb_gdf.crs is None. Set the correct CRS before computing lengths.")

    # 2) Read raster CRS
    with rasterio.open(input_raster) as src:
        raster_crs = src.crs

    # 3) Reproject both to the raster CRS **if needed**
    if extended_gdf.crs != raster_crs:
        extended_gdf = extended_gdf.to_crs(raster_crs)
    if obb_gdf.crs != raster_crs:
        obb_gdf = obb_gdf.to_crs(raster_crs)

    # 4) Compute buffer widths (in raster CRS units, typically meters)
    shorter_sides = calculate_shorter_side_lengths(obb_gdf)
    buffer_widths = [s * width_scale for s in shorter_sides]  # e.g., half of shorter side

    # 5) Compute azimuth/rotation (optional, but now in raster CRS)
    extended_gdf['azimuth'] = extended_gdf['geometry'].apply(calculate_azimuth)
    extended_gdf['rotation_angle'] = extended_gdf['azimuth'].apply(compute_rotation_angle)

    # 6) Buffer (flat caps)
    buffered_geometries = [
        geom.buffer(w, cap_style=2) for geom, w in zip(extended_gdf.geometry, buffer_widths)
    ]

    buffered_gdf = extended_gdf.copy()
    buffered_gdf['geometry'] = buffered_geometries
    buffered_gdf.set_crs(raster_crs, inplace=True)  # CRS is already correct after to_crs

    return buffered_gdf



def buffer_extended_gdf_uniform_width_3857(extended_gdf, input_raster,buffer_meters=100.0):
    """
    Uniform-width buffering:
      1) Project to EPSG:3857 (meters)
      2) Buffer by 100 m with flat caps 
      3) Reproject back to original CRS
      4) Reproject both to raster CRS 
    """
    # 1) Ensure both GDFs have correct CRS set before any operation
    if extended_gdf.crs is None:
        raise ValueError("extended_gdf.crs is None. Set the correct CRS before buffering.")

    original_crs = extended_gdf.crs

    # 2) Read raster CRS
    with rasterio.open(input_raster) as src:
        raster_crs = src.crs

    # --- Compute azimuth/rotation on original geometries (pre-buffer) ---
    extended_gdf = extended_gdf.copy()
    extended_gdf['azimuth'] = extended_gdf['geometry'].apply(calculate_azimuth)
    extended_gdf['rotation_angle'] = extended_gdf['azimuth'].apply(compute_rotation_angle)

    # --- Uniform X m buffer in EPSG:3857 ---
    # 1) Project to 3857 (units: meters)
    extended_gdf_um = extended_gdf.to_crs("EPSG:3857")

    # 2) Buffer by fixed X meters with flat caps (cap_style=2)
    buffered_um = extended_gdf_um.geometry.buffer(buffer_meters, cap_style=2)

    # 3) Reproject buffered geometries back to original CRS
    buffered_orig = buffered_um.to_crs(original_crs)

    # Replace geometry with the uniformly buffered geometry
    extended_gdf.set_geometry(buffered_orig, inplace=True)
    extended_gdf.set_crs(original_crs, inplace=True)

    # 4) Reproject both to the raster CRS (existing behavior)
    if extended_gdf.crs != raster_crs:
        extended_gdf = extended_gdf.to_crs(raster_crs)

    # Output in raster CRS with buffered geometry
    buffered_gdf = extended_gdf.copy()
    buffered_gdf.set_crs(raster_crs, inplace=True)

    return buffered_gdf

def buffer_extended_gdf_uniform_width(extended_gdf, input_raster,buffer_meters=100.0):
    """
    Uniform-width buffering using UTM projection:
      1) Determine appropriate UTM zone using _utm_epsg_from_lonlat()
      2) Project to UTM (meters)
      3) Buffer by specified meters with flat caps 
      4) Reproject back to original CRS
      5) Reproject to raster CRS 
    """
    # 1) Ensure GDF has correct CRS set before any operation
    if extended_gdf.crs is None:
        raise ValueError("extended_gdf.crs is None. Set the correct CRS before buffering.")
    
    if extended_gdf.empty:
        return extended_gdf.copy()

    original_crs = extended_gdf.crs

    # 2) Read raster CRS
    with rasterio.open(input_raster) as src:
        raster_crs = src.crs

    # --- Compute azimuth/rotation on original geometries (pre-buffer) ---
    extended_gdf = extended_gdf.copy()
    extended_gdf['azimuth'] = extended_gdf['geometry'].apply(calculate_azimuth)
    extended_gdf['rotation_angle'] = extended_gdf['azimuth'].apply(compute_rotation_angle)

    # --- Determine UTM zone from first valid geometry ---
    # Find first non-empty geometry to determine UTM zone
    representative_geom = None
    for geom in extended_gdf.geometry:
        if geom is not None and not geom.is_empty:
            representative_geom = geom
            break
    
    if representative_geom is None:
        # No valid geometries, return empty result
        buffered_gdf = extended_gdf.copy()
        if raster_crs != original_crs:
            buffered_gdf = buffered_gdf.to_crs(raster_crs)
        return buffered_gdf
    
    # Get centroid coordinates for UTM zone calculation
    #if original_crs != 'EPSG:4326':
    orig_epsg = extended_gdf.crs.to_epsg()
    if orig_epsg != 4326:
        # Temporarily convert to WGS84 for UTM zone calculation
        temp_geom = gpd.GeoSeries([representative_geom], crs=original_crs).to_crs('EPSG:4326').iloc[0]
        centroid = temp_geom.centroid
    else:
        centroid = representative_geom.centroid
    
    # Determine appropriate UTM EPSG code
    utm_epsg = _utm_epsg_from_lonlat(centroid.x, centroid.y)
    utm_crs = f"EPSG:{utm_epsg}"

    # --- Uniform buffer in UTM projection ---
    # 1) Project to UTM (units: meters)
    extended_gdf_utm = extended_gdf.to_crs(utm_crs)

    # 2) Buffer by fixed meters with flat caps (cap_style=2)
    buffered_utm = extended_gdf_utm.geometry.buffer(buffer_meters, cap_style=2)

    # 3) Reproject buffered geometries back to original CRS
    buffered_orig = buffered_utm.to_crs(original_crs)

    # Replace geometry with the uniformly buffered geometry
    extended_gdf.set_geometry(buffered_orig, inplace=True)
    extended_gdf.set_crs(original_crs, inplace=True)

    # 4) Reproject to raster CRS (existing behavior)
    if extended_gdf.crs != raster_crs:
        extended_gdf = extended_gdf.to_crs(raster_crs)

    # Output in raster CRS with buffered geometry
    buffered_gdf = extended_gdf.copy()
    buffered_gdf.set_crs(raster_crs, inplace=True)

    return buffered_gdf


def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """
    Pick a UTM EPSG code from lon/lat.
    Northern hemisphere -> EPSG:326xx
    Southern hemisphere -> EPSG:327xx
    """
    #zone = int((lon + 180) // 6) + 1
    zone = int((lon + 180) // 6) + 1
    zone = max(1, min(zone, 60))

    return (32600 + zone) if lat >= 0 else (32700 + zone)


def buffer_extended_gdf_uniform_width_v2(extended_gdf, input_raster,buffer_meters=50.0):
    """
    CRS-aware uniform-width buffering:
      - If raster CRS is EPSG:4326 (geographic), delegates to buffer_extended_gdf_uniform_width() 
        which uses UTM projection for accurate metric buffering
      - Otherwise, projects directly to raster CRS and buffers
    """
    # 1) Ensure GDF has correct CRS set before any operation
    if extended_gdf.crs is None:
        raise ValueError("extended_gdf.crs is None. Set the correct CRS before buffering.")

    # 2) Read raster CRS
    with rasterio.open(input_raster) as src:
        raster_crs = src.crs
    
    # 3) Check if raster CRS is geographic (EPSG:4326)
    #if raster_crs is not None and str(raster_crs).upper() in ['EPSG:4326', 'WGS84']:
    #epsg = raster_crs.to_epsg() if raster_crs else None
    #if epsg == 4326 or (raster_crs and raster_crs.is_geographic):
    raster_epsg = raster_crs.to_epsg()
    if raster_epsg == 4326 or raster_crs.is_geographic:
        # For geographic CRS, delegate to UTM-based buffering for accuracy
        return buffer_extended_gdf_uniform_width(extended_gdf, input_raster, buffer_meters)
    
    # 4) For projected CRS, proceed with direct buffering
    # --- Compute azimuth/rotation on original geometries (pre-buffer) ---
    extended_gdf = extended_gdf.copy()
    extended_gdf['azimuth'] = extended_gdf['geometry'].apply(calculate_azimuth)
    extended_gdf['rotation_angle'] = extended_gdf['azimuth'].apply(compute_rotation_angle)

    # Project to raster CRS
    extended_gdf_proj = extended_gdf.to_crs(raster_crs)

    # Buffer by fixed meters with flat caps (cap_style=2)
    buffered_geoms = extended_gdf_proj.geometry.buffer(buffer_meters, cap_style=2)

    # Create a new GeoDataFrame with buffered geometries
    buffered_gdf = extended_gdf_proj.copy()
    buffered_gdf['geometry'] = buffered_geoms
    buffered_gdf.set_crs(raster_crs, inplace=True)

    return buffered_gdf


def buffer_extended_gdf(extended_gdf, obb_gdf, input_raster, buffer_factor=3.0):
    """
    Applies a buffer to the extended geometries in a GeoDataFrame using OBB-derived widths.
    Ensures CRS alignment and computes azimuth/rotation in a projected CRS.
    """
    if extended_gdf.empty:
        # Return an empty copy with expected columns
        out = extended_gdf.copy()
        out["azimuth"] = None
        out["rotation_angle"] = None
        return out

    # 1) Read raster CRS
    with rasterio.open(input_raster) as src:
        raster_crs = src.crs

    if raster_crs is None:
        raise ValueError("Raster has no CRS; cannot safely align vector data.")

    if extended_gdf.crs is None:
        raise ValueError("extended_gdf has no CRS; set gdf.crs before calling.")
    if obb_gdf.crs is None:
        # If obb_gdf has no CRS, we cannot trust side lengths; raise.
        raise ValueError("obb_gdf has no CRS; cannot compute distances safely.")

    # 2) Bring both to a common working CRS (prefer raster CRS if projected)
    # If raster CRS is geographic (degrees), consider choosing a suitable local projected CRS instead.
    work_crs = raster_crs
    ext = extended_gdf.to_crs(work_crs) if extended_gdf.crs != work_crs else extended_gdf.copy()
    obb = obb_gdf.to_crs(work_crs) if obb_gdf.crs != work_crs else obb_gdf.copy()

    # 3) Compute buffer widths (must align with ext rows)
    #    Best practice: return a pandas Series with the same index as obb,
    #    then reindex/align it to ext via a merge or a common key/index.
    #    If ext and obb are guaranteed to be row-aligned and same length,
    #    convert to a Series with ext.index explicitly.
    widths = calculate_shorter_side_lengths(obb)
    # Ensure widths is a pandas Series aligned to ext
    import pandas as pd
    if not isinstance(widths, pd.Series):
        widths = pd.Series(widths, index=obb.index)
    # Align widths to ext by index (assumes same ordering & index; otherwise, merge on a key)
    try:
        widths = widths.reindex(ext.index)
    except Exception:
        raise ValueError("Could not align width Series to extended_gdf index. Ensure matching indices or join via a key.")

    widths = widths.astype(float) * float(buffer_factor)

    # 4) Compute azimuths and rotation angles AFTER CRS alignment
    def safe_azimuth(geom):
        if geom is None or geom.is_empty:
            return None
        return calculate_azimuth(geom)

    ext["azimuth"] = ext.geometry.apply(safe_azimuth)

    def safe_rotation(az):
        if az is None:
            return 0.0
        return compute_rotation_angle(az)

    ext["rotation_angle"] = ext["azimuth"].apply(safe_rotation)

    # 5) Buffer safely (guard None/empty)
    # Vectorized approach that respects index alignment:
    # GeoPandas supports a Series distance; however, invalid/None geometries still need guarding.
    buffered = []
    for idx, geom in ext.geometry.items():
        w = widths.get(idx, None)
        if geom is None or geom.is_empty or w is None:
            buffered.append(None)
            continue
        try:
            # Flat ends (cap_style=2). You can also choose join_style=1 (round) or 2 (miter) based on look.
            buffered.append(geom.buffer(w, cap_style=2))
        except Exception:
            buffered.append(None)

    # 6) Set geometry & CRS (ext already in work_crs)
    out = ext.set_geometry(buffered)
    out.crs = work_crs

    return out


def mask_raster_with_all_geometries(input_raster_path, gdf):
    # Masks a raster image using all geometries in a GeoDataFrame.
    # Parameters:
    #   input_raster_path (str): Path to the raster file.
    #   gdf (GeoDataFrame): A GeoDataFrame containing geometries for masking.
    # Returns:
    #   tuple: A tuple containing the masked raster array and updated metadata.

    # Check if GeoDataFrame is empty or has no valid geometries
    if gdf.empty or gdf.geometry.isna().all():
        raise ValueError("GeoDataFrame is empty or contains no valid geometries")

    # Filter out any invalid geometries
    valid_gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
    
    if valid_gdf.empty:
        raise ValueError("No valid geometries found after filtering")

    # Merge all geometries into a single geometry
    all_geometries = unary_union(valid_gdf.geometry)

    # Check if the unified geometry is valid
    if all_geometries is None or all_geometries.is_empty:
        raise ValueError("Unified geometry is empty or invalid")

    # Open the raster
    with rasterio.open(input_raster_path) as src:
        # Convert geometry to GeoJSON format for rasterio.mask
        if hasattr(all_geometries, '__geo_interface__'):
            geom_for_mask = [all_geometries.__geo_interface__]
        else:
            # Fallback for complex geometries
            geom_for_mask = [all_geometries]
            
        # Mask the raster with the merged geometry
        out_image, out_transform = mask(src, geom_for_mask, crop=True)
        out_meta = src.meta.copy()
        #Update the metadata to reflect the number of layers (bands)
        out_meta.update({
            "driver": "PNG",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

    return out_image,out_meta
    
def divide_linestring_into_equal_parts(linestring, num_parts):
    # Divides a Line in equal parts
    total_length = linestring.length
    part_length = total_length / num_parts
    return [linestring.interpolate(i * part_length) for i in range(num_parts + 1)]

def calculate_window_bounds(src, window):
    # Get the bounds of the window in the raster's coordinate reference system
    window_bounds = src.window_bounds(window)
    # Convert to a shapely box (polygon)
    return box(*window_bounds)

def clip_by_aoi(gdf,aoi):
    # Ensure that both GeoDataFrames have the same CRS
    gdf = gdf.to_crs(aoi.crs)

    # Assuming the area of interest is a single polygon, we extract the geometry
    aoi_polygon = aoi.geometry.iloc[0]

    # Clip the geometries in gdf by the aoi_polygon
    clipped_gdf = gdf.clip(aoi_polygon)

    return clipped_gdf


def dissolve_overlapping_polygons(gdf, overlap_threshold):
    # Initialize a column to mark polygons for dissolving
    gdf['dissolve_group'] = -1
    dissolve_group_id = 0

    for idx, row in gdf.iterrows():
        if row['dissolve_group'] != -1:
            # Skip polygons already assigned to a group
            continue

        geom = row.geometry
        # Create a group for this polygon
        gdf.at[idx, 'dissolve_group'] = dissolve_group_id

        for idx_other, row_other in gdf.iterrows():
            if idx != idx_other and row_other['dissolve_group'] == -1:
                geom_other = row_other.geometry
                intersection = geom.intersection(geom_other)
                if not intersection.is_empty:
                    # Calculate the overlap percentage
                    overlap_percentage = intersection.area / min(geom.area, geom_other.area)
                    if overlap_percentage > overlap_threshold:
                        # Mark overlapping polygon to the same group
                        gdf.at[idx_other, 'dissolve_group'] = dissolve_group_id
        
        dissolve_group_id += 1

    # Dissolve polygons based on the group
    dissolved_gdf = gdf.dissolve(by='dissolve_group')
    
    return dissolved_gdf

def dissolve_overlapping_polygons_iou(gdf, iou_threshold=0.5):
    # Initialize a column to mark polygons for dissolving
    gdf['dissolve_group'] = -1
    dissolve_group_id = 0

    for idx, row in gdf.iterrows():
        if row['dissolve_group'] != -1:
            # Skip polygons already assigned to a group
            continue

        geom = row.geometry
        # Create a group for this polygon
        gdf.at[idx, 'dissolve_group'] = dissolve_group_id

        for idx_other, row_other in gdf.iterrows():
            if idx != idx_other and row_other['dissolve_group'] == -1:
                geom_other = row_other.geometry
                intersection = geom.intersection(geom_other)
                if not intersection.is_empty:
                    # Calculate the Intersection over Union (IoU)
                    union = geom.union(geom_other)
                    iou = intersection.area / union.area
                    if iou > iou_threshold:
                        # Mark overlapping polygon to the same group
                        gdf.at[idx_other, 'dissolve_group'] = dissolve_group_id
        
        dissolve_group_id += 1

    # Dissolve polygons based on the group
    dissolved_gdf = gdf.dissolve(by='dissolve_group')
    return dissolved_gdf

def retain_larger_and_non_overlapping_polygons(gdf, overlap_threshold=50):
    """
    Retains larger polygons among overlapping polygons and includes non-overlapping polygons.

    Parameters:
    - gdf (GeoDataFrame): GeoPandas GeoDataFrame containing polygons.
    - overlap_threshold (float): Overlap percentage threshold (0-100).

    Returns:
    - GeoDataFrame: A new GeoDataFrame with retained polygons.
    """
    # Ensure the input GeoDataFrame has valid geometries
    gdf = gdf[gdf.is_valid].reset_index(drop=True)

    # Initialize a list to store indices to retain
    indices_to_retain = set(range(len(gdf)))

    # Iterate through all polygons
    for i, poly1 in gdf.iterrows():
        if i not in indices_to_retain:
            continue

        for j, poly2 in gdf.iterrows():
            if i >= j or j not in indices_to_retain:  # Avoid redundant comparisons
                continue

            # Check if the polygons overlap
            if poly1.geometry.intersects(poly2.geometry):
                # Compute intersection area
                intersection_area = poly1.geometry.intersection(poly2.geometry).area

                # Compute percentage overlap for each polygon
                overlap1 = (intersection_area / poly1.geometry.area) * 100
                overlap2 = (intersection_area / poly2.geometry.area) * 100

                # Retain the larger polygon if overlap exceeds threshold
                if overlap1 > overlap_threshold or overlap2 > overlap_threshold:
                    if poly1.geometry.area >= poly2.geometry.area:
                        indices_to_retain.discard(j)  # Remove smaller polygon
                    else:
                        indices_to_retain.discard(i)
                        break

    # Create a new GeoDataFrame with retained polygons
    indices_to_retain=list(indices_to_retain)
    return gdf.loc[indices_to_retain].reset_index(drop=True)
    
def extract_clips(input_raster_path, input_vector_mask, num_parts, clip_size,clips_path,save_clips=True):
    gdf = gpd.read_file(input_vector_mask)
    with rasterio.open(input_raster_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        clips = []
        clip_bounds_geometries = []
        clip_names=[]

        for line in gdf.geometry:
            points = divide_linestring_into_equal_parts(line, num_parts)
            for i in range(len(points) - 1):
                midpoint = LineString([points[i], points[i + 1]]).interpolate(0.5, normalized=True)
                row, col = src.index(midpoint.x, midpoint.y)

                left = max(col - clip_size // 2, 0)
                top = max(row - clip_size // 2, 0)
                right = min(col + clip_size // 2, src.width)
                bottom = min(row + clip_size // 2, src.height)

                if right - left < clip_size:
                    left = max(right - clip_size, 0)
                if bottom - top < clip_size:
                    top = max(bottom - clip_size, 0)

                window = Window(left, top, right - left, bottom - top)
                clip = src.read(window=window)
                clips.append(clip)

                # Convert window bounds to geographic coordinates
                clip_bounds_geometries.append(calculate_window_bounds(src, window))
                output_path = os.path.join(clips_path,f"{col}_{row}.png")
                clip_names.append(output_path)
                if save_clips:                    
                    transform = src.window_transform(window)
                    # Update the profile with the actual dimensions of the window
                    profile = src.profile
                    profile.update({
                        'height': window.height,
                        'width': window.width,
                        'transform': transform})
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(clip)                   
       
        # Create the GeoDataFrame with both geometry and clip_names columns        
        clip_bounds_gdf = gpd.GeoDataFrame({
            'geometry': clip_bounds_geometries,
            'clip_names': clip_names
        })

        # Remove any duplicate geometries
        clip_bounds_gdf=clip_bounds_gdf.drop_duplicates(keep='first')
        #Remove if there are significantly overlapping polygons are found.
        clip_bounds_gdf = dissolve_overlapping_polygons(clip_bounds_gdf, 0.6)
        
        return clips,clip_bounds_gdf

def extract_clips_near_points(raster_path, vector_path, clip_size,clips_path,save_clips=True):
    gdf = gpd.read_file(vector_path)
    with rasterio.open(raster_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        clips = []
        clip_bounds_geometries = []
        clip_names=[]
        for index, item in gdf.iterrows():
            point=item['geometry'].centroid
            
            row, col = src.index(point.x, point.y)

            left = max(col - clip_size // 2, 0)
            top = max(row - clip_size // 2, 0)
            right = min(col + clip_size // 2, src.width)
            bottom = min(row + clip_size // 2, src.height)

            if right - left < clip_size:
                left = max(right - clip_size, 0)
            if bottom - top < clip_size:
                top = max(bottom - clip_size, 0)

            window = Window(left, top, right - left, bottom - top)
            clip = src.read(window=window)
            clips.append(clip)

            # Convert window bounds to geographic coordinates
            clip_bounds_geometries.append(calculate_window_bounds(src, window))
            output_path = os.path.join(clips_path,f"{col}_{row}.png")
            clip_names.append(output_path)
            if save_clips:                    
                transform = src.window_transform(window)
                # Update the profile with the actual dimensions of the window
                profile = src.profile
                profile.update({
                    'height': window.height,
                    'width': window.width,
                    'transform': transform})
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(clip)                   
       
        # Create the GeoDataFrame with both geometry and clip_names columns        
        clip_bounds_gdf = gpd.GeoDataFrame({
            'geometry': clip_bounds_geometries,
            'clip_names': clip_names
        })

        # Remove any duplicate geometries
        clip_bounds_gdf=clip_bounds_gdf.drop_duplicates(keep='first')
        #Remove if there are significantly overlapping polygons are found.
        clip_bounds_gdf = dissolve_overlapping_polygons(clip_bounds_gdf, 0.6)
        
        return clips,clip_bounds_gdf
        
def get_tile_indices(src_geotiff,patch_size,overlap,tile_index_only,target_dir,csv_path):
    # Add your logic here to perform the desired actions based on the parameters
    parameters = ' '.join(['-ps',f'{patch_size} {patch_size}',
        '-overlap',str(overlap),
        '-tileIndexOnly', str(tile_index_only),
        '-csv',csv_path,
        '-targetDir',target_dir,
        src_geotiff,
      ])
    #print(parameters)
    parameters='gdal_retile.py' + ' '+parameters #Precede with a dummy parameter to be in sync with gdal_retile.py
    #print(parameters)
    rp = RetileProcessor(parameters.split())
    rp.run()
    minfo=rp.minfo
    df_tiles=rp.df
    
    return df_tiles

# Creates and saves tiles from a raster dataset based on the tiles specified in a GeoDataFrame.
# Useful for breaking down a large raster dataset into manageable tiles.

def create_and_save_tiles(mask_raster_path, tiles_gdf, preds_path):
    # Ensure the tiles directory exists
    tiles_dir = preds_path / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    with rasterio.open(mask_raster_path) as src:
        for _, row in tiles_gdf.iterrows():
            # Extract the bounds of the current tile
            bounds = row.geometry.bounds
            window = from_bounds(*bounds, src.transform)

            # Read the tile data from the raster
            tile_data = src.read(window=window)

            # Extract only the filename from the 'Tile' column
            tile_filename = os.path.basename(row['Tile']).split('.')[0]+'.png'
            tile_filepath = tiles_dir / tile_filename

            # Save the tile
            with rasterio.open(tile_filepath, 'w', driver='PNG', 
                               height=tile_data.shape[1], width=tile_data.shape[2],
                               count=src.count, dtype=tile_data.dtype,
                               crs=src.crs, transform=rasterio.windows.transform(window, src.transform)) as tile_dst:
                tile_dst.write(tile_data)

def get_tiles_with_preds(tiles_along_mask,predictions_path):

    gdf_preds = gpd.read_file(predictions_path)
    tiles_along_mask_gdf = gpd.read_file(tiles_along_mask)
    
    gdf_preds.plot()
    tiles_along_mask_gdf.plot()
    # Step 1: Use 'intersects' to filter polygons
    # Create a mask that is True for rows in tiles_along_mask_gdf that intersect with any line in gdf_preds
    mask = tiles_along_mask_gdf.intersects(gdf_preds.unary_union)
    # Apply the mask to filter tiles_along_mask_gdf
    tiles_with_preds_gdf = tiles_along_mask_gdf[mask]

    return tiles_with_preds_gdf 
    
def meters_to_degrees(meters, latitude):
    # Earth's radius in meters (approximate value)
    earth_radius = 6371000  # meters

    # Calculate the circumference of the Earth at the given latitude
    circumf_at_latitude = 2 * 3.14159265359 * earth_radius * abs(math.cos(math.radians(latitude)))

    # Calculate the equivalent degrees for the given meters
    degrees = meters / circumf_at_latitude * 360

    return degrees
    
from shapely.geometry import box

def extract_clip_for_polygon_envelope(raster_path, vector_path, clips_path, driver='GTiff', quality=None, rgb_only=False):
    """
    Extracts clipped tiles from a raster dataset based on the envelope of vector polygons.

    Parameters:
    - raster_path (str): Path to the input raster file.
    - vector_path (str): Path to the input vector file (GeoJSON, Shapefile, etc.) containing polygons.
    - clips_path (str): Path to the directory where the clipped tiles will be saved.
    - driver (str): Output format driver. Options: 'GTiff' (default), 'PNG', 'JP2OpenJPEG'.
    - quality (int or None): Quality setting (1-100). 
        - For JP2OpenJPEG: 1-99 = lossy, 100 = lossless (REVERSIBLE=YES)
        - For GTiff: Uses JPEG compression with this quality (1-100). None = DEFLATE (lossless)
        - For PNG: Ignored (always lossless)
    - rgb_only (bool): If True, output only the first 3 bands (RGB). Default False.

    Purpose:
    This function takes a raster dataset and a vector dataset containing polygons. It extracts tiles from the
    raster dataset based on the expanded envelope of each polygon. The resulting clipped tiles are saved as 
    individual files in the specified directory with format based on driver.
    """

    if not os.path.exists(clips_path):
        os.makedirs(clips_path)
    # Map driver to file extension
    driver_ext_map = {
        'GTiff': '.tif',
        'PNG': '.png',
        'JP2OpenJPEG': '.jp2'
    }
    ext = driver_ext_map.get(driver, '.tif')
    
    # Build creation options based on driver and quality
    creation_options = {}
    if driver == 'JP2OpenJPEG' and quality is not None:
        creation_options['QUALITY'] = quality
        if quality == 100:
            creation_options['REVERSIBLE'] = 'YES'  # Lossless
    elif driver == 'GTiff':
        if quality is not None:
            creation_options['COMPRESS'] = 'JPEG'
            creation_options['JPEG_QUALITY'] = quality
        else:
            creation_options['COMPRESS'] = 'DEFLATE'  # Lossless default
    
    # Open the raster dataset
    with rasterio.open(raster_path) as src:
        # Open the vector dataset as a GeoDataFrame
        vector_data = gpd.read_file(vector_path)

        for index, row in vector_data.iterrows():
            # Extract the bounds of the current polygon's envelope
            bounds = row.geometry.bounds

            # Create a buffered geometry with the specified margin_degrees
            #buffered_bbox = box(*bounds).buffer(0.00006)
            #buffered_bbox = box(*bounds).buffer(0.00018)
            buffered_bbox = box(*bounds)

            # Convert the buffered bounding box to a GeoJSON-like geometry
            buffered_bbox_geojson = buffered_bbox.__geo_interface__

            # Clip the raster using the buffered bounding box
            try:
                clipped_data, clipped_transform = rasterio.mask.mask(
                    src, [buffered_bbox_geojson], crop=True, nodata=None
                )
            except Exception as e:
                print(f"Error while clipping: {e}")
                continue

            # Extract RGB bands only if requested
            if rgb_only and clipped_data.shape[0] >= 3:
                output_data = clipped_data[:3]
                output_count = 3
            else:
                output_data = clipped_data
                output_count = clipped_data.shape[0]

            # Extract only the filename from the vector dataset's index
            tile_filename = str(index) + ext
            tile_filepath = os.path.join(clips_path, tile_filename)
            print("Saving #", tile_filepath)
            # Save the clipped tile
            with rasterio.open(
                tile_filepath,
                'w',
                driver=driver,
                height=output_data.shape[1],
                width=output_data.shape[2],
                count=output_count,
                dtype=output_data.dtype,
                crs=src.crs,
                transform=clipped_transform,
                **creation_options,
            ) as tile_dst:
                tile_dst.write(output_data)
                
                
def extract_clip_for_polygon_envelope_v2(
    raster_path,
    polygons_gdf,
    clips_path,
    margin_m=0.0,              # margin in meters (recommended)
    output_driver='GTiff',     # 'GTiff' for georeferenced outputs; 'PNG' if you want non-georeferenced image tiles
    compress='DEFLATE',        # GTiff compression
):
    """
    Extracts clipped tiles around the envelope (bbox) of each polygon in 'polygons_gdf',
    expanding by 'margin_m' meters. Assumes polygons_gdf is in the same CRS as raster.
    """
    os.makedirs(clips_path, exist_ok=True)

    with rasterio.open(raster_path) as src:
        # Ensure CRS alignment
        if polygons_gdf.crs != src.crs:
            poly_gdf = polygons_gdf.to_crs(src.crs)
        else:
            poly_gdf = polygons_gdf

        # If margin_m > 0 and raster CRS is geographic, temporarily go to metric CRS to buffer
        work_gdf = poly_gdf
        metric_needed = margin_m and not CRS.from_user_input(src.crs).is_projected
        if metric_needed:
            metric_crs = _guess_metric_crs_from_bounds(src.bounds)
            work_gdf = poly_gdf.to_crs(metric_crs)

        for idx, row in work_gdf.iterrows():
            # Envelope
            env = row.geometry.envelope
            # Apply margin in meters if applicable
            if margin_m and metric_needed:
                env = env.buffer(margin_m)
            elif margin_m and not metric_needed:
                # Margin interpreted in CRS units (e.g., degrees) – use with caution
                env = env.buffer(margin_m)

            # Bring envelope back to raster CRS if we detoured to metric CRS
            if metric_needed:
                env = gpd.GeoSeries([env], crs=work_gdf.crs).to_crs(src.crs).iloc[0]

            # Clip using the envelope polygon
            env_geojson = env.__geo_interface__

            try:
                clipped_data, clipped_transform = rasterio.mask.mask(
                    src, [env_geojson], crop=True, nodata=src.nodatavals[0] if src.nodatavals else None
                )
            except Exception as e:
                print(f"[{idx}] Error while clipping: {e}")
                continue

            # Build output path
            ext = 'tif' if output_driver == 'GTiff' else 'png'
            tile_filename = f"{idx}.{ext}"
            tile_filepath = os.path.join(clips_path, tile_filename)

            profile = src.profile.copy()
            profile.update(
                driver=output_driver,
                height=clipped_data.shape[1],
                width=clipped_data.shape[2],
                count=clipped_data.shape[0],
                transform=clipped_transform,
            )
            if output_driver == 'GTiff':
                profile.update(compress=compress)

            with rasterio.open(tile_filepath, 'w', **profile) as dst:
                dst.write(clipped_data)

def clip_by_raster_bounding_box_old(gdf, input_raster):
    # Open the raster file using rasterio
    with rasterio.open(input_raster) as rast_src:
        # Get the bounding box of the raster
        bbox = rast_src.bounds
    
    # Convert the bounding box to a shapely geometry (Polygon)
    bbox_polygon = box(bbox.left, bbox.bottom, bbox.right, bbox.top)

    # Ensure the GeoDataFrame has the same CRS as the raster
    gdf = gdf.to_crs(rast_src.crs)

    # Create a GeoDataFrame for the bounding box
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox_polygon]}, crs=rast_src.crs)

    # Clip the input GeoDataFrame using the bounding box
    clipped_gdf = gpd.clip(gdf, bbox_gdf)

    return clipped_gdf

def clip_by_raster_bounding_box(gdf: gpd.GeoDataFrame, input_raster: str) -> gpd.GeoDataFrame:
    """
    Clip a GeoDataFrame to the bounding box of a raster.

    - Ensures CRS alignment with raster CRS.
    - Returns geometries intersecting with the raster extent; others become empty and are dropped by gpd.clip.
    """
    if gdf is None or gdf.empty:
        return gdf.copy()

    # Read raster bounds and CRS safely
    with rasterio.open(input_raster) as src:
        raster_crs = src.crs
        bounds = src.bounds

    if raster_crs is None:
        raise ValueError("Raster has no CRS; cannot align coordinates for clipping.")
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS; set gdf.crs before clipping.")

    # Reproject gdf to raster CRS if needed
    gdf_in = gdf.to_crs(raster_crs) if gdf.crs != raster_crs else gdf.copy()

    # Build bbox polygon and bbox GeoDataFrame
    bbox_polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_polygon], crs=raster_crs)

    # Clip (this keeps geometry types as intersections; may yield empty geometries)
    clipped = gpd.clip(gdf_in, bbox_gdf)

    # Optional: drop empties explicitly (clip usually does this)
    clipped = clipped[~clipped.geometry.is_empty].copy()

    # Optional: if you only want specific geometry types, filter here
    # Example for polygons only:
    # clipped = clipped[clipped.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    return clipped
