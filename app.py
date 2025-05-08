from flask import Flask, render_template, jsonify, request
import ee
import logging
from functools import wraps
import os
import json
import time
import googleapiclient.errors
import requests.exceptions

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

CONFIG = {
    'gee_project': 'ee-vincentkipyegon',
    'county_dataset': 'projects/ee-vincentkipyegon/assets/KenyaAdminCountyLevel',
    'subcounty_dataset': 'projects/ee-vincentkipyegon/assets/SubCountyLevel',
    'mapbox_token': os.environ.get('MAPBOX_ACCESS_TOKEN')
}

# Initialize GEE with retry logic
def initialize_gee(max_retries=3, backoff_factor=2):
    credentials_json = os.environ.get('GOOGLE_EARTH_ENGINE_CREDENTIALS')
    if not credentials_json:
        logger.warning("GOOGLE_EARTH_ENGINE_CREDENTIALS not set. GEE routes will fail.")
        return False
    
    for attempt in range(max_retries):
        try:
            credentials_dict = json.loads(credentials_json)
            credentials = ee.ServiceAccountCredentials(
                email=credentials_dict['client_email'],
                key_data=credentials_json
            )
            ee.Initialize(credentials=credentials, project=CONFIG['gee_project'])
            logger.debug("Google Earth Engine initialized successfully")
            return True
        except (ee.ee_exception.EEException, googleapiclient.errors.HttpError, requests.exceptions.RequestException) as e:
            logger.error(f"GEE initialization attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(backoff_factor ** attempt)
            else:
                logger.warning("Max retries reached. GEE routes will fail.")
                return False
        except Exception as e:
            logger.error(f"Unexpected error in GEE initialization: {str(e)}")
            return False

GEE_INITIALIZED = initialize_gee()

def calculate_area(image, roi, scale=30, max_pixels=1e9):
    try:
        area_image = ee.Image.pixelArea()
        area = area_image.multiply(image).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=roi, scale=scale, maxPixels=max_pixels
        ).get('area').getInfo() / 1e6
        return round(area, 2)
    except Exception as e:
        logger.error(f"Error calculating area: {str(e)}")
        raise

def get_visualization_url(image, palette, min_val=0, max_val=1):
    try:
        return image.visualize(**{'min': min_val, 'max': max_val, 'palette': palette}).getMapId()['tile_fetcher'].url_format
    except Exception as e:
        logger.error(f"Error generating visualization URL: {str(e)}")
        raise

def load_datasets(roi, start_date):
    try:
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(roi) \
            .filterDate(start_date, ee.Date(start_date).advance(1, 'month')) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .select('VV') \
            .limit(10)
        jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').clip(roi)
        open_buildings = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons') \
            .filterBounds(roi).filter(ee.Filter.gte('confidence', 0.75))
        land_cover = ee.Image('ESA/WorldCover/v100/2020').select('Map').clip(roi)
        return s1, jrc, open_buildings, land_cover
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise

def process_flood_data(s1, jrc, land_cover, roi):
    try:
        farmland = land_cover.eq(40).selfMask()
        flood_extent = s1.mean().lt(-15).clip(roi)
        permanent_water = jrc.gt(90)
        flooded_areas = flood_extent.subtract(permanent_water).selfMask().clip(roi)
        flooded_farmland = farmland.updateMask(flooded_areas)
        return farmland, flooded_areas, flooded_farmland
    except Exception as e:
        logger.error(f"Error processing flood data: {str(e)}")
        raise

def process_flood_risk_map(roi, start_date, end_date):
    try:
        chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
            .filterDate(start_date, end_date) \
            .sum() \
            .clip(roi)
        dem = ee.Image('USGS/SRTMGL1_003').clip(roi)
        slope = ee.Terrain.slope(dem)
        jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').clip(roi)
        water = jrc.select('occurrence').gte(90)
        distance_to_water = water.fastDistanceTransform().sqrt().multiply(30).clip(roi)
        lulc = ee.ImageCollection('ESA/WorldCover/v200').first().clip(roi)
        dem_classes = dem.reduceRegion(
            reducer=ee.Reducer.percentile([0, 20, 40, 60, 80, 100]),
            geometry=roi,
            scale=30,
            maxPixels=1e13
        ).values()
        slope_classes = slope.reduceRegion(
            reducer=ee.Reducer.percentile([0, 20, 40, 60, 80, 100]),
            geometry=roi,
            scale=30,
            maxPixels=1e13
        ).values()
        precip_classes = chirps.reduceRegion(
            reducer=ee.Reducer.percentile([0, 20, 40, 60, 80, 100]),
            geometry=roi,
            scale=30,
            maxPixels=1e13
        ).values()
        distance_classes = distance_to_water.reduceRegion(
            reducer=ee.Reducer.percentile([0, 20, 40, 60, 80, 100]),
            geometry=roi,
            scale=30,
            maxPixels=1e13
        ).values()
        dem_reclass = ee.Image(0) \
            .where(dem.lte(dem_classes.getNumber(1)), 10) \
            .where(dem.gt(dem_classes.getNumber(1)).And(dem.lte(dem_classes.getNumber(2))), 8) \
            .where(dem.gt(dem_classes.getNumber(2)).And(dem.lte(dem_classes.getNumber(3))), 6) \
            .where(dem.gt(dem_classes.getNumber(3)).And(dem.lte(dem_classes.getNumber(4))), 4) \
            .where(dem.gt(dem_classes.getNumber(4)), 2)
        slope_reclass = ee.Image(0) \
            .where(slope.lte(slope_classes.getNumber(1)), 10) \
            .where(slope.gt(slope_classes.getNumber(1)).And(slope.lte(slope_classes.getNumber(2))), 8) \
            .where(slope.gt(slope_classes.getNumber(2)).And(slope.lte(slope_classes.getNumber(3))), 6) \
            .where(slope.gt(slope_classes.getNumber(3)).And(slope.lte(slope_classes.getNumber(4))), 4) \
            .where(slope.gt(slope_classes.getNumber(4)), 2)
        precip_reclass = ee.Image(0) \
            .where(chirps.lte(precip_classes.getNumber(1)), 2) \
            .where(chirps.gt(precip_classes.getNumber(1)).And(chirps.lte(precip_classes.getNumber(2))), 4) \
            .where(chirps.gt(precip_classes.getNumber(2)).And(chirps.lte(precip_classes.getNumber(3))), 6) \
            .where(chirps.gt(precip_classes.getNumber(3)).And(chirps.lte(precip_classes.getNumber(4))), 8) \
            .where(chirps.gt(precip_classes.getNumber(4)), 10)
        distance_reclass = ee.Image(0) \
            .where(distance_to_water.lte(distance_classes.getNumber(1)), 10) \
            .where(distance_to_water.gt(distance_classes.getNumber(1)).And(distance_to_water.lte(distance_classes.getNumber(2))), 8) \
            .where(distance_to_water.gt(distance_classes.getNumber(2)).And(distance_to_water.lte(distance_classes.getNumber(3))), 6) \
            .where(distance_to_water.gt(distance_classes.getNumber(3)).And(distance_to_water.lte(distance_classes.getNumber(4))), 4) \
            .where(distance_to_water.gt(distance_classes.getNumber(4)), 2)
        lulc_reclass = lulc \
            .where(lulc.eq(10), 4) \
            .where(lulc.eq(20), 6) \
            .where(lulc.eq(30), 6) \
            .where(lulc.eq(40), 6) \
            .where(lulc.eq(50), 10) \
            .where(lulc.eq(60), 8) \
            .where(lulc.eq(95), 6)
        dem_weighted = dem_reclass.multiply(0.10)
        slope_weighted = slope_reclass.multiply(0.20)
        precip_weighted = precip_reclass.multiply(0.30)
        distance_weighted = distance_reclass.multiply(0.30)
        lulc_weighted = lulc_reclass.multiply(0.10)
        flood_risk = dem_weighted \
            .add(slope_weighted) \
            .add(precip_weighted) \
            .add(distance_weighted) \
            .add(lulc_weighted) \
            .clip(roi)
        return flood_risk
    except Exception as e:
        logger.error(f"Error processing flood risk map: {str(e)}")
        raise

def get_flood_data(county_name, start_date, end_date):
    if not GEE_INITIALIZED:
        logger.error("GEE not initialized in get_flood_data")
        return {'error': 'Google Earth Engine not initialized'}, 500
    logger.debug(f"Processing flood data for {county_name}, {start_date} to {end_date}")
    try:
        counties = ee.FeatureCollection(CONFIG['county_dataset']) \
            .filter(ee.Filter.eq('ADM1_EN', county_name))
        if counties.size().getInfo() == 0:
            logger.error(f"No counties found for {county_name}")
            return {'error': f'No counties found for {county_name}'}, 404
        roi = counties.geometry()
        s1, jrc, open_buildings, land_cover = load_datasets(roi, start_date)
        farmland, flooded_areas, flooded_farmland = process_flood_data(s1, jrc, land_cover, roi)
        flooded_area = calculate_area(flooded_areas, roi, scale=30)
        farmland_area = calculate_area(farmland, roi, scale=10)
        flooded_farmland_area = calculate_area(flooded_farmland, roi, scale=10)
        total_buildings = open_buildings.size().getInfo()
        total_area = calculate_area(ee.Image(1), roi, scale=30)
        flood_area_proportion = flooded_area / total_area if total_area > 0 else 0
        flooded_buildings = int(total_buildings * flood_area_proportion)
        flood_risk_map = process_flood_risk_map(roi, start_date, end_date)
        flood_risk_url = get_visualization_url(flood_risk_map, ['blue', 'green', 'yellow', 'orange', 'red'], min_val=2, max_val=10)
        return {
            'flooded_area_km2': flooded_area,
            'total_buildings': total_buildings,
            'flooded_buildings': flooded_buildings,
            'farmland_area_km2': farmland_area,
            'flooded_farmland_area_km2': flooded_farmland_area,
            'flood_layer_url': get_visualization_url(flooded_areas, 'red'),
            'farmland_layer_url': get_visualization_url(farmland, 'green'),
            'flooded_farmland_url': get_visualization_url(flooded_farmland, 'orange'),
            'flooded_buildings_geojson': {'type': 'FeatureCollection', 'features': []},
            'county_center': counties.geometry().centroid().coordinates().getInfo(),
            'flood_risk_url': flood_risk_url
        }
    except Exception as e:
        logger.error(f"Error in get_flood_data for {county_name}: {str(e)}")
        return {'error': f'Failed to process flood data: {str(e)}'}, 500

def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({'error': f'Error in {f.__name__}: {str(e)}'}), 500
    return decorated_function

@app.route('/')
def index():
    try:
        if not GEE_INITIALIZED:
            logger.warning("GEE not initialized. Serving index with empty counties.")
            return render_template('index.html', counties=[], error="Google Earth Engine not initialized")
        counties_fc = ee.FeatureCollection(CONFIG['county_dataset'])
        if counties_fc.size().getInfo() == 0:
            logger.error("County dataset is empty")
            return render_template('index.html', counties=[], error="County dataset is empty")
        counties = counties_fc.aggregate_array('ADM1_EN').distinct().sort().getInfo()
        if not counties:
            logger.warning("No ADM1_EN properties found in county dataset")
            return render_template('index.html', counties=[], error="No counties found in dataset")
        return render_template('index.html', counties=counties, mapbox_token=CONFIG['mapbox_token'])
    except Exception as e:
        logger.error(f"Error loading counties: {str(e)}")
        return render_template('index.html', counties=[], error=f"Failed to load counties: {str(e)}")

@app.route('/flood-data', methods=['POST'])
@handle_errors
def flood_data():
    data = request.get_json() or {'county': 'Baringo', 'start_date': '2024-01-01', 'end_date': '2024-12-31'}
    result = get_flood_data(data['county'], data['start_date'], data['end_date'])
    return jsonify(result)

@app.route('/subcounty-labels', methods=['POST'])
@handle_errors
def subcounty_labels():
    if not GEE_INITIALIZED:
        logger.error("GEE not initialized in subcounty_labels")
        return jsonify({'error': 'Google Earth Engine not initialized'}), 500
    data = request.get_json() or {'county': 'Baringo'}
    try:
        county_geom = ee.FeatureCollection(CONFIG['county_dataset']) \
            .filter(ee.Filter.eq('ADM1_EN', data['county'])).geometry()
        subcounties = ee.FeatureCollection(CONFIG['subcounty_dataset']) \
            .filterBounds(county_geom)
        subcounty_data = subcounties.getInfo()
        logger.debug(f"Subcounty data fetched: {len(subcounty_data['features'])} features")
        return jsonify(subcounty_data)
    except Exception as e:
        logger.error(f"Error in subcounty_labels for {data['county']}: {str(e)}")
        return jsonify({'error': f'Failed to load subcounty labels: {str(e)}'}), 500

@app.route('/county-boundary', methods=['POST'])
@handle_errors
def county_boundary():
    if not GEE_INITIALIZED:
        logger.error("GEE not initialized in county_boundary")
        return jsonify({'error': 'Google Earth Engine not initialized'}), 500
    data = request.get_json() or {'county': 'Baringo'}
    try:
        counties = ee.FeatureCollection(CONFIG['county_dataset']) \
            .filter(ee.Filter.eq('ADM1_EN', data['county']))
        county_data = counties.getInfo()
        logger.debug(f"Returning boundary for {data['county']}")
        return jsonify(county_data)
    except Exception as e:
        logger.error(f"Error in county_boundary for {data['county']}: {str(e)}")
        return jsonify({'error': f'Failed to load county boundary: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
