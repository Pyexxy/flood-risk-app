from flask import Flask, render_template, jsonify, request
import ee
import logging
from functools import wraps
import os
import json
import time
import traceback
import googleapiclient.errors
import requests.exceptions
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a file handler for error logs
if not os.path.exists('logs'):
    os.makedirs('logs')
file_handler = logging.FileHandler('logs/error.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)

CONFIG = {
    'gee_project': 'ee-vincentkipyegon',
    'county_dataset': 'projects/ee-vincentkipyegon/assets/KenyaAdminCountyLevel',
    'subcounty_dataset': 'projects/ee-vincentkipyegon/assets/SubCountyLevel',
    'mapbox_token': os.environ.get('MAPBOX_ACCESS_TOKEN')
}

def log_environment():
    """Log critical environment information for debugging"""
    logger.info("Environment Variables:")
    logger.info(f"GEE_PROJECT: {CONFIG['gee_project']}")
    logger.info(f"MAPBOX_TOKEN_SET: {bool(CONFIG['mapbox_token'])}")
    logger.info(f"FLASK_ENV: {os.getenv('FLASK_ENV', 'production')}")
    logger.info(f"PORT: {os.getenv('PORT', '5000')}")

def initialize_gee(max_retries=3, backoff_factor=2):
    """Enhanced GEE initialization with detailed logging"""
    credentials_json = os.environ.get('GOOGLE_EARTH_ENGINE_CREDENTIALS')
    
    if not credentials_json:
        logger.critical("GOOGLE_EARTH_ENGINE_CREDENTIALS environment variable not set")
        return False
    
    logger.info("Attempting GEE initialization...")
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Initialization attempt {attempt + 1}/{max_retries}")
            
            # Write credentials to temporary file for debugging
            with open('temp_gee_credentials.json', 'w') as f:
                f.write(credentials_json)
            
            credentials_dict = json.loads(credentials_json)
            credentials = ee.ServiceAccountCredentials(
                email=credentials_dict['client_email'],
                key_data=credentials_json
            )
            
            # Enhanced initialization with timeout
            ee.Initialize(
                credentials=credentials,
                project=CONFIG['gee_project'],
                opt_url='https://earthengine-highvolume.googleapis.com',
                http_timeout=30
            )
            
            # Test connection with a simple operation
            test_image = ee.Image('NASA/NASADEM_HGT/001')
            test_info = test_image.getInfo()
            logger.debug(f"GEE test successful. Image info: {str(test_info)[:100]}...")
            
            logger.info("GEE initialized successfully")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in credentials (attempt {attempt + 1}): {str(e)}")
        except ee.ee_exception.EEException as e:
            logger.error(f"GEE API error (attempt {attempt + 1}): {str(e)}")
        except googleapiclient.errors.HttpError as e:
            logger.error(f"Google API HTTP error (attempt {attempt + 1}): {str(e)}")
            logger.debug(f"HTTP error details: {e.content.decode()}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error (attempt {attempt + 1}): {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during initialization (attempt {attempt + 1}): {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        if attempt < max_retries - 1:
            wait_time = backoff_factor ** attempt
            logger.info(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    logger.critical("GEE initialization failed after all retries")
    return False

GEE_INITIALIZED = initialize_gee()
log_environment()

@app.route('/debug/gee-status')
def gee_status():
    """Endpoint to check GEE connection status"""
    status = {
        'timestamp': datetime.now().isoformat(),
        'gee_initialized': GEE_INITIALIZED,
        'environment': {
            'gee_project': CONFIG['gee_project'],
            'mapbox_configured': bool(CONFIG['mapbox_token'])
        }
    }
    
    if GEE_INITIALIZED:
        try:
            # Test simple GEE operation
            test_image = ee.Image('NASA/NASADEM_HGT/001')
            test_info = test_image.getInfo()
            status['gee_test'] = {
                'status': 'success',
                'image_id': test_info['id'] if test_info else None
            }
        except Exception as e:
            status['gee_test'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"GEE test failed: {str(e)}")
    
    return jsonify(status)

def log_ge_error(method_name, error, extra_info=None):
    """Standardized error logging for GEE operations"""
    error_details = {
        'method': method_name,
        'error': str(error),
        'type': type(error).__name__,
        'timestamp': datetime.now().isoformat()
    }
    
    if extra_info:
        error_details.update(extra_info)
    
    logger.error(json.dumps(error_details, indent=2))
    return error_details

def handle_gee_operation(f):
    """Decorator to handle GEE operations with standardized error handling"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not GEE_INITIALIZED:
            error_msg = "Google Earth Engine not initialized"
            logger.error(error_msg)
            return {'error': error_msg, 'initialized': False}, 503
        
        try:
            start_time = time.time()
            logger.debug(f"Starting GEE operation: {f.__name__}")
            
            result = f(*args, **kwargs)
            
            duration = time.time() - start_time
            logger.debug(f"Completed GEE operation: {f.__name__} in {duration:.2f}s")
            
            return result
            
        except ee.ee_exception.EEException as e:
            error_details = log_ge_error(
                f.__name__,
                e,
                {'args': args, 'kwargs': kwargs}
            )
            return {'error': 'GEE API error', 'details': error_details}, 500
        except Exception as e:
            error_details = log_ge_error(
                f.__name__,
                e,
                {'traceback': traceback.format_exc()}
            )
            return {'error': 'Processing error', 'details': error_details}, 500
    
    return wrapper

@handle_gee_operation
def calculate_area(image, roi, scale=30, max_pixels=1e9):
    """Enhanced area calculation with detailed logging"""
    logger.debug(f"Calculating area for image: {image.getInfo()['id']}")
    
    area_image = ee.Image.pixelArea()
    area_result = area_image.multiply(image).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=scale,
        maxPixels=max_pixels
    )
    
    area = area_result.get('area').getInfo() / 1e6
    logger.debug(f"Area calculation complete: {area} km²")
    
    return round(area, 2)

@handle_gee_operation
def get_visualization_url(image, palette, min_val=0, max_val=1):
    """Enhanced visualization URL generation"""
    logger.debug(f"Generating visualization URL for image: {image.getInfo()['id']}")
    
    vis_params = {
        'min': min_val,
        'max': max_val,
        'palette': palette
    }
    
    map_id = image.visualize(**vis_params).getMapId()
    url = map_id['tile_fetcher'].url_format
    
    logger.debug(f"Generated visualization URL: {url[:100]}...")
    return url

@handle_gee_operation
def load_datasets(roi, start_date):
    """Enhanced dataset loading with progress logging"""
    logger.info(f"Loading datasets for ROI and date: {start_date}")
    
    # Sentinel-1
    logger.debug("Loading Sentinel-1 data...")
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(roi) \
        .filterDate(start_date, ee.Date(start_date).advance(1, 'month')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .select('VV') \
        .limit(10)
    
    # JRC Water
    logger.debug("Loading JRC water data...")
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').clip(roi)
    
    # Open Buildings
    logger.debug("Loading Open Buildings data...")
    open_buildings = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons') \
        .filterBounds(roi).filter(ee.Filter.gte('confidence', 0.75))
    
    # Land Cover
    logger.debug("Loading ESA WorldCover data...")
    land_cover = ee.Image('ESA/WorldCover/v100/2020').select('Map').clip(roi)
    
    logger.info("All datasets loaded successfully")
    return s1, jrc, open_buildings, land_cover

@handle_gee_operation
def process_flood_data(s1, jrc, land_cover, roi):
    """Enhanced flood data processing with detailed logging"""
    logger.info("Processing flood data...")
    
    # Farmland classification
    logger.debug("Identifying farmland...")
    farmland = land_cover.eq(40).selfMask()
    
    # Flood extent
    logger.debug("Calculating flood extent...")
    flood_extent = s1.mean().lt(-15).clip(roi)
    
    # Permanent water
    logger.debug("Identifying permanent water...")
    permanent_water = jrc.gt(90)
    
    # Flooded areas (excluding permanent water)
    logger.debug("Calculating flooded areas...")
    flooded_areas = flood_extent.subtract(permanent_water).selfMask().clip(roi)
    
    # Flooded farmland
    logger.debug("Calculating flooded farmland...")
    flooded_farmland = farmland.updateMask(flooded_areas)
    
    logger.info("Flood data processing complete")
    return farmland, flooded_areas, flooded_farmland

@handle_gee_operation
def process_flood_risk_map(roi, start_date, end_date):
    """Enhanced flood risk processing with detailed logging"""
    logger.info(f"Processing flood risk map from {start_date} to {end_date}")
    
    # Precipitation data
    logger.debug("Processing CHIRPS precipitation data...")
    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterDate(start_date, end_date) \
        .sum() \
        .clip(roi)
    
    # Terrain data
    logger.debug("Processing DEM and slope data...")
    dem = ee.Image('USGS/SRTMGL1_003').clip(roi)
    slope = ee.Terrain.slope(dem)
    
    # Water data
    logger.debug("Processing JRC water data...")
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').clip(roi)
    water = jrc.select('occurrence').gte(90)
    distance_to_water = water.fastDistanceTransform().sqrt().multiply(30).clip(roi)
    
    # Land cover
    logger.debug("Processing land cover data...")
    lulc = ee.ImageCollection('ESA/WorldCover/v200').first().clip(roi)
    
    # Calculate class breaks for each factor
    logger.debug("Calculating class breaks...")
    def get_class_breaks(image, percentiles):
        return image.reduceRegion(
            reducer=ee.Reducer.percentile(percentiles),
            geometry=roi,
            scale=30,
            maxPixels=1e13
        ).values()
    
    dem_classes = get_class_breaks(dem, [0, 20, 40, 60, 80, 100])
    slope_classes = get_class_breaks(slope, [0, 20, 40, 60, 80, 100])
    precip_classes = get_class_breaks(chirps, [0, 20, 40, 60, 80, 100])
    distance_classes = get_class_breaks(distance_to_water, [0, 20, 40, 60, 80, 100])
    
    # Reclassify each factor
    logger.debug("Reclassifying factors...")
    def reclassify(image, classes, inverse=False):
        reclass = ee.Image(0)
        weights = [10, 8, 6, 4, 2] if not inverse else [2, 4, 6, 8, 10]
        
        for i in range(1, 6):
            lower = classes.getNumber(i-1)
            upper = classes.getNumber(i)
            reclass = reclass.where(
                image.gt(lower).And(image.lte(upper)),
                weights[i-1]
            )
        
        return reclass
    
    dem_reclass = reclassify(dem, dem_classes)
    slope_reclass = reclassify(slope, slope_classes)
    precip_reclass = reclassify(chirps, precip_classes, inverse=True)
    distance_reclass = reclassify(distance_to_water, distance_classes)
    
    lulc_reclass = lulc \
        .where(lulc.eq(10), 4) \
        .where(lulc.eq(20), 6) \
        .where(lulc.eq(30), 6) \
        .where(lulc.eq(40), 6) \
        .where(lulc.eq(50), 10) \
        .where(lulc.eq(60), 8) \
        .where(lulc.eq(95), 6)
    
    # Apply weights and combine
    logger.debug("Combining factors with weights...")
    weighted_factors = [
        dem_reclass.multiply(0.10),    # Elevation
        slope_reclass.multiply(0.20),  # Slope
        precip_reclass.multiply(0.30), # Precipitation
        distance_reclass.multiply(0.30),# Distance to water
        lulc_reclass.multiply(0.10)    # Land cover
    ]
    
    flood_risk = ee.Image(0)
    for factor in weighted_factors:
        flood_risk = flood_risk.add(factor)
    
    flood_risk = flood_risk.clip(roi)
    logger.info("Flood risk map processing complete")
    
    return flood_risk

@handle_gee_operation
def get_flood_data(county_name, start_date, end_date):
    """Enhanced flood data retrieval with detailed logging"""
    logger.info(f"Getting flood data for {county_name} ({start_date} to {end_date})")
    
    # Get county geometry
    logger.debug(f"Fetching county geometry for {county_name}")
    counties = ee.FeatureCollection(CONFIG['county_dataset']) \
        .filter(ee.Filter.eq('ADM1_EN', county_name))
    
    if counties.size().getInfo() == 0:
        error_msg = f"No counties found for {county_name}"
        logger.error(error_msg)
        return {'error': error_msg}, 404
    
    roi = counties.geometry()
    logger.debug(f"ROI geometry: {roi.getInfo()}")
    
    # Load datasets
    s1, jrc, open_buildings, land_cover = load_datasets(roi, start_date)
    
    # Process flood data
    farmland, flooded_areas, flooded_farmland = process_flood_data(s1, jrc, land_cover, roi)
    
    # Calculate areas
    logger.debug("Calculating areas...")
    flooded_area = calculate_area(flooded_areas, roi, scale=30)
    farmland_area = calculate_area(farmland, roi, scale=10)
    flooded_farmland_area = calculate_area(flooded_farmland, roi, scale=10)
    
    # Building statistics
    logger.debug("Calculating building statistics...")
    total_buildings = open_buildings.size().getInfo()
    total_area = calculate_area(ee.Image(1), roi, scale=30)
    flood_area_proportion = flooded_area / total_area if total_area > 0 else 0
    flooded_buildings = int(total_buildings * flood_area_proportion)
    
    # Flood risk map
    logger.debug("Generating flood risk map...")
    flood_risk_map = process_flood_risk_map(roi, start_date, end_date)
    flood_risk_url = get_visualization_url(
        flood_risk_map, 
        ['blue', 'green', 'yellow', 'orange', 'red'], 
        min_val=2, 
        max_val=10
    )
    
    # Prepare results
    result = {
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
        'flood_risk_url': flood_risk_url,
        'metadata': {
            'processing_time': datetime.now().isoformat(),
            'gee_initialized': GEE_INITIALIZED
        }
    }
    
    logger.info(f"Successfully processed flood data for {county_name}")
    return result

@app.route('/')
def index():
    """Enhanced index route with better error handling"""
    try:
        if not GEE_INITIALIZED:
            logger.warning("GEE not initialized - serving limited index page")
            return render_template(
                'index.html',
                counties=[],
                error="Google Earth Engine not initialized",
                mapbox_token=CONFIG['mapbox_token']
            )
        
        logger.debug("Loading counties for index page")
        counties_fc = ee.FeatureCollection(CONFIG['county_dataset'])
        
        if counties_fc.size().getInfo() == 0:
            logger.error("County dataset is empty")
            return render_template(
                'index.html',
                counties=[],
                error="County dataset is empty",
                mapbox_token=CONFIG['mapbox_token']
            )
        
        counties = counties_fc.aggregate_array('ADM1_EN').distinct().sort().getInfo()
        
        if not counties:
            logger.warning("No counties found in dataset")
            return render_template(
                'index.html',
                counties=[],
                error="No counties found in dataset",
                mapbox_token=CONFIG['mapbox_token']
            )
        
        logger.debug(f"Loaded {len(counties)} counties for index page")
        return render_template(
            'index.html',
            counties=counties,
            mapbox_token=CONFIG['mapbox_token']
        )
        
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return render_template(
            'index.html',
            counties=[],
            error=f"Failed to load counties: {str(e)}",
            mapbox_token=CONFIG['mapbox_token']
        )

@app.route('/flood-data', methods=['POST'])
def flood_data():
    """Enhanced flood data endpoint with detailed logging"""
    try:
        request_data = request.get_json() or {
            'county': 'Baringo',
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        }
        
        logger.info(f"Flood data request: {request_data}")
        
        result = get_flood_data(
            request_data['county'],
            request_data['start_date'],
            request_data['end_date']
        )
        
        if 'error' in result:
            logger.error(f"Error in flood data response: {result['error']}")
            return jsonify(result), result.get('status_code', 500)
        
        return jsonify(result)
        
    except Exception as e:
        error_details = log_ge_error(
            'flood_data_endpoint',
            e,
            {'request_data': request.get_json()}
        )
        return jsonify({
            'error': 'Failed to process flood data request',
            'details': error_details
        }), 500

@app.route('/subcounty-labels', methods=['POST'])
def subcounty_labels():
    """Enhanced subcounty labels endpoint"""
    try:
        if not GEE_INITIALIZED:
            error_msg = "Google Earth Engine not initialized"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 503
        
        request_data = request.get_json() or {'county': 'Baringo'}
        logger.info(f"Subcounty labels request: {request_data}")
        
        county_geom = ee.FeatureCollection(CONFIG['county_dataset']) \
            .filter(ee.Filter.eq('ADM1_EN', request_data['county'])) \
            .geometry()
        
        subcounties = ee.FeatureCollection(CONFIG['subcounty_dataset']) \
            .filterBounds(county_geom)
        
        subcounty_data = subcounties.getInfo()
        logger.debug(f"Fetched {len(subcounty_data['features'])} subcounty features")
        
        return jsonify(subcounty_data)
        
    except Exception as e:
        error_details = log_ge_error(
            'subcounty_labels',
            e,
            {'request_data': request.get_json()}
        )
        return jsonify({
            'error': 'Failed to load subcounty labels',
            'details': error_details
        }), 500

@app.route('/county-boundary', methods=['POST'])
def county_boundary():
    """Enhanced county boundary endpoint"""
    try:
        if not GEE_INITIALIZED:
            error_msg = "Google Earth Engine not initialized"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 503
        
        request_data = request.get_json() or {'county': 'Baringo'}
        logger.info(f"County boundary request: {request_data}")
        
        counties = ee.FeatureCollection(CONFIG['county_dataset']) \
            .filter(ee.Filter.eq('ADM1_EN', request_data['county']))
        
        county_data = counties.getInfo()
        logger.debug(f"Fetched county boundary for {request_data['county']}")
        
        return jsonify(county_data)
        
    except Exception as e:
        error_details = log_ge_error(
            'county_boundary',
            e,
            {'request_data': request.get_json()}
        )
        return jsonify({
            'error': 'Failed to load county boundary',
            'details': error_details
        }), 500

@app.route('/logs')
def view_logs():
    """Endpoint to view recent error logs"""
    try:
        with open('logs/error.log', 'r') as f:
            logs = f.read().splitlines()[-100:]  # Get last 100 lines
        return jsonify({'logs': logs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port)
