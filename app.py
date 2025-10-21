from flask import Flask, render_template, jsonify, request
import ee
import logging
from functools import wraps
import os
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')

# Simplified logging - only INFO and above
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create logs directory
os.makedirs('logs', exist_ok=True)
os.makedirs('cache', exist_ok=True)

CONFIG = {
    'gee_project': os.environ.get('GEE_PROJECT', 'ee-vincentkipyegon'),
    'county_dataset': 'projects/ee-vincentkipyegon/assets/KenyaAdminCountyLevel',
    'subcounty_dataset': 'projects/ee-vincentkipyegon/assets/SubCountyLevel',
    'mapbox_token': os.environ.get('MAPBOX_ACCESS_TOKEN')
}


def initialize_gee(max_retries=3):
    """Initialize GEE with service account credentials"""
    credentials_json = os.environ.get('GEE_CREDENTIALS')
    credentials_file = os.environ.get('GEE_CREDENTIALS_FILE', 'gee-credentials.json')

    if not credentials_json and not os.path.exists(credentials_file):
        logger.critical("GEE credentials not found")
        return False

    logger.info("Initializing GEE...")

    for attempt in range(max_retries):
        try:
            if credentials_json:
                credentials_dict = json.loads(credentials_json)
            else:
                with open(credentials_file, 'r') as f:
                    credentials_dict = json.load(f)

            credentials = ee.ServiceAccountCredentials(
                email=credentials_dict['client_email'],
                key_data=credentials_dict['private_key']
            )

            ee.Initialize(
                credentials=credentials,
                project=CONFIG['gee_project'],
                opt_url='https://earthengine-highvolume.googleapis.com'
            )

            # Quick test
            ee.Image('NASA/NASADEM_HGT/001').getInfo()
            logger.info(f"✓ GEE initialized successfully")
            return True

        except Exception as e:
            logger.error(f"GEE init attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    logger.critical("✗ GEE initialization failed")
    return False


GEE_INITIALIZED = initialize_gee()


def handle_gee_operation(f):
    """Decorator for GEE operations"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not GEE_INITIALIZED:
            return {'error': 'GEE not initialized'}, 503
        try:
            start = time.time()
            result = f(*args, **kwargs)
            logger.info(f"{f.__name__} completed in {time.time() - start:.1f}s")
            return result
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            return {'error': str(e)}, 500

    return wrapper


def safe_get_info(ee_object, default=None):
    """Safely get info from EE object"""
    try:
        result = ee_object.getInfo()
        return result if result is not None else default
    except Exception as e:
        logger.warning(f"Failed to get info: {str(e)}")
        return default


@handle_gee_operation
def calculate_area_fast(image, roi, scale=150):
    """Fast area calculation with optimizations"""
    try:
        image = image.clip(roi)

        # Quick pixel count
        count_dict = image.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        )

        count = safe_get_info(count_dict.values().get(0), 0)
        if count == 0:
            return 0.0

        # Calculate area
        area_dict = ee.Image.pixelArea().multiply(image).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        )

        area = safe_get_info(area_dict.values().get(0), 0)
        return round(area / 1e6, 2) if area else 0.0

    except Exception as e:
        logger.error(f"Area calculation error: {str(e)}")
        return 0.0


@handle_gee_operation
def get_vis_url(image, palette, min_val=0, max_val=1):
    """Generate visualization URL"""
    vis_params = {
        'min': min_val,
        'max': max_val,
        'palette': palette if isinstance(palette, list) else [palette]
    }
    map_id = image.visualize(**vis_params).getMapId()
    return map_id['tile_fetcher'].url_format


@handle_gee_operation
def load_datasets_optimized(roi, start_date, end_date):
    """Load all datasets with optimizations"""
    logger.info(f"Loading datasets for {start_date} to {end_date}")

    # Sentinel-1
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .select('VV')

    s1_count = safe_get_info(s1.size(), 0)
    logger.info(f"Found {s1_count} S1 images")

    # Fallback if no images
    if s1_count == 0:
        logger.warning("Using wider date range")
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(roi) \
            .filterDate('2023-01-01', '2024-12-31') \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .select('VV') \
            .sort('system:time_start', False) \
            .limit(30)

    # Other datasets
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').clip(roi)
    buildings = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons') \
        .filterBounds(roi).filter(ee.Filter.gte('confidence', 0.75))
    landcover = ee.Image('ESA/WorldCover/v200/2021').select('Map').clip(roi)

    return s1, jrc, buildings, landcover


@handle_gee_operation
def process_flood_optimized(s1, jrc, landcover, roi):
    """Process flood data with optimizations"""
    logger.info("Processing flood data")

    # Farmland
    farmland = landcover.eq(40).selfMask()

    # Flood detection with adaptive threshold
    s1_mean = s1.mean()

    # Fast stats
    stats = s1_mean.reduceRegion(
        reducer=ee.Reducer.percentile([10]),
        geometry=roi,
        scale=200,
        maxPixels=1e9,
        bestEffort=True
    )

    threshold = safe_get_info(stats.get('VV_p10'), -12)
    logger.info(f"Threshold: {threshold:.2f}")

    # Detect floods
    flood_extent = s1_mean.lt(threshold).clip(roi)
    permanent_water = jrc.gte(70)
    flooded = flood_extent.And(permanent_water.Not()).selfMask().clip(roi)
    flooded_farm = farmland.And(flooded).selfMask()

    return farmland, flooded, flooded_farm


@handle_gee_operation
def process_risk_map_optimized(roi, start_date, end_date):
    """Process flood risk map - optimized"""
    logger.info("Processing risk map")

    SCALE = 250  # Larger scale for speed

    # Load data
    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterDate(start_date, end_date).sum().clip(roi)
    dem = ee.Image('USGS/SRTMGL1_003').clip(roi)
    slope = ee.Terrain.slope(dem)
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').clip(roi)
    water = jrc.gte(70)
    dist_water = water.fastDistanceTransform().sqrt() \
        .multiply(ee.Image.pixelArea().sqrt()).clip(roi)
    lulc = ee.Image('ESA/WorldCover/v200/2021').clip(roi)

    # Get percentiles
    def get_percentiles(img, pcts):
        return img.reduceRegion(
            reducer=ee.Reducer.percentile(pcts),
            geometry=roi,
            scale=SCALE,
            maxPixels=1e13,
            bestEffort=True
        ).values()

    dem_pct = get_percentiles(dem, [0, 20, 40, 60, 80, 100])
    slope_pct = get_percentiles(slope, [0, 20, 40, 60, 80, 100])
    precip_pct = get_percentiles(chirps, [0, 20, 40, 60, 80, 100])
    dist_pct = get_percentiles(dist_water, [0, 20, 40, 60, 80, 100])

    # Reclassify
    def reclass(img, classes, reverse=False):
        out = ee.Image(0)
        weights = [2, 4, 6, 8, 10] if reverse else [10, 8, 6, 4, 2]
        for i in range(1, 6):
            lower = ee.Number(classes.get(i - 1))
            upper = ee.Number(classes.get(i))
            out = out.where(img.gt(lower).And(img.lte(upper)), weights[i - 1])
        return out

    dem_r = reclass(dem, dem_pct)
    slope_r = reclass(slope, slope_pct)
    precip_r = reclass(chirps, precip_pct, True)
    dist_r = reclass(dist_water, dist_pct)

    # Land use reclassification
    lulc_r = ee.Image(0) \
        .where(lulc.eq(10), 2) \
        .where(lulc.eq(20), 4) \
        .where(lulc.eq(30), 4) \
        .where(lulc.eq(40), 6) \
        .where(lulc.eq(50), 10) \
        .where(lulc.eq(60), 8) \
        .where(lulc.eq(80), 10) \
        .where(lulc.eq(90), 6) \
        .where(lulc.eq(95), 6)

    # Combine with weights
    risk = dem_r.multiply(0.10) \
        .add(slope_r.multiply(0.20)) \
        .add(precip_r.multiply(0.30)) \
        .add(dist_r.multiply(0.30)) \
        .add(lulc_r.multiply(0.10)) \
        .clip(roi)

    # Get stats
    stats = risk.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=roi,
        scale=SCALE,
        maxPixels=1e9,
        bestEffort=True
    )

    risk_stats = safe_get_info(stats, {})
    logger.info(f"Risk stats: {risk_stats}")

    return risk, risk_stats


@handle_gee_operation
def get_flood_data(county, start_date, end_date):
    """Main function to get flood data"""
    logger.info(f"Processing {county}")

    # Check cache (24 hour TTL)
    cache_key = f"{county}_{start_date}_{end_date}"
    cache_file = Path(f"cache/{cache_key}.json")

    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 86400:
            logger.info(f"Cache hit for {cache_key}")
            with open(cache_file) as f:
                data = json.load(f)
                data['metadata']['cached'] = True
                return data

    # Get ROI
    counties = ee.FeatureCollection(CONFIG['county_dataset']) \
        .filter(ee.Filter.eq('ADM1_EN', county))

    if safe_get_info(counties.size(), 0) == 0:
        return {'error': f'County {county} not found'}, 404

    roi = counties.geometry()
    center = safe_get_info(counties.geometry().centroid().coordinates(), [36, 0])

    # Load datasets
    s1, jrc, buildings, landcover = load_datasets_optimized(roi, start_date, end_date)

    # Process floods
    farmland, flooded, flooded_farm = process_flood_optimized(s1, jrc, landcover, roi)

    # Calculate areas (parallel where possible)
    logger.info("Calculating areas")
    flooded_area = calculate_area_fast(flooded, roi, scale=150)
    farmland_area = calculate_area_fast(farmland, roi, scale=150)
    flooded_farm_area = calculate_area_fast(flooded_farm, roi, scale=150)
    total_area = calculate_area_fast(ee.Image(1), roi, scale=200)

    # Buildings
    total_buildings = safe_get_info(buildings.size(), 0)
    flood_proportion = flooded_area / total_area if total_area > 0 else 0
    flooded_buildings = int(total_buildings * flood_proportion)

    # Risk map
    logger.info("Generating risk map")
    risk, risk_stats = process_risk_map_optimized(roi, start_date, end_date)

    risk_min = risk_stats.get('constant_min', 0)
    risk_max = risk_stats.get('constant_max', 10)

    # Generate URLs
    flood_url = get_vis_url(flooded, ['#FF0000'])
    farm_url = get_vis_url(farmland, ['#00FF00'])
    flooded_farm_url = get_vis_url(flooded_farm, ['#FFA500'])
    risk_url = get_vis_url(
        risk,
        ['#0000FF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000'],
        risk_min,
        risk_max
    )

    result = {
        'flooded_area_km2': flooded_area,
        'total_buildings': total_buildings,
        'flooded_buildings': flooded_buildings,
        'farmland_area_km2': farmland_area,
        'flooded_farmland_area_km2': flooded_farm_area,
        'flood_layer_url': flood_url,
        'farmland_layer_url': farm_url,
        'flooded_farmland_url': flooded_farm_url,
        'flood_risk_url': risk_url,
        'county_center': center,
        'flooded_buildings_geojson': {'type': 'FeatureCollection', 'features': []},
        'metadata': {
            'processing_time': datetime.now().isoformat(),
            'gee_initialized': GEE_INITIALIZED,
            'cached': False,
            'risk_stats': risk_stats
        }
    }

    # Cache result
    try:
        cache_file.parent.mkdir(exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(result, f)
        logger.info(f"Cached {cache_key}")
    except Exception as e:
        logger.warning(f"Cache write failed: {str(e)}")

    logger.info(f"✓ Results: Flooded={flooded_area}km², Farm={farmland_area}km², "
                f"Flooded Farm={flooded_farm_area}km²")

    return result


# Routes
@app.route('/')
def index():
    """Main page"""
    try:
        if not GEE_INITIALIZED:
            return render_template('index.html', counties=[],
                                   error="GEE not initialized",
                                   mapbox_token=CONFIG['mapbox_token'])

        counties_fc = ee.FeatureCollection(CONFIG['county_dataset'])
        counties = safe_get_info(
            counties_fc.aggregate_array('ADM1_EN').distinct().sort(),
            []
        )

        return render_template('index.html', counties=counties,
                               mapbox_token=CONFIG['mapbox_token'])
    except Exception as e:
        logger.error(f"Index error: {str(e)}")
        return render_template('index.html', counties=[],
                               error=str(e),
                               mapbox_token=CONFIG['mapbox_token'])


@app.route('/flood-data', methods=['POST'])
def flood_data():
    """Flood data endpoint"""
    try:
        data = request.get_json() or {
            'county': 'Baringo',
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        }

        result = get_flood_data(data['county'], data['start_date'], data['end_date'])

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]

        return jsonify(result)

    except Exception as e:
        logger.error(f"Flood data error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/county-boundary', methods=['POST'])
def county_boundary():
    """County boundary endpoint"""
    try:
        if not GEE_INITIALIZED:
            return jsonify({'error': 'GEE not initialized'}), 503

        data = request.get_json() or {'county': 'Baringo'}

        counties = ee.FeatureCollection(CONFIG['county_dataset']) \
            .filter(ee.Filter.eq('ADM1_EN', data['county']))

        return jsonify(safe_get_info(counties, {}))

    except Exception as e:
        logger.error(f"Boundary error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/subcounty-labels', methods=['POST'])
def subcounty_labels():
    """Subcounty labels endpoint"""
    try:
        if not GEE_INITIALIZED:
            return jsonify({'error': 'GEE not initialized'}), 503

        data = request.get_json() or {'county': 'Baringo'}

        county_geom = ee.FeatureCollection(CONFIG['county_dataset']) \
            .filter(ee.Filter.eq('ADM1_EN', data['county'])).geometry()

        subcounties = ee.FeatureCollection(CONFIG['subcounty_dataset']) \
            .filterBounds(county_geom)

        return jsonify(safe_get_info(subcounties, {}))

    except Exception as e:
        logger.error(f"Subcounty error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'gee': GEE_INITIALIZED,
        'time': datetime.now().isoformat()
    })


@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear cache endpoint"""
    try:
        cache_dir = Path('cache')
        count = 0
        for file in cache_dir.glob('*.json'):
            file.unlink()
            count += 1
        return jsonify({'cleared': count, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)