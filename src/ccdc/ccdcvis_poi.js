// Simple CCDC Time Series Visualization 
// Plot E1 has highest BVOC emission in 2023
var poi = ee.Geometry.Point([19.0524377, 68.34836531]);
var roi = poi.buffer(30).bounds();
Map.addLayer(poi, {color: 'red'}, 'poi');
Map.addLayer(roi, {color: 'black'}, 'roi');
Map.centerObject(poi, 8);

// Time period matching your CCDC analysis
var startDate = '2014-01-01';
var endDate = '2024-12-31';
var dateStart = ee.Date(startDate);
var dateEnd = ee.Date(endDate);

var bands = ['GCC']; // Green and SWIR1 are used for Tmask
// var SEGS = ["S1", "S2", "S3", "S4", "S5", "S6"]



var utils = require('users/parevalo_bu/gee-ccdc-tools:ccdcUtilities/api');
// var ui_util = require('users/parevalo_bu/gee-ccdc-tools:ccdcUtilities/ui');

// Function to select bands and rename them
function renamHLSS30(image) {
    return image.select(
        ['B2',   'B3',    'B4',  'B6',    'Fmask'],
        ['Blue', 'Green', 'Red', 'SWIR1', 'Fmask']
    );
}

function renamHLSL30(image) {
    return image.select(
        ['B2',   'B3',    'B4',  'B11',   'Fmask'],
        ['Blue', 'Green', 'Red', 'SWIR1', 'Fmask']
    );
}

// Function to mask clouds and shadows
function maskhls(image) {
  var qa = image.select('Fmask');
  var imgtime = image.get('system:time_start');

  var cloudMask = qa.bitwiseAnd(1 << 1).eq(0);
  var adjacentCloudMask = qa.bitwiseAnd(1 << 2).eq(0);
  var cloudShadowMask = qa.bitwiseAnd(1 << 3).eq(0);

  var mask = cloudMask.and(adjacentCloudMask).and(cloudShadowMask);

  return image.updateMask(mask).divide(10000).copyProperties(image)
      .set('system:time_start', imgtime);
}

// Function for band math
function bandMath(image) {
    var GCC = image.expression(
        'Green / (Red + Green + Blue)', {
            'Green': image.select('Green'),
            'Red': image.select('Red'),
            'Blue': image.select('Blue')
        }).rename('GCC');
    return image.addBands(GCC).copyProperties(image, ['system:time_start']);
}


// Load and process data
var hlsL = ee.ImageCollection('NASA/HLS/HLSL30/v002')
    .filterBounds(roi)
    .filterDate(dateStart, dateEnd)
    .map(renamHLSL30)
    .map(maskhls)
    .map(bandMath);

var hlsS = ee.ImageCollection('NASA/HLS/HLSS30/v002')
    .filterBounds(roi)
    .filterDate(dateStart, dateEnd)
    .map(renamHLSS30)
    .map(maskhls)
    .map(bandMath);

var hls = hlsL.merge(hlsS).select(bands);

// Convert to daily average
var diff = dateEnd.difference(dateStart, 'day');
var dayNum = 1;
var range = ee.List.sequence(0, diff.subtract(1), dayNum).map(function(day){
  return dateStart.advance(day,'day');
});

var day_mosaics = function(date, newlist) {
  date = ee.Date(date);
  newlist = ee.List(newlist);
  var filtered = hls.filterDate(date, date.advance(dayNum,'day'));
  var image = ee.Image(
      filtered.mean().copyProperties(filtered.first()))
      .set('system:index', date.format('yyyy-MM-dd'))
      .set('system:time_start', filtered.first().get('system:time_start'));
  return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist));
};

var hlsDaily = ee.ImageCollection(ee.List(range.iterate(day_mosaics, ee.List([]))));

// Load the CCDC results
var ccdcAsset = ee.Image('projects/ku-gem/assets/ccdcAbisko2014-2021'); // GCC, R, G, B, SWIR1, Tmask applied
var ccdcImage = utils.CCDC.buildCcdImage(ccdcAsset, 1, ['GCC']);
// create image collection of synthetic images

var day_synthetic = function(date, newlist) {
  date = ee.Date(date);
  var inputDate = date.format('YYYY-MM-dd');
  var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1};
  var formattedDate = utils.Dates.convertDate(dateParams);

  var syntheticImage = utils.CCDC.getSyntheticForYear(
    ccdcImage, formattedDate, 1, 'GCC', 'S1'
  ).set('system:time_start', date.millis())
   .set('system:index', date.format('yyyy-MM-dd'))
   .rename('GCC_predicted');
  
  newlist = ee.List(newlist);

  return ee.List(newlist.add(syntheticImage));
}

function getSyntheticForYear(image, date, dateFormat, band, segs) {
  var tfit = date
  var PI2 = 2.0 * Math.PI
  var OMEGAS = [PI2 / 365.25, PI2, PI2 / (1000 * 60 * 60 * 24 * 365.25)]
  var omega = OMEGAS[dateFormat];
  var imageT = ee.Image.constant([1, tfit,
                                tfit.multiply(omega).cos(),
                                tfit.multiply(omega).sin(),
                                tfit.multiply(omega * 2).cos(),
                                tfit.multiply(omega * 2).sin(),
                                tfit.multiply(omega * 3).cos(),
                                tfit.multiply(omega * 3).sin()]).float()
                                
  // OLD CODE
  // Casting as ee string allows using this function to be mapped
  // var selectString = ee.String('.*' + band + '_coef.*')
  // var params = getSegmentParamsForYear(image, date) 
  //                       .select(selectString)
  // return imageT.multiply(params).reduce('sum').rename(band)
                        
  // Use new standard functions instead
  var COEFS = ["INTP", "SLP", "COS", "SIN", "COS2", "SIN2", "COS3", "SIN3"]
  var newParams = utils.CCDC.getMultiCoefs(image, date, [band], COEFS, false, segs, 'after')
  return imageT.multiply(newParams).reduce('sum').rename(band)
  
}

var day_synthetic_pre = function(date, newlist) {
  date = ee.Date(date);
  var inputDate = date.format('YYYY-MM-dd');
  var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1};
  var formattedDate = utils.Dates.convertDate(dateParams);

  var syntheticImage = getSyntheticForYear(
    ccdcImage, formattedDate, 1, 'GCC', 'S1'
  ).set('system:time_start', date.millis())
   .set('system:index', date.format('yyyy-MM-dd'))
   .rename('GCC_predicted');
  
  newlist = ee.List(newlist);

  return ee.List(newlist.add(syntheticImage));
}

var syntheticDaily = ee.ImageCollection(ee.List(range.iterate(day_synthetic, ee.List([]))));
var syntheticDaily_pre = ee.ImageCollection(ee.List(range.iterate(day_synthetic_pre, ee.List([]))));

var imgcollection = syntheticDaily.linkCollection(
  hlsDaily, 'GCC'
).select(['GCC_predicted', 'GCC']);

// Export the time series data to Google Drive
// Note: You can modify the export parameters as per your requirements
var allObs = imgcollection.map(function(image) {
  var obs = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: poi,
    scale: 30,
    maxPixels: 1e13
  });
  return image.set('GCC', obs.get('GCC'))
              .set('GCC_predicted', obs.get('GCC_predicted'))
              // .set('system:index', image.get('system:index'))
              .set('system:time_start', image.get('system:time_start'));
}
);

var timeSeries = ee.Feature(null, allObs);
Export.table.toDrive({
  collection: timeSeries,
  description: 'E1_GCC_GCC_predicted_2013_2024',
  folder: 'Abisko',
  fileFormat: 'CSV'
});

// create a time series chart
var chart = ui.Chart.image.seriesByRegion({
  imageCollection: imgcollection,
  regions: roi,
  reducer: ee.Reducer.mean(),
  scale: 30,
  xProperty: 'system:time_start'
}).setOptions({
  title: 'GCC Time Series',
  lineWidth: 1,
  pointSize: 3,
  series: {
    0: { type: 'line', color: 'blue' }, // GCC_predicted as line
    1: { type: 'scatter', color: 'red' }  // GCC as scatter
  }
});
print(chart);
// print(syntheticDaily, 'syntheticDaily');

// Create a time series chart
var chartPredicted = ui.Chart.image.seriesByRegion({
  imageCollection: syntheticDaily.select('GCC_predicted'),
  regions: roi,
  reducer: ee.Reducer.mean(),
  scale: 30,
  xProperty: 'system:time_start'
}).setOptions({
  title: 'Synthetic GCC Time Series',
  lineWidth: 1,
  pointSize: 3,
  // seriesProperty: 'system:time_start'
});
print(chartPredicted);

var chartPredictedPre = ui.Chart.image.seriesByRegion({
  imageCollection: syntheticDaily_pre.select('GCC_predicted'),
  regions: roi,
  reducer: ee.Reducer.mean(),
  scale: 30,
  xProperty: 'system:time_start'
}).setOptions({
  title: 'Synthetic GCC Time Series (Pre)',
  lineWidth: 1,
  pointSize: 3,
});
print(chartPredictedPre);
// Create a time series chart for the original data
var chartOriginal = ui.Chart.image.seriesByRegion({
  imageCollection: hls.select('GCC'),
  // bandName: 'GCC',
  regions: roi,
  reducer: ee.Reducer.mean(),
  scale: 30,
  xProperty: 'system:time_start'
}).setChartType('ScatterChart')
  .setOptions({
  title: 'Original GCC Time Series',
  // lineWidth: 1,
  // lineWidth: 1,
  pointSize: 3,
  // seriesProperty: 'system:index'
});
print(chartOriginal);