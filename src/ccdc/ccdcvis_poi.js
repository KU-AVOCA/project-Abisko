// Simple CCDC Time Series Visualization
var poi = ee.Geometry.Point([19.05077561, 68.34808742]);
var roi = poi.buffer(30).bounds();
Map.addLayer(poi, {color: 'red'}, 'poi');
Map.addLayer(roi, {color: 'black'}, 'roi');
Map.centerObject(poi, 8);

// Time period matching your CCDC analysis
var startDate = '2013-01-01';
var endDate = '2013-07-31';
var dateStart = ee.Date(startDate);
var dateEnd = ee.Date(endDate);

var bands = ['GCC', 'Green', 'SWIR1', 'Blue', 'Red']; // Green and SWIR1 are used for Tmask
var SEGS = ["S1", "S2", "S3", "S4", "S5", "S6"]

var dateFormat = 1;

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
    var cloudMask = qa.bitwiseAnd(1 << 1).eq(0);
    var adjacentCloudMask = qa.bitwiseAnd(1 << 2).eq(0);
    var cloudShadowMask = qa.bitwiseAnd(1 << 3).eq(0);
    var mask = cloudMask.and(adjacentCloudMask).and(cloudShadowMask);
    return image.updateMask(mask).divide(10000);
}

// Function for band math
function bandMath(image) {
    var GCC = image.expression(
        'Green / (Red + Green + Blue)', {
            'Green': image.select('Green'),
            'Red': image.select('Red'),
            'Blue': image.select('Blue')
        }).rename('GCC');
    return image.addBands(GCC);
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
      .set('system:time_start', filtered.first().get('system:time_start'));
  return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist));
};

var hslDaily = ee.ImageCollection(ee.List(range.iterate(day_mosaics, ee.List([]))));

// Load the CCDC results
var ccdcAsset = ee.Image('projects/ku-gem/assets/ccdcAbisko20250408');
var ccdcImage = utils.CCDC.buildCcdImage(ccdcAsset, SEGS.length, 'GCC');
// create image collection of synthetic images

var day_synthetic = function(date, newlist) {
  date = ee.Date(date);
  var inputDate = date.format('YYYY-MM-dd');
  var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1};
  var formattedDate = utils.Dates.convertDate(dateParams);

  var syntheticImage = utils.CCDC.getSyntheticForYear(
    ccdcImage, formattedDate, 1, 'GCC', SEGS
  );
  
  newlist = ee.List(newlist);
  
  return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(syntheticImage), newlist));
}

var syntheticDaily = ee.ImageCollection(ee.List(range.iterate(day_synthetic, ee.List([]))));
print(syntheticDaily, 'syntheticDaily');