// Simple CCDC Time Series Visualization 
// Plot E1 has highest BVOC emission in 2023
var poi = ee.Geometry.Point([19.0524377, 68.34836531]);
// var roi = poi.buffer(30).bounds();
var roi = 
    /* color: #d63000 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[18.650628517410595, 68.37246412370075],
          [18.650628517410595, 68.28955676896352],
          [19.180032203934033, 68.28955676896352],
          [19.180032203934033, 68.37246412370075]]], null, false);
Map.addLayer(poi, {color: 'red'}, 'poi');
Map.addLayer(roi, {color: 'black'}, 'roi');
Map.centerObject(poi, 8);
var proj = 'EPSG:32634'; // UTM zone 34N for Abisko

// Time period matching your CCDC analysis
var startDate = '2023-06-25';
var endDate = '2023-06-30';
var dateStart = ee.Date(startDate);
var dateEnd = ee.Date(endDate);

var bands = ['S1ratio']; // Green and SWIR1 are used for Tmask
// var SEGS = ["S1", "S2", "S3", "S4", "S5", "S6"]



var utils = require('users/parevalo_bu/gee-ccdc-tools:ccdcUtilities/api');
// var ui_util = require('users/parevalo_bu/gee-ccdc-tools:ccdcUtilities/ui');

// Function for band math
function bandMath(image) {
    
  var s1ratio = image.expression(
      'VH / VV', {
          'VH': image.select('VH'),
          'VV': image.select('VV')
      }).rename('S1ratio');

  return image.addBands(s1ratio).copyProperties(image);
}

/**
 * Load Data
 */
var s1col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .map(function(image) {
        var edge = image.select('VV').lt(-30);
        var maskedImage = image.mask().and(edge.not());
        return image.updateMask(maskedImage);
    });
// var s1col = s1col.select(['VV', 'VH'])
//     .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
//     .map(bandMath);
var s1col = s1col.select(['VV', 'VH'])
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .map(bandMath);

// Convert to daily average
var diff = dateEnd.difference(dateStart, 'day');
var dayNum = 1;
var range = ee.List.sequence(0, diff.subtract(1), dayNum).map(function(day){
  return dateStart.advance(day,'day');
});

var day_mosaics = function(date, newlist) {
  date = ee.Date(date);
  newlist = ee.List(newlist);
  var filtered = s1col.filterDate(date, date.advance(dayNum,'day'));
  var image = ee.Image(
      filtered.mean().copyProperties(filtered.first()))
      .set('system:index', date.format('yyyy-MM-dd'))
      .set('system:time_start', filtered.first().get('system:time_start'));
  return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist));
};

var hlsDaily = ee.ImageCollection(ee.List(range.iterate(day_mosaics, ee.List([]))));

// Load the CCDC results
var ccdcAsset = ee.Image('projects/ku-gem/assets/AbiskoS1descCCDC'); // S1ratio, R, G, B, SWIR1, Tmask applied
var ccdcImage = utils.CCDC.buildCcdImage(ccdcAsset, 1, ['S1ratio']);
// create image collection of synthetic images

var day_synthetic = function(date, newlist) {
  date = ee.Date(date);
  var inputDate = date.format('YYYY-MM-dd');
  var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1};
  var formattedDate = utils.Dates.convertDate(dateParams);

  var syntheticImage = utils.CCDC.getSyntheticForYear(
    ccdcImage, formattedDate, 1, 'S1ratio', 'S1'
  ).set('system:time_start', date.millis())
   .set('system:index', date.format('yyyy-MM-dd'))
   .rename('S1ratio_predicted');

  var coefImage = utils.CCDC.getMultiCoefs(
    ccdcImage, formattedDate, bands, ['INTP', 'SLP', 'RMSE'], true, ['S1'], 'after'
  ).set('system:time_start', date.millis())
   .set('system:index', date.format('yyyy-MM-dd'));
  var rmseImage = coefImage.select('S1ratio_RMSE')
  .set('system:time_start', date.millis())
   .set('system:index', date.format('yyyy-MM-dd'));
  
  newlist = ee.List(newlist);
  var newImage = syntheticImage.addBands(rmseImage);

  return ee.List(newlist.add(newImage));
};


var syntheticDaily = ee.ImageCollection(ee.List(range.iterate(day_synthetic, ee.List([]))));

var imgCollection = syntheticDaily.linkCollection(
  hlsDaily, 'S1ratio'
);


// Batch export images to Google Drive
// var batch = require('users/fitoprincipe/geetools:batch');

// batch.Download.ImageCollection.toDrive(
//   imgCollection.select(['S1ratio_predicted', 'S1ratio_RMSE', 'S1ratio', 'conditionScore']),
//   'Abisko',
//   {
//   crs: proj,
//   region: roi, 
//   type: 'uint16',
//   maxPixels: 1e13,
//   name: 'GEMEST_Landsat_{system_date}'
//   }
// );

// optional: add layers to map, choose day 2023-06-27
var palettes = require('users/gena/packages:palettes');
var imgshow = imgCollection.filterDate('2023-06-27', '2023-06-28').mean();
Map.addLayer(imgshow, {bands: ['S1ratio_predicted'], min: -20, max: 20, palette: palettes.colorbrewer.YlOrRd[9]}, 'S1ratio_predicted');
Map.addLayer(imgshow, {bands: ['S1ratio'], min: -20, max: 20, palette: palettes.colorbrewer.YlOrRd[9]}, 'S1ratio');
// Map.addLayer(imgshow, {bands: ['conditionScore'], min: -4, max: 4, palette: palettes.cmocean.Balance[7]}, 'conditionScore');
// Map.addLayer(imgCollection.first(), {bands: ['S1ratio_predicted', 'S1ratio'], min: 0, max: 1}, 'imgCollection');