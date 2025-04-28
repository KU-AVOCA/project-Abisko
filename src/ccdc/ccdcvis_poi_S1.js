// Simple CCDC Time Series Visualization 
// Plot E1 has highest BVOC emission in 2023
var poi = ee.Geometry.Point([19.0524377, 68.34836531]);
var roi = poi.buffer(30).bounds();
Map.addLayer(poi, {color: 'red'}, 'poi');
Map.addLayer(roi, {color: 'black'}, 'roi');
Map.centerObject(poi, 8);

// Time period matching your CCDC analysis
var startDate = '2015-01-01';
var endDate = '2024-12-31';
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

// // Convert to daily average
var diff = dateEnd.difference(dateStart, 'day');
var dayNum = 1;
var range = ee.List.sequence(0, diff.subtract(1), dayNum).map(function(day){
  return dateStart.advance(day,'day');
});

// var day_mosaics = function(date, newlist) {
//   date = ee.Date(date);
//   newlist = ee.List(newlist);
//   var filtered = hls.filterDate(date, date.advance(dayNum,'day'));
//   var image = ee.Image(
//       filtered.mean().copyProperties(filtered.first()))
//       .set('system:index', date.format('yyyy-MM-dd'))
//       .set('system:time_start', filtered.first().get('system:time_start'));
//   return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist));
// };

// var hlsDaily = ee.ImageCollection(ee.List(range.iterate(day_mosaics, ee.List([]))));

// Load the CCDC results
var ccdcAsset = ee.Image('projects/ku-gem/assets/AbiskoS1descCCDC'); // GCC, R, G, B, SWIR1, Tmask applied
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
  
  newlist = ee.List(newlist);

  return ee.List(newlist.add(syntheticImage));
}


var syntheticDaily = ee.ImageCollection(ee.List(range.iterate(day_synthetic, ee.List([]))));

// var imgcollection = syntheticDaily.linkCollection(
//   hlsDaily, 'GCC'
// ).select(['GCC_predicted', 'GCC']);

// Export the time series data to Google Drive
// Note: You can modify the export parameters as per your requirements
// var allObs = imgcollection.map(function(image) {
//   var obs = image.reduceRegion({
//     reducer: ee.Reducer.mean(),
//     geometry: poi,
//     scale: 30,
//     maxPixels: 1e13
//   });
//   return image.set('GCC', obs.get('GCC'))
//               .set('GCC_predicted', obs.get('GCC_predicted'))
//               // .set('system:index', image.get('system:index'))
//               .set('system:time_start', image.get('system:time_start'));
// }
// );

// var timeSeries = ee.Feature(null, allObs);
// Export.table.toDrive({
//   collection: timeSeries,
//   description: 'E1_GCC_GCC_predicted_2013_2024',
//   folder: 'Abisko',
//   fileFormat: 'CSV'
// });

// create a time series chart
// var chart = ui.Chart.image.seriesByRegion({
//   imageCollection: imgcollection,
//   regions: roi,
//   reducer: ee.Reducer.mean(),
//   scale: 30,
//   xProperty: 'system:time_start'
// }).setOptions({
//   title: 'GCC Time Series',
//   lineWidth: 1,
//   pointSize: 3,
//   series: {
//     0: { type: 'line', color: 'blue' }, // GCC_predicted as line
//     1: { type: 'scatter', color: 'red' }  // GCC as scatter
//   }
// });
// print(chart);
// print(syntheticDaily, 'syntheticDaily');

// Create a time series chart
var chart = ui.Chart.image.seriesByRegion({
  imageCollection: syntheticDaily.select('S1ratio_predicted'),
  regions: roi,
  reducer: ee.Reducer.mean(),
  scale: 30,
  xProperty: 'system:time_start'
}).setOptions({
  title: 'Synthetic S1ratio Time Series',
  lineWidth: 1,
  pointSize: 3,
  // seriesProperty: 'system:time_start'
});
print(chart);

// Create a time series chart for the original data
var chartOriginal = ui.Chart.image.seriesByRegion({
  imageCollection: s1col.select('S1ratio'),
  // bandName: 'GCC',
  regions: roi,
  reducer: ee.Reducer.mean(),
  scale: 30,
  xProperty: 'system:time_start'
}).setChartType('ScatterChart')
  .setOptions({
  title: 'Original S1ratio Time Series',
  // lineWidth: 1,
  // lineWidth: 1,
  pointSize: 3,
  // seriesProperty: 'system:index'
});
print(chartOriginal);