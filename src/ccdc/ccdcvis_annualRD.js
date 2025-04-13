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
// function to calcuate the defoliation intensity
function scoreCalculation(image) {
  var relativeDeviation = image.expression(
      '(GCC - GCC_predicted) / GCC_predicted', {
          'GCC_predicted': image.select('GCC_predicted'),
          'GCC': image.select('GCC')
      }).rename('relativeDeviation');

  return image.addBands(relativeDeviation).copyProperties(image, ['system:time_start']);
}

var colFilter = ee.Filter.and(
  ee.Filter.bounds(roi),
  ee.Filter.date(dateStart, dateEnd),
  ee.Filter.calendarRange(6, 8, 'month')
);

// Load and process data
var hlsL = ee.ImageCollection('NASA/HLS/HLSL30/v002')
    .filter(colFilter)
    .map(renamHLSL30)
    .map(maskhls)
    .map(bandMath);

var hlsS = ee.ImageCollection('NASA/HLS/HLSS30/v002')
    .filter(colFilter)
    .map(renamHLSS30)
    .map(maskhls)
    .map(bandMath);

var hls = hlsL.merge(hlsS).select(bands);

// prepare an image collection of daily hls images and add predicted GCC
// and condition score

// Load the CCDC results
var ccdcAsset = ee.Image('projects/ku-gem/assets/ccdcAbisko2014-2021'); // GCC, R, G, B, SWIR1, Tmask applied
var ccdcImage = utils.CCDC.buildCcdImage(ccdcAsset, 1, ['GCC']);

// Convert to daily average
var diff = dateEnd.difference(dateStart, 'day');
var dayNum = 1;
var range = ee.List.sequence(0, diff.subtract(1), dayNum).map(function(day){
  return dateStart.advance(day,'day');
});

var day_mosaics = function(date, newlist) {
  date = ee.Date(date);
  var inputDate = date.format('YYYY-MM-dd');
  var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1};
  var formattedDate = utils.Dates.convertDate(dateParams);

  newlist = ee.List(newlist);
  var filtered = hls.filterDate(date, date.advance(dayNum,'day'));
  var image = ee.Image(
      filtered.mean().copyProperties(filtered.first()))
      .set('system:index', date.format('yyyy-MM-dd'))
      .set('system:time_start', filtered.first().get('system:time_start'));
  
  var syntheticImage = utils.CCDC.getSyntheticForYear(
    ccdcImage, formattedDate, 1, 'GCC', 'S1'
  ).set('system:time_start', date.millis())
   .set('system:index', date.format('yyyy-MM-dd'))
   .rename('GCC_predicted');

  // var coefImage = utils.CCDC.getMultiCoefs(
  //   ccdcImage, formattedDate, bands, ['INTP', 'SLP', 'RMSE'], true, ['S1'], 'after'
  // ).set('system:time_start', date.millis())
  //  .set('system:index', date.format('yyyy-MM-dd'));
  // var rmseImage = coefImage.select('GCC_RMSE')
  // .set('system:time_start', date.millis())
  //  .set('system:index', date.format('yyyy-MM-dd'));

  var newImage = syntheticImage.addBands(image);

  return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(newImage), newlist));
};

var hlsDaily = ee.ImageCollection(ee.List(range.iterate(day_mosaics, ee.List([]))));


var imgCollection = hlsDaily.map(scoreCalculation).select('relativeDeviation');

// calculate the annual average of relative deviation
// Difference in years between start and end date
var years = dateEnd.difference(dateStart, 'year');
// Create a list of years
var yearNum = 1; // 1 year interval
var yearRange = ee.List.sequence(0, years.subtract(1), yearNum).map(function(year){
  return dateStart.advance(year,'year');
});

// Function to calculate the annual average of relative deviation
var annualAverage = function(year, newlist) {
  year = ee.Date(year);
  newlist = ee.List(newlist);

  var filtered = imgCollection.filterDate(year, year.advance(1,'year'));
  var image = ee.Image(
      filtered.mean().copyProperties(filtered.first()))
      .set('system:index', year.format('yyyy'))
      .set('system:time_start', filtered.first().get('system:time_start'));

  return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist));
};

var annualCollection = ee.ImageCollection(ee.List(yearRange.iterate(annualAverage, ee.List([]))));
var relativeDeviationSTD = annualCollection.select('relativeDeviation').reduce(ee.Reducer.stdDev());

var annualCollection = annualCollection.map(function(image) {
  var imgtime = image.get('system:time_start');
  var annualImage = image.abs();
  var mask = annualImage.gt(relativeDeviationSTD);
  var newAnnualImage = image.updateMask(mask).rename('relativeDeviation_masked');
  
  return image.addBands(newAnnualImage)
      .copyProperties(image)
      .set('system:time_start', imgtime);
});

// Export the annual average of relative deviation year by year and the relativeDeviationSTD
Export.image.toDrive({
  image: relativeDeviationSTD,
  folder: 'Abisko',
  description: 'relativeDeviationSTD',
  scale: 30,
  region: roi,
  crs: proj,
  maxPixels: 1e13
});
// Export the annual average of relative deviation year by year
Export.image.toDrive({
  image: annualCollection.filter(ee.Filter.calendarRange(2014, 2014, 'year')).first(),
  folder: 'Abisko',
  description: 'annualRelativeDeviation2014',
  scale: 30,
  region: roi,
  crs: proj,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: annualCollection.filter(ee.Filter.calendarRange(2015, 2015, 'year')).first(),
  folder: 'Abisko',
  description: 'annualRelativeDeviation2015',
  scale: 30,
  region: roi,
  crs: proj,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: annualCollection.filter(ee.Filter.calendarRange(2016, 2016, 'year')).first(),
  folder: 'Abisko',
  description: 'annualRelativeDeviation2016',
  scale: 30,
  region: roi,
  crs: proj,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: annualCollection.filter(ee.Filter.calendarRange(2017, 2017, 'year')).first(),
  folder: 'Abisko',
  description: 'annualRelativeDeviation2017',
  scale: 30,
  region: roi,
  crs: proj,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: annualCollection.filter(ee.Filter.calendarRange(2018, 2018, 'year')).first(),
  folder: 'Abisko',
  description: 'annualRelativeDeviation2018',
  scale: 30,
  region: roi,
  crs: proj,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: annualCollection.filter(ee.Filter.calendarRange(2019, 2019, 'year')).first(),
  folder: 'Abisko',
  description: 'annualRelativeDeviation2019',
  scale: 30,
  region: roi,
  crs: proj,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: annualCollection.filter(ee.Filter.calendarRange(2020, 2020, 'year')).first(),
  folder: 'Abisko',
  description: 'annualRelativeDeviation2020',
  scale: 30,
  region: roi,
  crs: proj,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: annualCollection.filter(ee.Filter.calendarRange(2021, 2021, 'year')).first(),
  folder: 'Abisko',
  description: 'annualRelativeDeviation2021',
  scale: 30,
  region: roi,
  crs: proj,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: annualCollection.filter(ee.Filter.calendarRange(2022, 2022, 'year')).first(),
  folder: 'Abisko',
  description: 'annualRelativeDeviation2022',
  scale: 30,
  region: roi,
  crs: proj,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: annualCollection.filter(ee.Filter.calendarRange(2023, 2023, 'year')).first(),
  folder: 'Abisko',
  description: 'annualRelativeDeviation2023',
  scale: 30,
  region: roi,
  crs: proj,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: annualCollection.filter(ee.Filter.calendarRange(2024, 2024, 'year')).first(),
  folder: 'Abisko',
  description: 'annualRelativeDeviation2024',
  scale: 30,
  region: roi,
  crs: proj,
  maxPixels: 1e13
});
// Batch export images to Google Drive
// var batch = require('users/fitoprincipe/geetools:batch');


// batch.Download.ImageCollection.toDrive(
//   imgCollection.select(['relativeDeviation']),
//   'Abisko',
//   {
//    crs: proj,
//    region: roi, 
//    type: 'double',
//    scale: 30,
//    maxPixels: 1e13,
//    name: 'Annual_Relative_Deviation_Abisko_{system_date}'
//   }
// );


// optional: add layers to map, choose day 2023-06-27
// var palettes = require('users/gena/packages:palettes');
// var imgshow = imgCollection.filterDate('2023-06-27', '2023-06-28').mean();
// Map.addLayer(imgshow, {bands: ['GCC_predicted'], min: 0, max: 1, palette: palettes.colorbrewer.YlOrRd[9]}, 'GCC_predicted');
// Map.addLayer(imgshow, {bands: ['GCC'], min: 0, max: 1, palette: palettes.colorbrewer.YlOrRd[9]}, 'GCC');
// Map.addLayer(imgshow, {bands: ['conditionScore'], min: -4, max: 4, palette: palettes.cmocean.Balance[7]}, 'conditionScore');
