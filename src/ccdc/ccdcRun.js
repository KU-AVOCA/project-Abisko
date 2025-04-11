
var poi = ee.Geometry.Point([19.05077561, 68.34808742]);
// var roi = poi.buffer(10000).bounds();
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

var startDate = '2014-01-01';
var endDate = '2021-12-31';
var dateStart = ee.Date(startDate);
var dateEnd = ee.Date(endDate);

var bands = ['GCC', 'Green', 'SWIR1', 'Blue', 'Red']; // Green and SWIR1 are used for Tmask

/**
 * Functions for preprocessing Landsat and Sentinel-2 data
 */

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

    return image.addBands(GCC).copyProperties(image);
}

/**
 * Load Data
 */
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

// // convert to daily average
// // Difference in days between start and finish
// var diff = dateEnd.difference(dateStart, 'day');

// // Make a list of all dates
// var dayNum = 1; // steps of day number
// var range = ee.List.sequence(0, diff.subtract(1), dayNum).map(function(day){return dateStart.advance(day,'day')});

// // Function for iteration over the range of dates
// var day_mosaics = function(date, newlist) {
//   // Cast
//   date = ee.Date(date);
//   newlist = ee.List(newlist);

//   // Filter collection between date and the next day
//   var filtered = hls.filterDate(date, date.advance(dayNum,'day'));
//   // Make the mosaic
//   var image = ee.Image(
//       filtered.mean().copyProperties(filtered.first()))
//     //   .set({'system:index': date.format('yyyy_MM_dd')})
//       .set('system:time_start', filtered.first().get('system:time_start'));

//   // Add the mosaic to a list only if the collection has images
//   return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist));
// };

// var hslDaily = ee.ImageCollection(ee.List(range.iterate(day_mosaics, ee.List([]))));
// Map.addLayer(hslDaily.first(), {bands: ['GCC'], min: 0, max: 1}, 'hslDaily');
/**
 * Run CCDC
 */
// set CCDC parameters
var ccdcParams = {
    collection: hls,
    breakpointBands: bands,
    tmaskBands: ['Green', 'SWIR1'],
    minObservations: 6,
    chiSquareProbability: 0.99,
    minNumOfYearsScaler: 1.33,
    dateFormat: 1,
    lambda: 0.002,
    maxIterations: 10000
};
// Run CCDC
var ccdcResults = ee.Algorithms.TemporalSegmentation.Ccdc(ccdcParams);
// print(ccdcResults);

// Export results, assigning the metadata as image properties.
var exportResults = true;
if (exportResults) {
    // Create a metadata dictionary with the parameters and arguments used.
    var metadata = ccdcParams;
    metadata['breakpointBands'] = metadata['breakpointBands'].toString();
    metadata['tmaskBands'] = metadata['tmaskBands'].toString();
    metadata['startDate'] = startDate;
    metadata['endDate'] = endDate;
    metadata['bands'] = bands.toString();

    // Export results, assigning the metadata as image properties.
    // 
    Export.image.toAsset({
        image: ccdcResults.set(metadata),
        region: roi,
        pyramidingPolicy: {
            ".default": 'sample'
        },
        scale: 30
    });
}