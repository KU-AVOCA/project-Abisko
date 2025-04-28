
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

var startDate = '2015-01-01';
var endDate = '2021-12-31';
var dateStart = ee.Date(startDate);
var dateEnd = ee.Date(endDate);

var bands = ['S1ratio']; // Green and SWIR1 are used for Tmask


// Function for band math
function bandMath(image) {
    
    var s1ratio = image.expression(
        'pow(10, VH / 10) / pow(10, VV / 10)', {
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
var s1asc = s1col.select(['VV', 'VH'])
    .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    .map(bandMath);
var s1desc = s1col.select(['VV', 'VH'])
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .map(bandMath);

var ccdcParams = {
    collection: s1desc,
    breakpointBands: bands,
    // tmaskBands: ['Green', 'SWIR1'],
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
    // metadata['tmaskBands'] = metadata['tmaskBands'].toString();
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