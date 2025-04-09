//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  Chapter:      F4.7 Interpreting Time Series with CCDC
//  Checkpoint:   F47b
//  Authors:      Paulo Arévalo, Pontus Olofsson
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

var utils = require(
    'users/parevalo_bu/gee-ccdc-tools:ccdcUtilities/api');

var poi = ee.Geometry.Point([19.05077561, 68.34808742]);
var studyRegion = poi.buffer(10000).bounds();
Map.addLayer(studyRegion, {color: 'red'}, 'studyRegion');
Map.centerObject(studyRegion, 8);

// Define start, end dates and Landsat bands to use.
var startDate = '2000-01-01';
var endDate = '2020-01-01';
var bands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'];

// Retrieve all clear, Landsat 4, 5, 7 and 8 observations (Collection 2, Tier 1).
var filteredLandsat = utils.Inputs.getLandsat({
        collection: 2
    })
    .filterBounds(studyRegion)
    .filterDate(startDate, endDate)
    .select(bands);

print(filteredLandsat.first());

// Set CCD params to use.
var ccdParams = {
    breakpointBands: ['GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'],
    tmaskBands: ['GREEN', 'SWIR1'],
    minObservations: 6,
    chiSquareProbability: 0.99,
    minNumOfYearsScaler: 1.33,
    dateFormat: 1,
    lambda: 0.002,
    maxIterations: 10000,
    collection: filteredLandsat
};

// Run CCD.
var ccdResults = ee.Algorithms.TemporalSegmentation.Ccdc(ccdParams);
print(ccdResults);

var exportResults = true;
if (exportResults) {
    // Create a metadata dictionary with the parameters and arguments used.
    var metadata = ccdParams;
    metadata['breakpointBands'] = metadata['breakpointBands'].toString();
    metadata['tmaskBands'] = metadata['tmaskBands'].toString();
    metadata['startDate'] = startDate;
    metadata['endDate'] = endDate;
    metadata['bands'] = bands.toString();

    // Export results, assigning the metadata as image properties.
    // 
    Export.image.toAsset({
        image: ccdResults.set(metadata),
        region: studyRegion,
        pyramidingPolicy: {
            ".default": 'sample'
        },
        scale: 30
    });
}

//  -----------------------------------------------------------------------
//  CHECKPOINT 
//  -----------------------------------------------------------------------

//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  Chapter:      F4.7 Interpreting Time Series with CCDC
//  Checkpoint:   F47c
//  Authors:      Paulo Arévalo, Pontus Olofsson
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

var palettes = require('users/gena/packages:palettes');

var resultsPath =
    'projects/ku-gem/assets/ccdcTest1';
var ccdResults = ee.Image(resultsPath);
Map.centerObject(ccdResults, 10);
print(ccdResults);

// Select time of break and change probability array images.
var change = ccdResults.select('tBreak');
var changeProb = ccdResults.select('changeProb');

// Set the time range we want to use and get as mask of 
// places that meet the condition.
var start = 2000;
var end = 2021;
var mask = change.gt(start).and(change.lte(end)).and(changeProb.eq(
1));
Map.addLayer(changeProb, {}, 'change prob');

// Obtain the number of breaks for the time range.
var numBreaks = mask.arrayReduce(ee.Reducer.sum(), [0]);
Map.addLayer(numBreaks, {
    min: 0,
    max: 5
}, 'Number of breaks');

// Obtain the first change in that time period.
var dates = change.arrayMask(mask).arrayPad([1]);
var firstChange = dates
    .arraySlice(0, 0, 1)
    .arrayFlatten([
        ['firstChange']
    ])
    .selfMask();

var timeVisParams = {
    palette: palettes.colorbrewer.YlOrRd[9],
    min: start,
    max: end
};
Map.addLayer(firstChange, timeVisParams, 'First change');

// Obtain the last change in that time period.
var lastChange = dates
    .arraySlice(0, -1)
    .arrayFlatten([
        ['lastChange']
    ])
    .selfMask();
Map.addLayer(lastChange, timeVisParams, 'Last change');

// Get masked magnitudes.
var magnitudes = ccdResults
    .select('SWIR1_magnitude')
    .arrayMask(mask)
    .arrayPad([1]);

// Get index of max abs magnitude of change.
var maxIndex = magnitudes
    .abs()
    .arrayArgmax()
    .arrayFlatten([
        ['index']
    ]);

// Select max magnitude and its timing
var selectedMag = magnitudes.arrayGet(maxIndex);
var selectedTbreak = dates.arrayGet(maxIndex).selfMask();

var magVisParams = {
    palette: palettes.matplotlib.viridis[7],
    min: -0.15,
    max: 0.15
};
Map.addLayer(selectedMag, magVisParams, 'Max mag');
Map.addLayer(selectedTbreak, timeVisParams, 'Time of max mag');

//  -----------------------------------------------------------------------
//  CHECKPOINT 
//  -----------------------------------------------------------------------
 