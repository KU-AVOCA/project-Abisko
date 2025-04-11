var palettes = require('users/gena/packages:palettes');

var resultsPath ='projects/ku-gem/assets/ccdcAbisko2014-2021';

// Obtain water mask from RC Global Surface Water Mapping Layers
var waterMask = ee.Image('JRC/GSW1_3/GlobalSurfaceWater')
    .select('max_extent').eq(0);

var ccdResults = ee.Image(resultsPath).updateMask(waterMask);
Map.centerObject(ccdResults, 10);
print(ccdResults);
var proj = 'EPSG:32634'; // UTM zone 34N for Abisko

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

// Select time of break and change probability array images.
var change = ccdResults.select('tBreak');
var changeProb = ccdResults.select('changeProb');

// Set the time range we want to use and get as mask of 
// places that meet the condition.
var start = 2014;
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
    .select('GCC_magnitude')
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


// Export the results and water mask to Google Drive
// Flatten the array image to a single band image
numBreaks = numBreaks.arrayFlatten([['numBreaks']]).toFloat();
Export.image.toDrive({
    image: numBreaks,
    description: 'numBreaks',
    folder:'Abisko',
    scale: 30,
    region: roi,
    maxPixels: 1e13,
    crs: proj
});
Export.image.toDrive({
    image: firstChange,
    description: 'firstChangeTime',
    folder:'Abisko',
    scale: 30,
    region: roi,
    maxPixels: 1e13,
    crs: proj
});
Export.image.toDrive({
    image: lastChange,
    description: 'lastChangeTime',
    folder:'Abisko',
    scale: 30,
    region: roi,
    maxPixels: 1e13,
    crs: proj
});
Export.image.toDrive({
    image: selectedMag,
    description: 'maxMagnitude',
    folder:'Abisko',
    scale: 30,
    region: roi,
    maxPixels: 1e13,
    crs: proj
});
Export.image.toDrive({
    image: selectedTbreak,
    description: 'timeOfMaxMagnitude',
    folder:'Abisko',
    scale: 30,
    region: roi,
    maxPixels: 1e13,
    crs: proj
});

Export.image.toDrive({
    image: waterMask,
    description: 'waterMask',
    folder:'Abisko',
    scale: 30,
    region: roi,
    maxPixels: 1e13,
    crs: proj
});