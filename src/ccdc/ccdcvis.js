// CCDC Time Series Visualization
var poi = ee.Geometry.Point([19.05077561, 68.34808742]);
var roi = poi.buffer(10000).bounds();
Map.addLayer(poi, {color: 'red'}, 'poi');
Map.addLayer(roi, {color: 'black'}, 'roi');
Map.centerObject(poi, 8);

// Time period matching your CCDC analysis
var startDate = '2013-01-01';
var endDate = '2021-12-31';
var dateStart = ee.Date(startDate);
var dateEnd = ee.Date(endDate);
var bands = ['GCC', 'Green', 'SWIR1'];

// Load the utilities
var ccdc_util = require('users/parevalo_bu/gee-ccdc-tools:ccdcUtilities/ccdc.js');
var ui_util = require('users/parevalo_bu/gee-ccdc-tools:ccdcUtilities/ui');

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

// Load and process data - same as in your CCDC script
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

// Convert to daily average - same as in your CCDC script
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
// Adjust the asset path to where your CCDC results were saved
var ccdcAsset = ee.Image('projects/ku-gem/assets/ccdcAbisko20250408');

// Prepare for visualization
var nSegments = 6;  // Change this if your model has more/fewer segments
var dateFormat = 1; // Using fractional years as in your CCDC run

// Convert the CCDC array image to a more friendly format for visualization
var ccdImage = ccdc_util.buildCcdImage(ccdcAsset, nSegments, bands);

// Create panel for chart
var chartPanel = ui.Panel({
  style: {
    position: 'bottom',
    height: '300px',
    width: '600px'
  }
});
Map.add(chartPanel);

// Create UI for band selection
var bandSelect = ui.Select({
  items: bands,
  value: 'GCC',
  onChange: function(selected) {
    updateChart(selected);
  },
  style: {stretch: 'horizontal'}
});

var controlPanel = ui.Panel({
  widgets: [
    ui.Label('Select band to visualize:'),
    bandSelect
  ],
  style: {
    position: 'top-left'
  }
});
Map.add(controlPanel);

// Function to update the chart based on band selection
function updateChart(selectedBand) {
  // Generate time series suitable for charting
  var timeSeries = ui_util.ccdcTimeseries(
    hslDaily, 
    dateFormat, 
    ccdcAsset, 
    poi, 
    selectedBand, 
    0.1  // padding factor for smoother visualization
  );
  
  // List of properties needed for the chart
  var templist = ["dateString", "value"];
  for (var i = 0; i < nSegments; i++) {
    templist.push("h" + i);
  }
  templist.push("fit");
  
  // Create the data table for the chart
  var table = timeSeries.reduceColumns(ee.Reducer.toList(templist.length), templist)
                       .get('list');
  
  // Create and update the chart
  table.evaluate(function(t) {
    if (t && t.length > 0) {
      var chart = ui_util.chartTimeseries(t, selectedBand, poi.coordinates().get(1).getInfo(), poi.coordinates().get(0).getInfo(), nSegments);
      chartPanel.clear();
      chartPanel.add(chart);
    } else {
      chartPanel.clear();
      chartPanel.add(ui.Label('No data available for the selected band at this location.'));
    }
  });
}

// Initialize with default band (GCC)
updateChart('GCC');

// Add a date slider for generating synthetic images at different dates
var dateSlider = ui.DateSlider({
  start: startDate,
  end: endDate,
  value: startDate,
  period: 30, // Step size in days
  onChange: function(range) {
    var selectedDate = range.start();
    showSyntheticImage(selectedDate);
  },
  style: {
    position: 'bottom-right',
    width: '300px'
  }
});
Map.add(dateSlider);

// Function to show synthetic image at selected date
function showSyntheticImage(date) {
  var dateValue = ee.Date(date);
  var fractionalYear = dateValue.difference(ee.Date.fromYMD(dateValue.get('year'), 1, 1), 'year')
                      .add(dateValue.get('year'));
  
  // Generate synthetic image for selected date
  var synthetic = ccdc_util.getMultiSynthetic(
    ccdImage, 
    fractionalYear, 
    dateFormat, 
    bands,
    ee.List.sequence(1, nSegments).map(function(i) {
      return 'S' + i;
    })
  );
  
  // Display on map
  Map.layers().set(2, ui.Map.Layer(synthetic.clip(roi), {
    bands: ['GCC'], 
    min: 0, 
    max: 0.5,
    palette: ['blue', 'cyan', 'green', 'yellow', 'red']
  }, 'Synthetic ' + dateValue.format('YYYY-MM-dd').getInfo()));
}

// Add a legend for the synthetic image
var legend = ui_util.generateColorbarLegend(0, 0.5, ['blue', 'cyan', 'green', 'yellow', 'red'], 'vertical', 'GCC');
Map.add(legend);

// Add an info panel
Map.add(ui.Panel({
  widgets: [
    ui.Label('CCDC Visualization', {fontWeight: 'bold'}),
    ui.Label('• Chart shows actual observations and fitted model segments'),
    ui.Label('• Use the band selector to change the displayed band'),
    ui.Label('• Date slider creates synthetic images from the CCDC model')
  ],
  style: {
    position: 'bottom-left',
    padding: '8px'
  }
}));