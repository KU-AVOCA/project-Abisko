/**
 * @file InsectDefoliationGEEapp.js
 * @description This script is a Google Earth Engine (GEE) application for visualizing and analyzing 
 * Green Chromatic Coordinate (GCC) time series data derived from the Harmonized Landsat and Sentinel-2 (HLS) dataset.
 * It includes synthetic GCC data generated using the Continuous Change Detection and Classification (CCDC) algorithm.
 * 
 * The application allows users to explore the time series data, visualize the results on a map, and download the data.
 * It also provides a user interface for selecting specific dates and viewing the corresponding GCC and defoliation intensity maps.
 * 
 * https://ku-avoca.projects.earthengine.app/view/defoliationdetector
 * Shunan Feng (shf@ign.ku.dk)
*/


/*
 * Map layer configuration
 */

// Define extent for visualization and masking
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

// Import color palette package
var palettes = require('users/gena/packages:palettes');
// Import functions for ccdc
var utils = require('users/parevalo_bu/gee-ccdc-tools:ccdcUtilities/api');
// Obtain water mask from RC Global Surface Water Mapping Layers
var waterMask = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    .select('max_extent').eq(0);
/*
 * Prepare data
 */

// Set the date range for data analysis
var date_start = ee.Date.fromYMD(2014, 1, 1),
    date_end = ee.Date(Date.now());

// Set color palette and visualization parameters for GCC
var GCCpallete = palettes.cmocean.Algae[7];
var GCCvis = {min: 0, max: 1, palette: GCCpallete};
// Set color palette and visualization parameters for defoliation intensity
var defoliationPalette = palettes.crameri.lajolla[50].reverse();
var defoliationVis = {min: -1, max: 0, palette: defoliationPalette};

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
  // var conditionScore = image.expression(
  //     '(GCC - GCC_predicted) / GCC_RMSE', {
  //         'GCC_predicted': image.select('GCC_predicted'),
  //         'GCC': image.select('GCC'),
  //         'GCC_RMSE': image.select('GCC_RMSE')
  //     }).rename('conditionScore');
  var relativeDeviation = image.expression(
      '(GCC - GCC_predicted) / GCC_predicted', {
          'GCC_predicted': image.select('GCC_predicted'),
          'GCC': image.select('GCC')
      }).rename('relativeDeviation');

  return image//.addBands(conditionScore).copyProperties(image, ['system:time_start'])
              .addBands(relativeDeviation).copyProperties(image, ['system:time_start']);
}
// Load the CCDC results
var ccdcAsset = ee.Image('projects/ku-avoca/assets/ccdcAbisko2014-2021'); // GCC, R, G, B, SWIR1, Tmask applied
var ccdcImage = utils.CCDC.buildCcdImage(ccdcAsset, 1, ['GCC']);

// Create the main map with hybrid basemap
var mapPanel = ui.Map();
mapPanel.setOptions('HYBRID').setControlVisibility(true);
var layers = mapPanel.layers();


/*
 * Panel setup
 */

// Create a panel to hold title, intro text, chart and legend components
var inspectorPanel = ui.Panel({style: {width: '30%'}});

// Create an intro panel with title and description
var intro = ui.Panel([
  ui.Label({
    value: 'GCC - Time Series Inspector',
    style: {fontSize: '20px', fontWeight: 'bold'}
  }),
  ui.Label('This web app displays time series of Green Chromatic Coordinate (GCC) index derived using the Harmonized Landsat and Sentinel-2 (HLS) dataset. Synthetic GCC data is produced using the Continuous Change Detection and Classification (CCDC) algorithm.')
]);
inspectorPanel.add(intro);

// Create panels to display longitude/latitude coordinates
var lon = ui.Label();
var lat = ui.Label();
inspectorPanel.add(ui.Panel([lon, lat], ui.Panel.Layout.flow('horizontal')));

// Add placeholders for the chart and legend
inspectorPanel.add(ui.Label('[Chart]'));
inspectorPanel.add(ui.Label('[Legend]'));


/*
 * Chart setup
 */

// Generate a time series chart of GCC for clicked coordinates
var generateChart = function (coords) {
  // Update the lon/lat panel with values from the click event
  lon.setValue('lon: ' + coords.lon.toFixed(4));
  lat.setValue('lat: ' + coords.lat.toFixed(4));

  // Add a dot for the point clicked on
  var point = ee.Geometry.Point(coords.lon, coords.lat);
  var dot = ui.Map.Layer(point, {color: '#ff0000'}, 'clicked location');
  // Add the dot as the third layer, so it shows up on top of all layers
  mapPanel.layers().set(0, dot);

  // Load and process data - using raw HLS data without daily averaging
  var hlsL = ee.ImageCollection('NASA/HLS/HLSL30/v002')
    .filterBounds(point)
    .filterDate(date_start, date_end)
    .map(renamHLSL30)
    .map(maskhls)
    .map(bandMath);

  var hlsS = ee.ImageCollection('NASA/HLS/HLSS30/v002')
    .filterBounds(point)
    .filterDate(date_start, date_end)
    .map(renamHLSS30)
    .map(maskhls)
    .map(bandMath);

  var hls = hlsL.merge(hlsS).select('GCC');
  
  // Generate daily synthetic GCC values
  var days = date_end.difference(date_start, 'day');
  var dateRange = ee.List.sequence(0, days.subtract(1));
  
  var dailyDates = dateRange.map(function(day) {
    return date_start.advance(day, 'day');
  });
  
  // Function to create synthetic images for each day
  var getSyntheticAtDates = function(date) {
    date = ee.Date(date);
    var inputDate = date.format('YYYY-MM-dd');
    var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1};
    var formattedDate = utils.Dates.convertDate(dateParams);
    
    return utils.CCDC.getSyntheticForYear(
      ccdcImage, formattedDate, 1, 'GCC', 'S1'
    ).set('system:time_start', date.millis())
     .set('system:index', date.format('yyyy-MM-dd'))
     .rename('GCC_predicted');
  };
  
  // Map the function over the dates
  var syntheticImages = ee.ImageCollection.fromImages(
    dailyDates.map(getSyntheticAtDates)
  );
  
  // Create time series chart for the selected point
  var geeChart = ui.Chart.image.series({
    imageCollection: syntheticImages.select('GCC_predicted'),
    region: point,
    reducer: ee.Reducer.mean(),
    scale: 30
  }).setOptions({
    title: 'GCC: Synthetic',
    vAxis: {title: 'GCC', viewWindow: {min: 0.25, max: 0.50}},
    hAxis: {title: 'Date'},
    series: {
      0: {
        color: 'blue',
        lineWidth: 1,
        pointsVisible: false,
        pointSize: 0
      }
    },
    legend: {position: 'right'}
  });
  
  // Create chart for actual HLS observations
  var hlsChart = ui.Chart.image.series({
    imageCollection: hls.select('GCC'),
    region: point,
    reducer: ee.Reducer.mean(),
    scale: 30
  }).setOptions({
    title: 'GCC: HLS observations',
    vAxis: {title: 'GCC', viewWindow: {min: 0.25, max: 0.50}},
    hAxis: {title: 'Date'},
    series: {
      0: {
        color: 'red',
        lineWidth: 0,
        pointsVisible: true,
        pointSize: 2
      }
    },
    legend: {position: 'right'}
  });
  
  // Add the charts at fixed positions
  inspectorPanel.widgets().set(2, ui.Panel([geeChart, hlsChart], ui.Panel.Layout.flow('vertical')));
};


/*
 * Legend setup
 */

// Create a color bar thumbnail image for the legend
function makeColorBarParams(palette) {
  return {
    bbox: [0, 0, 1, 0.1],
    dimensions: '100x10',
    format: 'png',
    min: 0,
    max: 1,
    palette: palette,
  };
}

// Create the color bar for the GCC legend
var colorBar = ui.Thumbnail({
  image: ee.Image.pixelLonLat().select(0),
  params: makeColorBarParams(GCCvis.palette),
  style: {stretch: 'horizontal', margin: '0px 8px', maxHeight: '24px'},
});
var colorBar_Defoliation = ui.Thumbnail({
  image: ee.Image.pixelLonLat().select(0),
  params: makeColorBarParams(defoliationVis.palette),
  style: {stretch: 'horizontal', margin: '0px 8px', maxHeight: '24px'},
});
// Create a panel with min, middle, and max values for the legend
var legendLabels = ui.Panel({
  widgets: [
    ui.Label(GCCvis.min, {margin: '4px 8px'}),
    ui.Label(
        ((GCCvis.max + GCCvis.min) / 2),
        {margin: '4px 8px', textAlign: 'center', stretch: 'horizontal'}),
    ui.Label(GCCvis.max, {margin: '4px 8px'})
  ],
  layout: ui.Panel.Layout.flow('horizontal')
});
var legendLabels_Defoliation = ui.Panel({
  widgets: [
    ui.Label(defoliationVis.min, {margin: '4px 8px'}),
    ui.Label(
        ((defoliationVis.max + defoliationVis.min) / 2),
        {margin: '4px 8px', textAlign: 'center', stretch: 'horizontal'}),
    ui.Label(defoliationVis.max, {margin: '4px 8px'})
  ],
  layout: ui.Panel.Layout.flow('horizontal')
});

var legendTitle = ui.Label({
  value: 'Map Legend: GCC',
  style: {fontWeight: 'bold'}
});
var legendTitle_Defoliation = ui.Label({
  value: 'Map Legend: Relative Deviation',
  style: {fontWeight: 'bold'}
});
var legendPanel = ui.Panel([legendTitle, colorBar, legendLabels, legendTitle_Defoliation, colorBar_Defoliation,legendLabels_Defoliation], ui.Panel.Layout.flow('vertical'), {stretch: 'horizontal', margin: '0px 8px'});
inspectorPanel.widgets().set(3, legendPanel);

/*
 * Map setup
 */

// Register a callback on the map to be invoked when the map is clicked
mapPanel.onClick(generateChart);

// Configure the map
mapPanel.style().set('cursor', 'crosshair');

// Initialize with a test point E1 at Abisko
var initialPoint = ee.Geometry.Point(19.0524377, 68.34836531); 
mapPanel.centerObject(initialPoint, 16);


/*
 * Initialize the app
 */

// Replace the root with a SplitPanel that contains the inspector and map
ui.root.clear();
ui.root.add(ui.SplitPanel(inspectorPanel, mapPanel));

// Generate initial chart for the test point
generateChart({
  lon: initialPoint.coordinates().get(0).getInfo(),
  lat: initialPoint.coordinates().get(1).getInfo()
});


/*
 * Date selector for GCC maps
 */

var dateIntro = ui.Panel([
  ui.Label({
    value: 'Map Viewer',
    style: {fontSize: '20px', fontWeight: 'bold'}
  }),
  ui.Label("Change date (YYYY-MM-DD) to load daily GCC and relative deviation map."),
  ui.Label("Note that the relative deviation is calculated as (GCC - GCC_predicted) / GCC_predicted and is only available when HLS clear observation is available."),
]);
inspectorPanel.widgets().set(4, dateIntro);

// Create dropdown panel for date selection
var dropdownPanel = ui.Panel({
  layout: ui.Panel.Layout.flow('horizontal'),
});
inspectorPanel.widgets().set(5, dropdownPanel);

// Create selectors for year, month, and day
var yearSelector = ui.Select({
  placeholder: 'please wait..',
});
var monthSelector = ui.Select({
  placeholder: 'please wait..',
});
var daySelector = ui.Select({
  placeholder: 'please wait..',
});
var button = ui.Button('Load');
dropdownPanel.add(yearSelector);
dropdownPanel.add(monthSelector);
dropdownPanel.add(daySelector);
dropdownPanel.add(button);
var urlLabel = ui.Label('Download', {shown: false});
dropdownPanel.add(urlLabel);

// Generate lists for dropdown options
var years = ee.List.sequence(date_end.get('year'), date_start.get('year'), -1),
    months = ee.List.sequence(1, 12),
    days = ee.List.sequence(1, 31);

// Format numbers as strings for dropdown items
var yearStrings = years.map(function(year){
  return ee.Number(year).format('%04d');
});
var monthStrings = months.map(function(month){
  return ee.Number(month).format('%02d');
});
var dayStrings = days.map(function(day){
  return ee.Number(day).format('%02d');
});

// Populate dropdown menus with options
yearStrings.evaluate(function(yearList) {
  yearSelector.items().reset(yearList);
  yearSelector.setPlaceholder('select a year');
});

monthStrings.evaluate(function(monthList) {
  monthSelector.items().reset(monthList);
  monthSelector.setPlaceholder('select a month');
});

dayStrings.evaluate(function(dayList) {
  daySelector.items().reset(dayList);
  daySelector.setPlaceholder('select a day');
});

// Function to load and display image composite for selected date
var loadComposite = function() {
    // var aoi = ee.Geometry.Rectangle(mapPanel.getBounds()); // comment this line as roi is already defined and it's small
    var year = yearSelector.getValue(),
        month = monthSelector.getValue(),
        day = daySelector.getValue();

    var startDate = ee.Date.fromYMD(
      ee.Number.parse(year), ee.Number.parse(month), ee.Number.parse(day));
    var endDate = startDate.advance(1, 'day');

    // Load and process data
    var hlsL = ee.ImageCollection('NASA/HLS/HLSL30/v002')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .map(renamHLSL30)
    .map(maskhls)
    .map(bandMath);

    var hlsS = ee.ImageCollection('NASA/HLS/HLSS30/v002')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .map(renamHLSS30)
    .map(maskhls)
    .map(bandMath);

    var hls = hlsL.merge(hlsS).select('GCC');
    
    // Convert to daily average
    var diff = endDate.difference(startDate, 'day');
    var dayNum = 1;
    var range = ee.List.sequence(0, diff.subtract(1), dayNum).map(function(day){
      return startDate.advance(day,'day');
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

    var syntheticDaily = ee.ImageCollection(ee.List(range.iterate(day_synthetic, ee.List([]))));

    var imgcollection = syntheticDaily.linkCollection(
      hlsDaily, 'GCC'
    ).select(['GCC_predicted', 'GCC']).map(scoreCalculation);

    // Create and display images
    var imgGCC_origional = imgcollection.select('GCC').mean().clip(roi).updateMask(waterMask);
    var imgGCC_predicted = imgcollection.select('GCC_predicted').mean().clip(roi).updateMask(waterMask);
    var imgDefoliation = imgcollection.select('relativeDeviation').mean().clip(roi).updateMask(waterMask);

    var layerName_origional = 'GCC_HLS_' + year + '-' + month + '-' + day;
    var layerName_predicted = 'GCC_predicted_' + year + '-' + month + '-' + day;
    var layerName_defoliation = 'Defoliation_' + year + '-' + month + '-' + day;

    var imgComposite_origional = imgGCC_origional.visualize(GCCvis);
    var imgComposite_predicted = imgGCC_predicted.visualize(GCCvis);
    var imgComposite_defoliation = imgDefoliation.visualize(defoliationVis);
    var imgCompositeLayer_predicted = ui.Map.Layer(imgComposite_predicted).setName(layerName_predicted);
    mapPanel.layers().set(1, imgCompositeLayer_predicted);
    var imgCompositeLayer_origional = ui.Map.Layer(imgComposite_origional).setName(layerName_origional);
    mapPanel.layers().set(2, imgCompositeLayer_origional);
    var imgCompositeLayer_defoliation = ui.Map.Layer(imgComposite_defoliation).setName(layerName_defoliation);
    mapPanel.layers().set(3, imgCompositeLayer_defoliation);
};
button.onClick(loadComposite);


/*
 * Add GEM logo and project information
 */
// var logo = ee.Image('projects/ku-gem/assets/GEM_Top-h100').visualize({
//   bands:  ['b1', 'b2', 'b3'],
//   min: 0,
//   max: 255
// });
// var thumb = ui.Thumbnail({
//   image: logo,
//   params: {
//       dimensions: '516x100',
//       format: 'png'
//   },
//   style: {height: '100/2px', width: '516/2px', padding: '0'}
// });
// var logoPanel = ui.Panel(thumb, 'flow', {width: '300px'});
// inspectorPanel.widgets().set(6, logoPanel);

// Add project information and links
var logoIntro = ui.Panel([
  ui.Label("Developed and maintained by Shunan Feng. This study is part of the Arctic biogenic Volatile Organic Compounds from Above (AVOCA) research project supported by the Villum Foundation (project No. 42069)"),
  ui.Label("https://github.com/KU-AVOCA/project-Abisko", {}, "https://github.com/KU-AVOCA/project-Abisko"),
  // ui.Label("https://github.com/KU-AVOCA/GEMLST", {}, "https://github.com/KU-AVOCA/GEMLST"),
]);
inspectorPanel.widgets().set(6, logoIntro);
