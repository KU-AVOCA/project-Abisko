// Plot E1 has highest BVOC emission in 2023
var poi = ee.Geometry.Point([19.0524377, 68.34836531]);
var roi = poi.buffer(30).bounds();
Map.addLayer(poi, {color: 'red'}, 'poi');
Map.addLayer(roi, {color: 'black'}, 'roi');
Map.centerObject(poi, 8);

var startDate = '2018-01-01';
var endDate = '2024-12-31';
var dateStart = ee.Date(startDate);
var dateEnd = ee.Date(endDate);

var s5col = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_AER_AI')
    .filterDate(dateStart, dateEnd)
    .filterBounds(roi)
    .select('absorbing_aerosol_index');

// extract point data
var allObs = s5col.map(function(image) {
    var obs = image.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: poi,
        scale: 1000,
        maxPixels: 1e13
    });

    return image.set('system:time_start', image.get('system:time_start'))
                .set('absorbing_aerosol_index', obs.get('absorbing_aerosol_index'));
});

// Convert the collection with properties to a feature collection for export
var timeSeries = ee.FeatureCollection(allObs.map(function(image) {
    return ee.Feature(null, {
        'date': image.get('system:time_start'),
        'absorbing_aerosol_index': image.get('absorbing_aerosol_index')
    });
}));

// Export the feature collection to Drive
Export.table.toDrive({
    collection: timeSeries,
    description: 'E1_S5P_Aerosol_Index_Time_Series',
    fileFormat: 'CSV',
    folder: 'gee'
});

var chart = ui.Chart.image.seriesByRegion({
    imageCollection: s5col,
    regions: roi,
    reducer: ee.Reducer.mean(),
    scale: 1000,
    xProperty: 'system:time_start'
}).setOptions({
    title: 'S5P Absorbing Aerosol Index Time Series',
    lineWidth: 1,
    pointSize: 3,
    series: {
        0: { type: 'line', color: 'blue' } // Absorbing Aerosol Index as line
    }
});
print(chart);