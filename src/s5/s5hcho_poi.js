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

var s5hcho = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_HCHO') 
    .filterDate(dateStart, dateEnd)
    .filterBounds(roi)
    .select(
            ['tropospheric_HCHO_column_number_density', 
            'cloud_fraction',
            'HCHO_slant_column_number_density']
           );

// extract point data
var allObs = s5hcho.map(function(image) {
    var obs = image.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: poi,
        scale: 1000,
        maxPixels: 1e13
    });

    return image.set('system:time_start', image.get('system:time_start'))
                .set('tropospheric_HCHO_column_number_density', obs.get('tropospheric_HCHO_column_number_density'))
                .set('cloud_fraction', obs.get('cloud_fraction'))
                .set('HCHO_slant_column_number_density', obs.get('HCHO_slant_column_number_density'));
});

// Convert the collection with properties to a feature collection for export
var timeSeries = ee.FeatureCollection(allObs.map(function(image) {
    return ee.Feature(null, {
        'date': image.get('system:time_start'),
        'tropospheric_HCHO_column_number_density': image.get('tropospheric_HCHO_column_number_density'),
        'cloud_fraction': image.get('cloud_fraction'),
        'HCHO_slant_column_number_density': image.get('HCHO_slant_column_number_density')
    });
}));

// Export the feature collection to Drive
Export.table.toDrive({
    collection: timeSeries,
    description: 'E1_S5P_HCHO_Time_Series',
    fileFormat: 'CSV',
    folder: 'gee'
});

var chart = ui.Chart.image.seriesByRegion({
    imageCollection: s5hcho.select('tropospheric_HCHO_column_number_density'),
    regions: roi,
    reducer: ee.Reducer.mean(),
    scale: 1000,
    xProperty: 'system:time_start'
}).setOptions({
    title: 'S5P HCHO Time Series',
    vAxis: {title: 'HCHO Column Density (mol/m²)'},
    hAxis: {title: 'Date'},
    lineWidth: 1,
    pointSize: 3,
    series: {
        0: { type: 'line', color: 'blue' } // Absorbing Aerosol Index as line
    }
});
print(chart);