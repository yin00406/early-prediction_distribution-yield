// download 5-day images
var s2Tools = require("users/leikuny101/TOOLBOX:s2Tools")

var aoi = B5 //c
var aoi_str = "B5" //c
var year = 2023 //c
Map.centerObject(aoi)

var startDay = ee.Date.fromYMD(year, 5, 1)
var endDay = ee.Date.fromYMD(year, 10, 1)
var bands = ee.List(['red1','red2','red3','nir','red4','swir1', 'swir2', "REP", "RENDVI2"])

// Construct feature candicates from Sentinel-2 images
HS2L2A = HS2L2A
        .filterBounds(aoi)
        .filterDate(startDay,endDay)
        .map(HS2SR_CS)
        .map(s2Tools.sentinel2sr)
        .map(s2Tools.addVariables)
        .select(bands)

// generate composites
function get_orbit_num(imgCol){
    return imgCol.aggregate_array('SENSING_ORBIT_NUMBER').distinct()
}

var startDoy = startDay.getRelative('day','year')
var endDoy = 270
var step_size = 5
var starts = ee.List.sequence(startDoy, endDoy-1, step_size)
print(starts)
var composites = ee.ImageCollection(starts.map(function(start) {
  var doy = start
  var filtered = HS2L2A.filter(ee.Filter.dayOfYear(start, ee.Number(start).add(step_size)))
  var orbit = get_orbit_num(filtered)
  var mosaiced = ee.ImageCollection(orbit.map(function mosaic(i){
    var ds = filtered.filter(ee.Filter.eq('SENSING_ORBIT_NUMBER', i))
    return ds.mosaic()
})).mosaic()

  return mosaiced
}));
composites = composites.toList(composites.size())

for (var i =0; i<starts.size().getInfo(); i++){

//t output
Export.image.toDrive({
  image: composites.get(i), 
  description: aoi_str+"_" + year + "_" + starts.getNumber(i).getInfo(),
  folder: "GEE_Exported_Images_"+aoi_str,
  region: aoi, 
  scale: 30, 
  maxPixels: 1e13})
}

//cloud and shadow mask
function HS2SR_CS(image) {
  var cloudProb = image.select('MSK_CLDPRB');
  var snowProb = image.select('MSK_SNWPRB');
  var cloud = cloudProb.gte(5); //c
  var snow = snowProb.gte(5); //c
  var scl = image.select('SCL');
  var shadow = scl.eq(3); // 3 = cloud shadow
  var cirrus = scl.eq(10); // 10 = cirrus
  var cloudsBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var qa = image.select('QA60')
  var cloud1 = qa.bitwiseAnd(cloudsBitMask)
  var cirrus1 = qa.bitwiseAnd(cirrusBitMask)
  var mask = cloud.neq(1).and(snow.neq(1)).and(cirrus.neq(1)).and(shadow.neq(1))
    .and(cloud1.neq(1)).and(cirrus1.neq(1))
    .focal_min(3);
  return image.updateMask(mask);
}
