[im_1stbreak, R_1stbreak] = readgeoraster("I:\SCIENCE-IGN-ALL\AVOCA_Group\2_Shared_folders\5_Projects\2025Abisko\CCDC\data\firstChangeTime.tif");
improj = projcrs(32634);
lonlim = [18.650628517410595,  19.180032203934033];
latlim = [68.28955676896352,  68.37246412370075];
[A,RA,attrib] = readBasemapImage("satellite",latlim,lonlim);

[xGrid,yGrid] = worldGrid(RA);
[latGrid,lonGrid] = projinv(RA.ProjectedCRS,xGrid,yGrid);
[xUTM34N,yUTM34N] = projfwd(improj,latGrid,lonGrid);

figure;
mapshow(xUTM34N, yUTM34N, A);
mapshow(xUTM34N, yUTM34N, A);
hold on
mapshow(im_1stbreak, R_1stbreak, 'DisplayType', 'surface');
colorbar
clim([2014 2020]);