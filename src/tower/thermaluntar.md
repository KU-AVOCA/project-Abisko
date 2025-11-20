# Untar all .tar files in the current folder and show progress, skipping already untarred files
Get-ChildItem -Filter *.tar | ForEach-Object {
    $outDir = $_.BaseName
    if (Test-Path $outDir) {
        Write-Host "Skipping $($_.Name), $outDir already exists."
    } else {
        New-Item -ItemType Directory -Path $outDir -Force | Out-Null
        tar -xf $_.FullName -C $outDir
        Write-Host "Untarred $($_.Name) to $outDir"
    }
}


'/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/6_Tower_Data/Tower Thermal images/1 Data/All/North-facing (SN_10600001)'

'/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/6_Tower_Data/Tower Thermal images/1 Data/All/West-facing (SN_10600002)'

'/home/geofsn/data/North-facing (SN_10600001)'

'/home/geofsn/data/West-facing (SN_10600002)'



'/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower thermal images/preview'  # Change to your desired output folder

'/home/geofsn/data/timelapsethermal'    
'/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower thermal images/preview/all'  # Change to your desired output folder
