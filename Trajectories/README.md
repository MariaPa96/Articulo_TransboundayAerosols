
The code Trajectories_AOD has shown high versatility, for its use only four key steps are needed:

1. Clone the Trajectories_AOD.py code.

2. Download the wind components U-wind, V-wind and Omega 4X Daily, pressure levels provided by the NOAA PSL, Boulder, Colorado, USA, from their website at https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/pressure/

2. Download Total Aerosol Optical Depth at 550nm, step 0, Time  00:00 y 12:00 provided by CAMS-COpernicus from their website at
https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-atmospheric-composition-forecasts?tab=overview
Name the files as AOD_%m-%Y.nc (example: AOD_08-2019.nc)

3. Change the self.path in the .py for the file where you store the data dowloaded on step 2 and 3. 