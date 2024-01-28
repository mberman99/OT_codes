if __name__ == "__main__":
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xry
    import pandas as pd
    from matplotlib import gridspec
    import warnings # Silence the warnings from SHARPpy
    warnings.filterwarnings("ignore")
    import sharppy.plot.skew as skew
    from matplotlib.ticker import ScalarFormatter, MultipleLocator
    from matplotlib.collections import LineCollection
    import matplotlib.transforms as transforms
    import matplotlib.pyplot as plt
    from datetime import datetime
    from scipy import interpolate
    from scipy import integrate
    import scipy.stats
    import numpy.ma as ma
    import numpy as np
    from matplotlib import gridspec
    from sharppy.sharptab import winds, utils, params, thermo, interp, profile
    from sharppy.io.spc_decoder import SPCDecoder
    import sharppy.plot.skew as skew
    from datetime import datetime as dt
    import metpy.calc as mpcalc
    from metpy.cbook import get_test_data
    from metpy.plots import add_metpy_logo, SkewT
    from metpy.units import units

    from matplotlib.ticker import ScalarFormatter, MultipleLocator
    from matplotlib.collections import LineCollection
    import matplotlib.transforms as transforms

    from skimage.measure import label, regionprops

    import numpy as np

    import glob
    import sys
    import numpy as np
    import cartopy
    import scipy
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    from matplotlib.colors import ListedColormap
    from scipy.ndimage import gaussian_filter
    import pandas as pd
    import glob
    from IPython.core.debugger import set_trace
    from shapely.geometry import Polygon
    import geopandas as gpd
    import geopy.distance
    import shapely
    from cartopy.geodesic import Geodesic
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import pandas as pd
    from dateutil import parser

    from math import sin, cos, sqrt, atan2, radians

    import warnings
    warnings.filterwarnings("ignore")
    crs = {'init': 'epsg:4326', 'no_defs': True}
    gdf = gpd.GeoDataFrame(crs=crs)
    import cProfile
    from skimage.draw import ellipse
    from skimage.measure import label, regionprops, regionprops_table
    from skimage.transform import rotate
    from skimage import data, io, segmentation, color
    from skimage.future import graph

    mda = xry.open_dataset(r'/data/keeling/a/melinda3/NASA/MERRA2_data/MERRA2_400.inst3_3d_asm_Nv.20181102.nc4')
    #mda = xry.open_dataset('https://goldsmr5.gesdisc.eosdis.nasa.gov/dods/M2I3NVASM', engine='netcdf4')
    merra_trop = xry.open_dataset('/data/jtrapp/a/jamessg3/MERRA2/MERRA2_400.inst1_2d_asm_Nx.20181102.nc4', engine = 'netcdf4')


    import glob
    datestr='20181102'
    ihour='20'
    ihour2 = '19'
    ihour3 = '21'
    #files = sorted(glob.glob(f'/data/accp/a/snesbitt/arm/goesV2/*{datestr}.{ihour}????.cdf'))
    files = sorted(glob.glob(f'/data/accp/a/snesbitt/arm/goesV2/*{datestr}.*.cdf'))
    #files2 = sorted(glob.glob(f'/data/accp/a/snesbitt/arm/goesV2/*{datestr}.{ihour2}????.cdf'))
    #files3 = sorted(glob.glob(f'/data/accp/a/snesbitt/arm/goesV2/*{datestr}.{ihour3}????.cdf'))

    #files_all = files2 + files3 +files
    
    #Create subdomain for the data to interpolate across

    lat_down = -45
    lat_up = -21
    lon_left = -75
    lon_right = -53
    weather_vars = ['T', 'RH', 'U', 'V', 'PL', 'H']
    trop_vars = ['TROPPT', 'TROPPV']
    merra_selected_variables = mda[weather_vars]
    trop_selected_variables = merra_trop[trop_vars]
    tt = merra_selected_variables.sel(lat=slice(lat_down, lat_up), lon = slice(lon_left, lon_right))
    sliced_trop_data = trop_selected_variables.sel(lat = slice(lat_down, lat_up), lon = slice(lon_left, lon_right))
    levs = np.arange(1,73,1)
    m_slice = tt.assign_coords(lev = levs)

    def process_ot_data(file, thresh=.8):

        llcrnr = [-35, -66]
        urcrnr = [-30, -60]
        
        # Set the coordinate reference system
        crs = {'init': 'epsg:4326', 'no_defs': True}
        ot_data = gpd.GeoDataFrame(crs=crs)
        
        ds_og = xry.open_dataset(file).squeeze()
        
        
        time = ds_og.time.values
        
        inx = np.where((ds_og['longitude'] >= llcrnr[1]) & (ds_og['longitude'] <= urcrnr[1]))[0]
        iny = np.where((ds_og['latitude'] >= llcrnr[0]) & (ds_og['latitude'] <= urcrnr[0]))[0]

        ds = ds_og.sel(npixels=inx, nlines=iny)
        
        lat = ds['latitude'].values
        lon = ds['longitude'].values
        
        #time = bedka['time'].isel(time=t).values    
        lon_2d, lat_2d = np.meshgrid(lon,lat)
        
        lon_2d_corrected = lon_2d + ds['parallax_correction_longitude'].fillna(0).values
        lat_2d_corrected = lat_2d + ds['parallax_correction_latitude'].fillna(0).values
        
        #inx = np.where((ds['longitude'] >= llcrnr[1]) & (ds['longitude'] <= urcrnr[1]))[0]
        #iny = np.where((ds['latitude'] >= llcrnr[0]) & (ds['latitude'] <= urcrnr[0]))[0]

        time_label = pd.to_datetime(ds.time.values).strftime('%Y-%m-%d %H:%M')
        
        ds_whole = ds

        db = ds.ot_probability
        da = ds.ir_brightness_temperature
        dc = ds.ot_id_number
        de = ds.ot_anvilmean_brightness_temperature_difference
        cloud_hgt = ds.cloud_top_height
        trop_hgt = ds.tropopause_height
        trop_tmp = ds.tropopause_temperature
        trop_pres = ds.tropopause_pressure

        smoothed_tb = gaussian_filter(da.values, 1)

        plot_num=0

        file_filt = ds.where(ds.ot_probability > 0.8)
        ot_id = file_filt.ot_id_number[:,:]
        mask = ot_id > 0
        label_ot = label(mask, connectivity=2)
        regions = regionprops(label_ot)
        props = regionprops_table(label_ot, properties=('centroid', 'orientation'))

        ot_df = pd.DataFrame(props)

        
        y_ot = ot_df[['centroid-0'][0]]
        x_ot =  ot_df[['centroid-1'][0]] 

        y_ot = np.round(y_ot, decimals=0)
        x_ot = np.round(x_ot, decimals=0)   
    
        for y, x in zip(y_ot, x_ot):
            y_in = np.int(y)
            x_in = np.int(x)

            otid = dc.sel(nlines=y_in, npixels=x_in).values

            minpt = np.where(dc == otid)
            #print(minpt)
            
        
            dbs = db.values[minpt]
            ddlat = lat_2d[minpt]
            ddlon = lon_2d[minpt]
            ddlat_corr = lat_2d_corrected[minpt]
            ddlon_corr = lon_2d_corrected[minpt]
            das = da.values[minpt]
            dde = de.values[minpt]
            
            try:
                mintb = np.nanmin(das)
                
            except:
                continue
                
            dbs = db.values[minpt]
            minloc = np.argmin(das)
            prob = dbs[minloc]
            ddc = dc.values[minpt]
            otid = ddc[minloc]
            lat = ddlat[minloc]
            lon = ddlon[minloc]
            lon_corrected = ddlon_corr[minloc]
            lat_corrected = ddlat_corr[minloc]
            thgt = trop_hgt.values[minpt][minloc]
            ttmp = trop_tmp.values[minpt][minloc]
            tprs = trop_pres.values[minpt][minloc]
            clhgt = cloud_hgt.values[minpt][minloc]


            if prob > thresh:

                def find_nearest(array, value):
                    array = np.asarray(array)
                    idx = (np.abs(array - value)).argmin()
                    return array[idx]
        
                try:
                #print(minpt)
                    y0, x0 = np.where(ds.ir_brightness_temperature.values == ds.ir_brightness_temperature.values[minpt][minloc])
                
                    yt = find_nearest(y0, y_in)
                    xt = find_nearest(x0, x_in)


                    y0 = yt
                    x0 = xt
                    lat, lon = ds.sel(nlines=y0, npixels=x0).latitude.values, ds.sel(nlines=y0, npixels=x0).longitude.values



                except Exception as e: 
                    print(e)
            
                num = 20
                hyp = 20 * np.cos(45, dtype='d')
                x1_all = [x0      , x0 + hyp, x0 + num, x0 + hyp, x0      , x0 - hyp, x0 - num, x0 - hyp]
                y1_all = [y0 + num, y0 + hyp, y0      , y0 - hyp, y0 - num, y0 - hyp, y0      , y0 + hyp]
            
                pts=np.zeros([8,41])
                lats = np.zeros([8,41])
                lons = np.zeros([8,41])
                x_rads = np.zeros([8,41])
                y_rads = np.zeros([8,41])
                                
                j=0
                
                for x1, y1 in zip(x1_all,y1_all):
                    x, y = np.linspace(x0, x1, (num * 2) + 1), np.linspace(y0, y1, (num * 2) + 1)
                    pts[j,:] = scipy.ndimage.map_coordinates(smoothed_tb, np.vstack((y,x)))
                    lats[j,:] = scipy.ndimage.map_coordinates(lat_2d, np.vstack((y,x)))
                    lons[j,:] = scipy.ndimage.map_coordinates(lon_2d, np.vstack((y,x)))
                    j = j + 1
                
                dels = np.gradient(np.gradient(pts.T,axis=0),axis=0)
                
                tb_rads = pts
                
                # Find inflection points
                dels_zero = np.argmax(dels <= 0, axis=0)
                
                # Calculate mean and standard deviation of radials
                mean = np.mean(dels_zero)
                std = np.std(dels_zero)
                
                # Replace radials with mean radial if greater than 1 standard deviation from mean
                dels_zero = np.where(dels_zero > (mean + std), int(np.round(mean)), dels_zero)
                
                
                ind = np.arange(0,8)
                
                lat_mins = lats[ind, dels_zero]
                lon_mins = lons[ind, dels_zero]
                
                # Calcualte the distance for each radial
                
                dists = []
                for i in range(len(dels_zero)):
                    dists.append(geopy.distance.distance((lat, lon), (lat_mins[i], lon_mins[i])).kilometers)
                dists = np.array(dists)
                                
                # setup a polygon
                crs = {'init': 'epsg:4326', 'no_defs': True}
                polygon_geom = Polygon(zip(lon_mins, lat_mins))
            
                gdf = gpd.GeoDataFrame(crs="EPSG:4326")
                gdf = gdf.append({'otid':i,
                                'geometry':polygon_geom}, ignore_index=True)
                gdf['otarea_from_polygon'] = gdf['geometry'].set_crs(epsg=4326).to_crs("EPSG:32720")\
                                            .map(lambda p: p.area / 10**6)
                
                
                otarea_from_polygon = gdf.otarea_from_polygon.values[0]
                otarea_circle = np.pi * (mean **2)
                
                #print(list(tuple(dels[:, 0])))
                try:
                    
                    ot_data = ot_data.append({'time':time,
                                            'lat':lat,
                                            'lon':lon,
                                            'lat_corr':lat_corrected,
                                            'lon_corr':lon_corrected,
                                            'otid':otid,
                                            'mintb':mintb,
                                            'tropopause_height':thgt,
                                            'tropopause_temperature':ttmp,
                                            'tropopause_pressure':tprs,
                                            'cloudtop_height':clhgt,
                                            'prob':prob,
                                            'area_polygon':otarea_from_polygon,
                                            'otarea_circle':otarea_circle,
                                            's_radial':dists[0],
                                            's_radial_del2':dels[:, 0],
                                            's_tb':tb_rads[0],
                                            'se_radial':dists[1],
                                            'se_radial_del2':dels[:, 1],
                                            'se_tb':tb_rads[1],
                                            'e_radial':dists[2],
                                            'e_radial_del2':dels[:, 2],
                                            'e_tb':tb_rads[2],
                                            'ne_radial':dists[3],
                                            'ne_radial_del2':dels[:, 3],
                                            'ne_tb':tb_rads[3],
                                            'n_radial':dists[4],
                                            'n_radial_del2':dels[:, 4],
                                            'n_tb':tb_rads[4],
                                            'nw_radial':dists[5],
                                            'nw_radial_del2':dels[:, 5],
                                            'nw_tb':tb_rads[5],
                                            'w_radial':dists[6],
                                            'w_radial_del2':dels[:, 6],
                                            'w_tb':tb_rads[6],
                                            'sw_radial':dists[7],
                                            'sw_radial_del2':dels[:, 7],
                                            'sw_tb':tb_rads[7],
                                            'geometry':polygon_geom}, ignore_index=True)
                    
                    
                
                except Exception as e:
                    print(e)
                try:
                    ot_data = ot_data[ot_data.area_polygon < 1000]
                    
            #filename = pd.to_datetime(time).strftime('ot_output/%Y%m%d/%Y%m%d_%H%M.csv')
                
                    #ot_data.to_csv(filename)
            
                except:
                    None
        
        ds_og.close()
        ds.close()
        return ot_data
        #print(ot_data)
    
        
    #Get the closest files for the MERRA data and run a linear interpolation over them

    def linear_interpolation_function(ot_time, sliced_merra_data,
            ot_lon, ot_lat, ot_height, merra_trop):

        """ This function will create linear interpolations of all relevant data from MERRA-2 data
        and will get vertical profiles of all the data fields to prepare the data for profile creation and 
        thermodynamic calculations for analysis between OT charactertistics and LMS thermodynamic profiles.

        This function also plots soundings and calculates the thermodynamic quantities that quantify static 
        stability in the lowermost stratosphere. 

        """
        #Extract the tropopause information for the spatio-temporally closest grid cell
        lontp = merra_trop.lon
        lattp = merra_trop.lat
        sel_trop = merra_trop.sel(time= ot_time, method = 'nearest')
        trop_pt_rbs = interpolate.RectBivariateSpline(lattp, lontp, sel_trop.TROPPT[:,:])
        trop_pt = trop_pt_rbs.ev(ot_lat, ot_lon)
        trop_pv_rbs = interpolate.RectBivariateSpline(lattp, lontp, sel_trop.TROPPV[:,:])
        trop_pv = trop_pv_rbs.ev(ot_lat, ot_lon)

        #trop_pt = merra_trop['TROPPT'].sel(time = ot_time, lat = ot_lat,\
            #                               lon = ot_lon, method = 'nearest').values
        #trop_pv = merra_trop['TROPPV'].sel(time = ot_time, lat = ot_lat,\
            #                               lon = ot_lon, method = 'nearest').values
        trop = max(trop_pt, trop_pv)
        trop = trop/100
        sliced_merra_data = sliced_merra_data.where(sliced_merra_data.time <= ot_time, drop = True)
        ctime_merra = sliced_merra_data.sel(time = ot_time, method = 'nearest')
        #print(ctime_merra.time)   
        lons = ctime_merra.lon
        lats = ctime_merra.lat

        del sliced_merra_data, merra_trop
       
        
        #Convert the OT height in km to a pressure level
        #ot_height_m = ot_height * 1000 
        #ot_pres = mpcalc.height_to_pressure_std(ot_height_m * units.meter)
        #ot_pres = ot_pres.magnitude

        ot_pres = trop - 50


        if ot_pres < trop:
            def find_near(ar, vals):
                array = np.asarray(ar)
                idx = (np.abs(ar - vals)).argmin()
                return idx

            #print(ot_lon, ot_lat)
            #Create the linear interpolation

            t_interp = []
            h_interp = []
            rh_interp = []
            u_interp = []
            v_interp = []
            pl_interp = []



            levs = np.arange(0,len(ctime_merra.lev))
            for lev in levs:
                rbs_temp = interpolate.RectBivariateSpline(lats, lons, ctime_merra.T[lev, :, :])
                t_interp.append(rbs_temp.ev(ot_lat, ot_lon))
                rbs_gh = interpolate.RectBivariateSpline(lats, lons, ctime_merra.H[lev, :, :])
                h_interp.append(rbs_gh.ev(ot_lat, ot_lon))
                rbs_rh = interpolate.RectBivariateSpline(lats, lons, ctime_merra.RH[lev, :, :])
                rh_interp.append(rbs_rh.ev(ot_lat, ot_lon))
                rbs_uwind = interpolate.RectBivariateSpline(lats, lons, ctime_merra.U[lev, :, :])
                u_interp.append(rbs_uwind.ev(ot_lat, ot_lon))
                rbs_vwind = interpolate.RectBivariateSpline(lats, lons, ctime_merra.V[lev, :, :])
                v_interp.append(rbs_vwind.ev(ot_lat, ot_lon))
                rbs_pres = interpolate.RectBivariateSpline(lats, lons, ctime_merra.PL[lev, :, :])
                pl_interp.append(rbs_pres.ev(ot_lat, ot_lon))
            
            t_interp = np.asarray(np.stack(t_interp, axis = 0)).flatten()
            h_interp = np.asarray(np.stack(h_interp, axis = 0)).flatten()
            rh_interp = np.asarray(np.stack(rh_interp, axis = 0)).flatten()
            u_interp = np.asarray(np.stack(u_interp, axis = 0)).flatten()
            v_interp = np.asarray(np.stack(v_interp, axis = 0)).flatten()
            pl_interp = np.asarray(np.stack(pl_interp, axis = 0)).flatten()
            


            ot_trop_thermal = trop
            #Regrid the data to a 1hPa interval between levels
            ot_pres_idx = find_near(pl_interp, ot_pres*100)
            el_pres_idx = find_near(pl_interp, ot_trop_thermal*100)

            if len(t_interp[ot_pres_idx:el_pres_idx]) > 2:
                t_af = t_interp[:ot_pres_idx]
                t_be = t_interp[el_pres_idx:]
                rh_be = rh_interp[el_pres_idx:]
                rh_af = rh_interp[:ot_pres_idx]
                u_af = u_interp[:ot_pres_idx]
                u_be = u_interp[el_pres_idx:]
                v_af = v_interp[:ot_pres_idx]
                v_be = v_interp[el_pres_idx:]
                h_af = h_interp[:ot_pres_idx]
                h_be = h_interp[el_pres_idx:]
                p_be = pl_interp[el_pres_idx:]
                p_af = pl_interp[:ot_pres_idx]

                temp2 = []
                rh2 = []
                u_wind2 = []
                v_wind2 =[]
                hgts2 =[]
                pres_vals2 = []

                press = pl_interp[ot_pres_idx:el_pres_idx]
                nspace = (max(press) - min(press))/500
                new_pr_coords = np.arange(min(press), max(press), 500)
                temp_interp = interpolate.interp1d(press, t_interp[ot_pres_idx:el_pres_idx].flatten())
                t_int = temp_interp(new_pr_coords)
                rh_interp2 = interpolate.interp1d(press, rh_interp[ot_pres_idx:el_pres_idx].flatten())
                rh_int = rh_interp2(new_pr_coords)
                u_interp2 = interpolate.interp1d(press, u_interp[ot_pres_idx:el_pres_idx].flatten())
                u_int = u_interp2(new_pr_coords)
                v_interp2 = interpolate.interp1d(press, v_interp[ot_pres_idx:el_pres_idx].flatten())
                v_int = v_interp2(new_pr_coords)
                hg_interp = interpolate.interp1d(press, h_interp[ot_pres_idx:el_pres_idx].flatten())
                h_int = hg_interp(new_pr_coords)
            
                temp2 = np.concatenate((t_af, t_int, t_be))
                rh2 = np.concatenate((rh_af, rh_int, rh_be))
                u_wind2 = np.concatenate((u_af, u_int, u_be))
                v_wind2 = np.concatenate((v_af, v_int, v_be))
                hgts2 = np.concatenate((h_af, h_int, h_be))
                pres_vals2 = np.concatenate((p_af, new_pr_coords, p_be))
                print("Interp")
                
            else:
                t_af = t_interp[:ot_pres_idx-2]
                t_be = t_interp[el_pres_idx:]
                rh_be = rh_interp[el_pres_idx:]
                rh_af = rh_interp[:ot_pres_idx-2]
                u_af = u_interp[:ot_pres_idx-2]
                u_be = u_interp[el_pres_idx:]
                v_af = v_interp[:ot_pres_idx-2]
                v_be = v_interp[el_pres_idx:]
                h_af = h_interp[:ot_pres_idx-2]
                h_be = h_interp[el_pres_idx:]
                p_be = pl_interp[el_pres_idx:]
                p_af = pl_interp[:ot_pres_idx-2]

                temp2 = []
                rh2 = []
                u_wind2 = []
                v_wind2 =[]
                hgts2 =[]
                pres_vals2 = []
                press = pl_interp[ot_pres_idx-2:el_pres_idx]
                nspace = (max(press) - min(press))/500
                new_pr_coords = np.arange(min(press), max(press), 500)
                temp_interp = interpolate.interp1d(press, t_interp[ot_pres_idx-2:el_pres_idx].flatten())
                t_int = temp_interp(new_pr_coords)
                rh_interp2 = interpolate.interp1d(press, rh_interp[ot_pres_idx-2:el_pres_idx].flatten())
                rh_int = rh_interp2(new_pr_coords)
                u_interp2 = interpolate.interp1d(press, u_interp[ot_pres_idx-2:el_pres_idx].flatten())
                u_int = u_interp2(new_pr_coords)
                v_interp2 = interpolate.interp1d(press, v_interp[ot_pres_idx-2:el_pres_idx].flatten())
                v_int = v_interp2(new_pr_coords)
                hg_interp = interpolate.interp1d(press, h_interp[ot_pres_idx-2:el_pres_idx].flatten())
                h_int = hg_interp(new_pr_coords)
            
                temp2 = np.concatenate((t_af, t_int, t_be))
                rh2 = np.concatenate((rh_af, rh_int, rh_be))
                u_wind2 = np.concatenate((u_af, u_int, u_be))
                v_wind2 = np.concatenate((v_af, v_int, v_be))
                hgts2 = np.concatenate((h_af, h_int, h_be))
                pres_vals2 = np.concatenate((p_af, new_pr_coords, p_be))
                print("Extended Interp")
            

            t2 = []
            rh = []
            uw = []
            vw = []
            ht = []
            pv = []
            [t2.append(x) for x in temp2 if x not in t2]
            [rh.append(x) for x in rh2 if x not in rh]
            [uw.append(x) for x in u_wind2 if x not in uw]
            [vw.append(x) for x in v_wind2 if x not in vw]
            [ht.append(x) for x in hgts2 if x not in ht]
            [pv.append(x) for x in pres_vals2 if x not in pv]
            #Grid Cell Test Section
            ot_temp2 = np.flip(t2)[:].flatten()   
            ot_rh2 = np.flip(np.array(rh))[:].flatten()
            ot_uwind2 = np.flip(np.array(uw))[:].flatten()
            ot_vwind2 = np.flip(np.array(vw))[:].flatten()
            ot_hgt2 = np.flip(np.array(ht))[:].flatten()
            ot_pres_vals2 = np.flip(np.array(pv)[:]).flatten()/100

            ot_temps_C2 = ot_temp2 - 273.15
            ot_temps_C2 = ot_temps_C2

            td2 = mpcalc.dewpoint_from_relative_humidity(ot_temps_C2*units.degC, ot_rh2)
            td2 = np.array(td2)
            

            #ot_temps_C2 = ot_temps_C2[temp_nans2]
            u_filt_ot2 = ot_uwind2*units.meter/units.second
            v_filt_ot2 = ot_vwind2*units.meter/units.second
            ot_hgt_filt2 = ot_hgt2
            ot_len = np.array(len(ot_hgt2))
            dz = [ot_hgt_filt2[ind+1] - ot_hgt_filt2[ind] for ind in range(ot_len-1)]


            #Calculate wind speed and direction
            wind_direction= mpcalc.wind_direction(u_filt_ot2, v_filt_ot2, convention = 'from')
            wind_speed = mpcalc.wind_speed(u_filt_ot2, v_filt_ot2)

            
            prof = profile.create_profile(profile='default', pres=ot_pres_vals2, hght=ot_hgt2, tmpc=ot_temps_C2, \
                                            dwpc=td2, wspd = wind_speed, wdir = wind_direction, strictQC=False)
            mupcl = params.parcelx( prof, flag=3 ) # Most-Unstable Parcel
            mucape = mupcl.bplus

            
            #Calculate lapse rate and buoyancy for parcels into the stratosphere 
            
            #Calculate theta
            #Choose the temperature closest to the OT height: https://stackoverflow.com/questions/9706041/finding-index-of-an-item-closest-to-the-value-in-a-list-thats-not-entirely-sort
            trop_top = ot_trop_thermal - 50 
            ot_pres_idx = find_near(ot_pres_vals2, ot_pres)
            trop_pres_idx = find_near(ot_pres_vals2, ot_trop_thermal)
            trop50_idx = find_near(ot_pres_vals2, trop_top)
            pres_int = ot_pres_vals2[trop_pres_idx:trop50_idx]
            theta = mpcalc.potential_temperature(ot_pres_vals2*units.millibar, ot_temps_C2*units.degC)
            trop_lowest = ot_trop_thermal
            trop_pt_temp_idx = find_near(prof.pres, trop_pt/100)
            trop_pt_temp = prof.tmpc[trop_pt_temp_idx]
            trop_pv_temp_idx = find_near(prof.pres, trop_pv/100)
            trop_pv_temp = prof.tmpc[trop_pv_temp_idx]
            trop_temp_lowest_idx = find_near(prof.pres, trop_lowest)
            trop_temp_lowest = prof.tmpc[trop_temp_lowest_idx]
            mix_rat = mpcalc.mixing_ratio_from_relative_humidity(ot_pres_vals2*units.millibar, ot_temps_C2*units.degC,\
                            ot_rh2)
            depth = ot_pres_vals2[0]*units.millibar - ot_trop_thermal*units.millibar
            height_trop_hypso = mpcalc.thickness_hydrostatic(ot_pres_vals2*units.millibar, ot_temps_C2*units.degC,\
                mixing_ratio = mix_rat, depth = depth, bottom = ot_pres_vals2[0]*units.millibar)
            if type(height_trop_hypso) != 'int':
                height_trop_hypso = height_trop_hypso.magnitude
            else:
                height_trop_hypso = height_trop_hypso
            if len(pres_int) > 1:
            #Tropopause Height
                trop_height = np.array(mpcalc.pressure_to_height_std(ot_trop_thermal* units.mbar))*units.km
                trop_height = trop_height.magnitude
            #Lapse Rate
                lapses = []
                lapse_rate = None
                for a in pres_int:
                    lapses.append(params.lapse_rate(prof, a, a+1))
                    lapse_rate = np.mean(lapses)
                del lapses
            
            #Buoyancy (BV Frequency Squared)
                static_stab = mpcalc.brunt_vaisala_frequency_squared(ot_hgt_filt2*units.meter, theta)
                bv_lms = static_stab[trop_pres_idx:ot_pres_idx]
                bv_lms = np.mean(bv_lms.magnitude)
                
                dtdz = mpcalc.first_derivative(theta, delta = dz)
            
                n2_hand = 9.80665/theta * dtdz  
                n2_handt = np.mean(n2_hand[trop_pres_idx:trop50_idx]).magnitude
                del trop_pres_idx, ot_pres_idx

                if lapse_rate is None:
                    lapse_rate == -9999
                else:
                    lapse_rate = lapse_rate
                return (mucape, lapse_rate, n2_handt, prof, mupcl, ot_pres, trop_height, ot_trop_thermal, trop_pt, trop_pv, height_trop_hypso, trop_temp_lowest, trop_pt_temp, trop_pv_temp) 


            else:
                mucape, lapse_rate, n2_handt, prof, mupcl, ot_pres, trop_height, ot_trop_thermal, trop_pt, trop_pv, height_trop_hypso, trop_temp_lowest, trop_pt_temp, trop_pv_temp =\
                -9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999, -9999, -9999, -9999, -9999, -9999, -9999
                return (mucape, lapse_rate, n2_handt, prof, mupcl, ot_pres, trop_height, ot_trop_thermal, trop_pt, trop_pv, height_trop_hypso, trop_temp_lowest, trop_pt_temp, trop_pv_temp)

        else:
             mucape, lapse_rate, n2_handt, prof, mupcl, ot_pres, trop_height, ot_trop_thermal, trop_pt, trop_pv, height_trop_hypso, trop_temp_lowest, trop_pt_temp, trop_pv_temp  =\
            -9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999, -9999, -9999, -9999, -9999, -9999, -9999
        return (mucape, lapse_rate, n2_handt, prof, mupcl, ot_pres, trop_height, ot_trop_thermal, trop_pt, trop_pv, height_trop_hypso, trop_temp_lowest, trop_pt_temp, trop_pv_temp)
   

    files[1]
    df_list = []
    jan25_mucape = []
    jan25_lapserate = []
    jan25_n2calc = []
    jan25_prof = []
    area_poly = []
    ct_heights = []
    trop_height = []
    df_time_out= []
    df_lat_out = []
    df_lon_out = []
    w_max = []
    mucalc = []
    ot_press = []
    therm_trop =[]
    exp_top = []
    cin_lev = []
    trop_pt = []
    trop_pv = []
    height_hypso = []
    trop_temp_pv = []
    trop_temp_pt = []
    trop_temp_lowest = []
    gpd_out = gpd.GeoDataFrame()

    from dask.distributed import Client
    client = Client()

    #Nov 10 Plots

    for file in files:
        ds = xry.open_dataset(file).squeeze()  
        df = process_ot_data(file)

        ir_vals = ds.ir_brightness_temperature.values  
        
        try:
            df_ex = df[df.mintb == np.nanmin(df.mintb.values)]
            
        except:
            continue
            
        
        lons = ds.longitude.values
        lats = ds.latitude.values   
        area_poly.extend(df.area_polygon.values)
        ct_heights.extend(df.cloudtop_height.values)  
        ct_height = df.cloudtop_height.values

        print(df.area_polygon)

        #merra_lon = m_slice['lon'].sel(lon = df_ex.lon, method = 'nearest')
        #merra_lat = m_slice['lat'].sel(lat = df_ex.lat, method = 'nearest')
        for time, lat, lon, ot_height in zip(df.time.values, df.lat.values, df.lon.values, ct_heights):
            mucape, lapse_rate, n2_calc, prof, mupcl, ot_pres, trop_heights, thermal_trop, troppt, troppv, hth, ttl, tpt, tpv  = linear_interpolation_function(time,  m_slice, lon, lat, ot_height, sliced_trop_data)
            jan25_mucape.append(mucape)
            jan25_lapserate.append(lapse_rate)
            jan25_n2calc.append(n2_calc)
            trop_height.append(trop_heights)
            therm_trop.append(thermal_trop)
            ot_press.append(ot_pres)
            w = np.sqrt(2 * mucape)
            w_max.append(w)
            trop_pt.append(troppt)
            trop_pv.append(troppv)
            height_hypso.append(hth)
            trop_temp_pt.append(tpt)
            trop_temp_pv.append(tpv)
            trop_temp_lowest.append(ttl)
        

            #exp_top.append(exp_t)
        print(len(ct_heights), len(area_poly), len(trop_height))
        print(df.time)

        
        """ax = plt.subplot(gs[0:3, 0:3], projection=ccrs.PlateCarree())
        ax.set_extent((lon-0.5, lon+0.5, lat-0.5, lat+0.5))
        cf = plt.contourf(lons, lats, ir_vals, np.arange(180, 270, 2), cmap='gist_rainbow_r')
        plt.colorbar(cf, label='Brightness Temperature (K)')




        merra_lon = m_slice['lon'].sel(lon = df_ex.lon, method = 'nearest')
        merra_lat = m_slice['lat'].sel(lat = df_ex.lat, method = 'nearest')




        df_ex.plot(ax=ax, facecolor="none", edgecolor='black', lw=3)


        ax.plot(0, 0, color='black', lw=3, label=f'OT Area {np.round(df_ex.area_polygon.values[0], 2)} km2 \n Prob: {np.round(df_ex.prob.values[0], 2)}')
        ax.plot(df_ex.lon, df_ex.lat, 'o', color = 'black', markersize = 12, label = 'OT Center')
        ax.plot(merra_lon, merra_lat, '*', color = 'red', markersize = 12, label = 'Closest Sounding')
        plt.legend(fontsize=16)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                            linewidth=2, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        plt.title(pd.to_datetime(df.time.values[0]).strftime('%d %b %y %H:%M:%S'), fontsize=24)



        #Converted model levels to pressure levels Sounding

        ax2 = plt.subplot(gs[0:3, 3:], projection='skewx')
        #title = ('Most Unstable Parcel Sounding, 1-25-2019, 18Z')
        #skew.draw_title(ax2, title)
        ax2.grid(True)
        plt.grid(True)


        # Let's set the y-axis bounds of the plot and create the profile to plot
        #Pressure level data sounding
        pmax = 1000
        pmin = 10
        dp = -10
        presvals = np.arange(int(pmax), int(pmin)+dp, dp)

        pcl = mupcl
        muel = pcl.elpres



        for t in np.arange(-10,45,5):
            tw = []
            for p in presvals:
                tw.append(thermo.wetlift(1000, t, p))
                # Plot the moist-adiabat with a black line that is faded a bit.
            ax2.semilogy(tw, presvals, 'k-', lw = 0.5, alpha=.2)

        ax2.semilogy(prof.tmpc, prof.pres, 'r', lw=2)
        ax2.semilogy(prof.dwpc, prof.pres, 'g', lw=2) 
        ax2.semilogy(pcl.ttrace, pcl.ptrace, 'k-.', lw = 2,\
            label = 'Lapse Rate = %0.2f'%lapse_rate + ' C/km\n' + 'N$^{2}$ = %0.2E\n'%n2_calc\
                    + 'CAPE = %0.2f'%mucape + ' J/kg')
        #ax2.semilogy(pcl.ttrace, pcl.ptrace, 'k-.', lw=2,\
        #label = 'MUPCL')
        #ax2.semilogy(pcl2.ttrace, pcl2.ptrace, 'b-.', lw = 2, label = 'SFCPCL')
        #ax2.semilogy(pcl3.ttrace, pcl3.ptrace, 'p-.', lw=2, label = 'FCSTPCL ')


        #ax1.axhline(trop_pres, -40, 40, label = 'Tropopause Pressure', color = 'black')
        t50 = thermal_trop-50 
        #ax1.axhline(trop_pres, -40, 40, label = 'Tropopause Pressure', color = 'black')
        ax2.axhline(t50, -40, 40,  label = 'Interpolation Upper Bound', color = 'darkblue')
        ax2.axhline(thermal_trop, -40, 40, label = 'MERRA-2 Tropopause', color = 'blue')
        ax2.yaxis.set_major_formatter(plt.ScalarFormatter())
        ax2.xaxis.set_major_locator(MultipleLocator(10))
        ax2.set_xlim(-50,50)
        #ax3 = plt.subplot(gs[2:,3])
        #skew.plot_wind_axes(ax3)
        #skew.plot_wind_barbs(ax3, prof.pres, prof.u, prof.v)
        ax2.set_yticks(np.linspace(100,1000,10))
        ax2.set_ylim(1050,40)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        ax2.xaxis.set_major_locator(plt.MultipleLocator(10))
        ax2.set_xlim(-50,50)

        ax2.legend(fontsize = 12, loc = 'lower left')




        #plt.savefig(pd.to_datetime(df.time.values[0]).strftime('/data/keeling/a/melinda3/RELAMPAGO_soundings//%b_%d_%y_%H%M.%S_integratedlms_07182022.png'), dpi=300)
        plt.show()
        plt.close()"""

        gpd_out = pd.concat([gpd_out,df], ignore_index=True)
        df_time_out.append(df.time)
        df_lat_out.append(df.lat)
        df_lon_out.append(df.lon)

    gpd_out['lapse_rate'] = jan25_lapserate
    gpd_out['n2'] = jan25_n2calc
    gpd_out['trop_height'] = trop_height
    gpd_out['w'] = w_max
    gpd_out['mucape'] = jan25_mucape
    gpd_out['trop_pt'] = trop_pt
    gpd_out['trop_pv'] = trop_pv
    gpd_out['trop_hypso'] = height_hypso
    gpd_out['trop_temp_lowest'] = trop_temp_lowest
    gpd_out['trop_temp_pt'] = trop_temp_pt
    gpd_out['trop_temp_pv'] = trop_temp_pv
    



    gpd_out.to_csv('./02nov_allday_out_final.csv')
