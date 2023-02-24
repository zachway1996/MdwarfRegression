#!/usr/bin/env python3

from tqdm import tqdm
import numpy as np
from astropy.io.votable import parse
from astropy.table import Table
import matplotlib.pyplot as plt
import gaiaxpy
from gaiaxpy.converter.converter import get_design_matrices
from gaiaxpy.converter.config import get_config, load_config
from gaiaxpy.input_reader.input_reader import InputReader
from gaiaxpy.config.paths import config_path
from tqdm import tqdm
from os import path
import mpl_scatter_density
from corner import corner
#import xgboost as xgb
import pickle
import pandas as pd
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch

print("Loading XP spec...")
temp = parse("XP_CONTINUOUS_COMBINED.xml_Gaia_DR3.vot")
table = temp.get_first_table().to_table()
data = table.to_pandas()

print("Loading in star metadata...")
df = pd.read_hdf("single_stars_sample.hdf")

print("Converting spectra to sampled versions")
eflux, ps_wvl = gaiaxpy.converter.converter.convert(data)

print("Measuring total RP flux")
rp_flux_tots = []
for i,row in eflux.iterrows():
    if row.xp=="BP":
        continue
    else:
        rp_flux_tots.append(np.trapz(np.array(row.flux), x=ps_wvl))

rp_flux_tots = np.array(rp_flux_tots)
data['rp_flux_tots'] = rp_flux_tots

# Spit out source_ids
data[["source_id"]].to_csv("source_ids.csv", index=False)

# Read in extra Gaia data and merge. Stuff like
# gs.ipd_frac_multi_peak,
# gs.ipd_frac_odd_win,
# xp.bp_n_transits,
# xp.bp_n_contaminated_transits,
# xp.bp_n_blended_transits,
# xp.rp_n_transits,
# xp.rp_n_contaminated_transits,
# xp.rp_n_blended_transits
extra_data = pd.read_csv("regression_xp_info-result.csv")
df = df.merge(extra_data)

print("Length of df is", len(df))
###

# Merge metadata and XP spectra
data = data.merge(df, on="source_id")

# Cut out data with >90% blended or contaminated transits
data = data[(1-data.rp_n_contaminated_transits/data.rp_n_transits > 0.9)& \
            (1-data.rp_n_blended_transits/data.rp_n_transits > 0.9)]

# Make huge array of coefficients
rp = np.array([row["rp_coefficients"] for i,row in data.iterrows()])
rp_f_t = np.array([row["rp_flux_tots"] for i,row in data.iterrows()])

# Correct coefficients for flux
rpc = np.array([rp[i]/rp_f_t[i] for i in range(len(rp))])


### 

# Begin Machine Learning
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

keep = np.ones(len(rpc), dtype=bool)

for i_ in tqdm(range(10)):
    print("Training on fraction of stars:", np.sum(keep)/len(data))
    # Set input and output data
    X = rpc[keep]
    y = np.array(data[["absG", "g_rp"]])[keep]
    X_train, X_test, i_train, i_test = train_test_split(X, data.index[keep], test_size=0.5, random_state=42)
    y_train = np.array(data.loc[np.array(i_train), ["absG", "g_rp"]])
    y_test  = np.array(data.loc[np.array(i_test), ["absG", "g_rp"]])
    absGmean, absGstd = np.mean(y.T[0]), np.std(y.T[0])
    g_rpmean, g_rpstd = np.mean(y.T[1]), np.std(y.T[1])
    np.array([absGmean, absGstd, g_rpmean, g_rpstd]).tofile("norm%s.csv"%(str(i_)), sep=",")
    y_train = np.array([(y_train.T[0] - absGmean)/absGstd , (y_train.T[1] - g_rpmean)/g_rpstd]).T
    y_test = np.array([(y_test.T[0] - absGmean)/absGstd , (y_test.T[1] - g_rpmean)/g_rpstd]).T
    
    # Set depth and train
    depth = 30
    Nest = 20
    rf = RandomForestRegressor(n_estimators = Nest, max_features = 'sqrt', max_depth = depth,
                           random_state = 18, n_jobs=-1).fit(X_train, y_train)
    
    #Save RF model
    with open("single_rf%s.pkl"%(str(i_)), 'wb') as pickle_file:
        pickle.dump(rf, pickle_file)
    
    print("X_test shape is", X_test.shape)
    
    prediction = rf.predict(X_test)
    prediction = np.array([prediction.T[0]*absGstd + absGmean,  \
                        prediction.T[1]*g_rpstd + g_rpmean]).T

    y_train = np.array([y_train.T[0]*absGstd + absGmean,  \
                        y_train.T[1]*g_rpstd + g_rpmean]).T
    
    y_test = np.array([y_test.T[0]*absGstd + absGmean,  \
                        y_test.T[1]*g_rpstd + g_rpmean]).T


    new_df = pd.DataFrame()
    new_df['train'] = np.concatenate([np.ones(len(y_train), dtype=bool), np.zeros(len(y_test), dtype=bool)])
    new_df["absG"] = np.concatenate([y_train.T[0], y_test.T[0]])
    new_df["g_rp"] = np.concatenate([y_train.T[1], y_test.T[1]])
    train_pred = rf.predict(X_train)
    new_df["p_absG"] = np.concatenate([train_pred.T[0]*absGstd + absGmean, prediction.T[0]])
    new_df["p_g_rp"] = np.concatenate([train_pred.T[1]*g_rpstd + g_rpmean, prediction.T[1]])
    new_df["source_id"] = np.concatenate([data.loc[np.array(i_train), "source_id"], \
                                          data.loc[np.array(i_test), "source_id"]])
    new_df.to_csv("rf_results%s.csv"%(str(i_)))
    
    try:
        # Plot it pretty
        fig = plt.figure(figsize=(8,8))
        dpi_ = None
        
        norm = ImageNormalize(stretch=LogStretch())
        
        ax1 = plt.subplot(221, projection="scatter_density")
        ax1.scatter_density(y_train.T[1], y_train.T[0],
                            color="#3b528b", dpi = dpi_, norm=norm)
        #ax1.scatter(prediction.T[1], prediction.T[0], s=1)
        #ax1.set_title("Training")
        ax1.set_xlabel("G-RP (mag)")
        ax1.set_ylabel(r"$M_G$ (mag)")
        ax1.invert_yaxis()
        
        ax2 = plt.subplot(222, projection="scatter_density")
        ax2.scatter_density(prediction.T[1], prediction.T[0],
                            color="#5ec962", dpi = dpi_, norm=norm)
        #ax2.set_title("Prediction")
        ax2.set_xlabel("G-RP (mag)")
        ax2.set_ylabel(r"$M_G$ (mag)")
        ax2.invert_yaxis()
        
        ax3 = plt.subplot(223, projection="scatter_density")
        ax3.scatter_density(y_test.T[1],
                    prediction.T[1] - y_test.T[1],
                            color="#440154", dpi = dpi_, norm=norm)
        ax3.set_xlabel("G-RP (mag)")
        ax3.set_ylabel(r"$\Delta$ G-RP (mag)")
        
        ax4 = plt.subplot(224, projection="scatter_density")
        ax4.scatter_density(y_test.T[1],
                            prediction.T[0] - y_test.T[0],
                            color="#440154", dpi = dpi_, norm=norm)
        ax4.set_xlabel("G-RP (mag)")
        ax4.set_ylabel(r"$\Delta~M_G$ (mag)")
        
        
        #plt.tight_layout()
        plt.savefig("AAS_4x4_%s.png"%(str(i_)), transparent=False, facecolor="white", dpi=150)
        plt.close()
    except:
        continue

    k_ = []
    for j,row in tqdm(data.iterrows()):
        msk = new_df.source_id==row.source_id
        if np.sum(msk)==0:
            k_.append(False)
        try:
            foo = new_df.loc[msk]
            test_val = foo.p_absG - foo.absG
            if float(test_val) >0.6:
                k_.append(False)
            else:
                k_.append(True)
        except:
            k_.append(False)
    print(len(k_))
    print(len(data))
    print(len(rpc))
    keep = np.array(k_)

