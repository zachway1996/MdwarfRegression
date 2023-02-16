#!/usr/bin/env python3

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

fig = plt.figure(figsize=(12,6))
ax1 = plt.subplot(221, projection="scatter_density")
sc1 = ax1.scatter_density(df.ra, df.dec, c=1-df.rp_n_contaminated_transits/df.rp_n_transits)
plt.colorbar(sc1, ax=ax1, label=r"$1-N_{CONTAM}/N_{TOT}$")

ax2 = plt.subplot(222, projection="scatter_density")
sc2 = ax2.scatter_density(df.ra, df.dec, c=1-df.rp_n_blended_transits/df.rp_n_transits)
plt.colorbar(sc2, ax=ax2, label=r"$1-N_{BLEND}/N_{TOT}$")

ax3 = plt.subplot(223)
ax3.hist(1-df.rp_n_contaminated_transits/df.rp_n_transits, bins=100)
ax3.set_yscale("log")
ax3.set_ylabel("Count")
ax3.set_xlabel(r"$1-N_{CONTAM}/N_{TOT}$")
ax3.axvline(0.9, color="black")

ax4 = plt.subplot(224)
ax4.hist(1-df.rp_n_blended_transits/df.rp_n_transits, bins=100)
ax4.set_yscale("log")
ax4.set_ylabel("Count")
ax4.set_xlabel(r"$1-N_{BLEND}/N_{TOT}$")
ax4.axvline(0.9, color="black")
for ax in [ax1,ax2]:
    ax.set_xlabel("Right Ascension")
    ax.set_ylabel("Declination")
#plt.tight_layout()
plt.savefig("quality_test.png", dpi=150, transparent=False, facecolor="white")
plt.close()

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

# Set input and output data
X = rpc
y = np.array(data[["absG", "g_rp"]])
absGmean, absGstd = np.mean(y.T[0]), np.std(y.T[0])
g_rpmean, g_rpstd = np.mean(y.T[1]), np.std(y.T[1])
y= np.array([(y.T[0] - absGmean)/absGstd , (y.T[1] - g_rpmean)/g_rpstd]).T
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Set depth and train
depth = 30
Nest = 20
rf = RandomForestRegressor(n_estimators = Nest, max_features = 'sqrt', max_depth = depth,
                           random_state = 18, n_jobs=-1).fit(X_train, y_train)

#Save RF model
with open("single_rf.pkl", 'wb') as pickle_file:
    pickle.dump(rf, pickle_file)

print("X_test shape is", X_test.shape)

prediction = rf.predict(X_test)
prediction = np.array([prediction.T[0]*absGstd + absGmean,  \
                        prediction.T[1]*g_rpstd + g_rpmean]).T

y_train = np.array([y_train.T[0]*absGstd + absGmean,  \
                        y_train.T[1]*g_rpstd + g_rpmean]).T

y_test = np.array([y_test.T[0]*absGstd + absGmean,  \
                        y_test.T[1]*g_rpstd + g_rpmean]).T

# Plot prediction
fig = plt.figure(figsize=(12,4))

ax1 = plt.subplot(131)
ax1.scatter(y_train.T[1], y_train.T[0], s=1)
#ax1.scatter(prediction.T[1], prediction.T[0], s=1)
ax1.set_title("Train")

ax2 = plt.subplot(132)
ax2.scatter(y_test.T[1], y_test.T[0], s=1)
#ax2.scatter(prediction.T[1], prediction.T[0], s=1)
ax2.set_title("Test")

ax3 = plt.subplot(133)
ax3.scatter(prediction.T[1], prediction.T[0], s=1)
ax3.set_title("Prediction")

for ax in [ax1,ax2,ax3]:
    ax.set_xlim(0.5,1.65)
    ax.set_ylim(5.2,18)
    ax.invert_yaxis()
    ax.set_ylabel("M_G")
    ax.set_xlabel("G-RP")
#plt.tight_layout()
plt.savefig("random_forest.png", dpi=200, facecolor="white", transparent=False)
plt.close()

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
plt.savefig("AAS_4x4.png", transparent=False, facecolor="white", dpi=150)
plt.close()
