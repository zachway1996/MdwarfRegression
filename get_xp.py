#!/usr/bin/env python3

# This file reads in the "single_stars_sample.hdf" file and retrieves 
# their XP spectra. Will need to log into Gaia archive!
# This is also largely stolen from https://www.cosmos.esa.int/web/gaia-users/archive/datalink-products

from astropy.table import Table, vstack
from astroquery.gaia import Gaia
import numpy as np
import pandas as pd

def chunks(lst, n):
    ""
    "Split an input list into multiple chunks of size =< n"
    ""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

Gaia.login()

df = pd.read_hdf("single_stars_sample.hdf")

dl_threshold = 1000               # DataLink server threshold. It is not possible to download products for more than 5000 sources in one single call.
ids          = df.source_id
ids_chunks   = list(chunks(ids, dl_threshold))
datalink_all = []


print(f'* Input list contains {len(ids)} source_IDs')
print(f'* This list is split into {len(ids_chunks)} chunks of <= {dl_threshold} elements each')

retrieval_type = 'XP_CONTINUOUS'        # Options are: 'EPOCH_PHOTOMETRY', 'MCMC_GSPPHOT', 'MCMC_MSC', 'XP_SAMPLED', 'XP_CONTINUOUS', 'RVS' 
data_structure = 'COMBINED'   # Options are: 'INDIVIDUAL', 'COMBINED', 'RAW' - but as explained above, we strongly recommend to use COMBINED for massive downloads.
data_release   = 'Gaia DR3'   # Options are: 'Gaia DR3' (default), 'Gaia DR2'
dl_key         = f'{retrieval_type}_{data_structure}.xml'


ii = 0
for chunk in ids_chunks:
    ii = ii + 1
    print(f'Downloading Chunk #{ii}; N_files = {len(chunk)}')
    datalink  = Gaia.load_data(ids=chunk, data_release = data_release, retrieval_type=retrieval_type, format = 'votable', data_structure = data_structure)
    datalink_all.append(datalink)


if 'RVS' not in dl_key and 'XP_SAMPLED' not in dl_key:
    temp       = [inp[dl_key][0].to_table() for inp in datalink_all]
    merged     = vstack(temp)
    file_name  = f"{dl_key}_{data_release.replace(' ','_')}.vot"

    print(f'Writting table as: {file_name}')
    merged.write(file_name, format = 'votable', overwrite = True)

    #display(merged)