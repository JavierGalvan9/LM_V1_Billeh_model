# -*- coding: utf-8 -*-

import os
import pandas as pd

# This code constructs the units data frame used for Figure 1, Figure 3,
# and Extended Data Figures 1, 4, and 10

# 1. Download pre-computed response metrics from AllenSDK

all_metrics = pd.read_csv("../Neuropixels_data/cortical_metrics_1.4.csv")

# 2. Add time to first spike (not included in data release)

first_spike = pd.read_csv(os.path.join('data', 'time_to_first_spike.csv'),index_col=0)
all_metrics = all_metrics.merge(first_spike, on='ecephys_unit_id', sort=False)

# 3. Add intrinsic timescale and response decay timescale

timescale_metrics = pd.read_csv(os.path.join('data', 'timescale_metrics.csv'),
                                             index_col=0, low_memory=False)
all_metrics = all_metrics.merge(timescale_metrics, on='ecephys_unit_id', how='outer')

# 5. Write to a CSV file for use by other scripts
all_metrics.to_csv(os.path.join('data', 'unit_table.csv'), index=False)

