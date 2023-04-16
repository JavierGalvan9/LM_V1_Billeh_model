# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 23:50:52 2022

@author: UX325
"""

import pandas as pd
import argparse
import dataframe_image as dfi
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns


#https://coderzcolumn.com/tutorials/python/simple-guide-to-style-display-of-pandas-dataframes
#https://pandas.pydata.org/docs/user_guide/style.html

def background_gradient(s, m, M, cmap='PuBu', low=0, high=0, two_color_palette=False):
    # Defines a gradient colouring taking into account the whole dataset values
    rng = M - m
    
    if two_color_palette:
        norm=colors.TwoSlopeNorm(vmin=m, vcenter=1., vmax=M)
    else:
        norm = colors.Normalize(m - (rng * low),
                                M + (rng * high))
        
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]


def make_pretty(styler, min_value=0, max_value=1, format_float="%.4f", two_color_palette=False):
#     styler.format("{:.2%}") # uncomment for probabilites format
    formatter = lambda s: '0' if s==0 else format_float %s 
    styler.format(formatter)
    
    if two_color_palette:
        cm = sns.color_palette('coolwarm', as_cmap=True)
    else:
        cm = sns.light_palette('red', as_cmap=True)

    styler.apply(background_gradient,
                 cmap=cm,
                 m=min_value, #interarea_connectivity.min().min(),
                 M=max_value, #interarea_connectivity.max().max(),
                 low=0,
                 high=1,
                 two_color_palette=two_color_palette)

    styler.set_table_styles(
                            [{"selector": 'th.col_heading', 'props': 'text-align: center;'},
                             {"selector": 'th.row_heading', 'props': 'text-align: center;'},
                             {"selector": "tbody td", "props": [("border", "1px solid gray")]},
                             {'selector': 'th.col_heading.level0', 'props': 'border-bottom: 2px solid black; background-color: white'},
                             {'selector': 'th.col_heading.level1', 'props': 'border-bottom: 2px solid black; border-left: 2px solid black; border-right: 2px solid black; background-color: white'},
                             {'selector': 'th.col_heading.level2', 'props': 'border-bottom: 2px solid black; border-left: 2px solid black; border-right: 2px solid black; background-color: white'},
                             {'selector': 'th.row_heading', 'props': 'border-right: 2px solid black; background-color: white'},
                             {'selector': 'th.row_heading.level1', 'props': 'border-top: 2px solid black; background-color: white'}
                            ])
    for l0 in styler.index:
        styler.set_table_styles({(l0): [{'selector': '', 'props': 'border-top: 2px solid black; border-bottom: 2px solid black;'}]
                                }, overwrite=False, axis=1)
    
    col_pops = styler.columns
    e_col_pops = [col_pop for col_pop in col_pops if 'E' in ' '.join(col_pop)]
    for e_col in e_col_pops:
        styler.set_table_styles({e_col: [{'selector': 'td', 'props': 'border-left: 2px solid black;'}]},
                                overwrite=False, axis=0)

    indices = [list(col_pops).index(e_col_pop) for e_col_pop in e_col_pops]
    for idx in indices:
        styler.set_table_styles(
                            [{'selector': f'th.col_heading.col{idx}', 'props': 'border-left: 2px solid black; background-color: white'}
                            ], overwrite=False)
    
    styler.set_table_styles(
                            [{'selector': 'td:last-child', 'props': 'border-right: 2px solid black'},
                             {'selector': 'th.row_heading.level1', 'props': 'background-color: white'},
                             {'selector': 'th.row_heading.level0', 'props': 'border-bottom: 0px solid black; border-top: 1px solid white; background-color: white'}
                            ], overwrite=False)
    
    return styler


def csv_to_pretty_png_table(target_column_name, source_column_name):   
    
    # interarea_connectivity_path = f'Source_{source_column_name}_Target_{target_column_name}/{source_column_name}_to_{target_column_name}_total_weights_matrix.csv'
    interarea_connectivity_path = f'Source_{source_column_name}_Target_{target_column_name}/{source_column_name}_to_{target_column_name}_connection_probability_matrix.csv'
    
    try:
        print(interarea_connectivity_path)
        interarea_connectivity = pd.read_csv(interarea_connectivity_path, sep='\t', decimal=',', index_col='Source')
        print(interarea_connectivity)
    except ValueError:
        print("Value error: The csv file may be full of zeros or at least it has not got the proper style.")
    
    # interarea_connectivity = interarea_connectivity.loc[~(interarea_connectivity==0).all(axis=1)] # Remove sources with zero connections
    interarea_connectivity.sort_index(axis=0, inplace=True)
    interarea_connectivity.sort_index(axis=1, inplace=True)
        
    allen_column_name_dict = {'VISp':'V1', 'VISl':'LM', 'billeh':'V1 Allen', 'VISli':'VISli'}
    target_column_allen = allen_column_name_dict[target_column_name]
    source_column_allen = allen_column_name_dict[source_column_name]
    
    cols = pd.MultiIndex.from_tuples([(f'Target: {target_column_allen}', "L1", "iHtr3a"), 
                                      (f'Target: {target_column_allen}', "L2/3", "E"), 
                                      (f'Target: {target_column_allen}', "L2/3", "iHtr3a"),
                                      (f'Target: {target_column_allen}', "L2/3", "iPvalb"),
                                      (f'Target: {target_column_allen}', "L2/3", "iSst"),
                                      (f'Target: {target_column_allen}', "L4", "E"),
                                      (f'Target: {target_column_allen}', "L4", "iHtr3a"),
                                      (f'Target: {target_column_allen}', "L4", "iPvalb"),
                                      (f'Target: {target_column_allen}', "L4", "iSst"),
                                      (f'Target: {target_column_allen}', "L5", "E"),
                                      (f'Target: {target_column_allen}', "L5", "iHtr3a"),
                                      (f'Target: {target_column_allen}', "L5", "iPvalb"),
                                      (f'Target: {target_column_allen}', "L5", "iSst"),
                                      (f'Target: {target_column_allen}', "L6", "E"),
                                      (f'Target: {target_column_allen}', "L6", "iHtr3a"),
                                      (f'Target: {target_column_allen}', "L6", "iPvalb"),
                                      (f'Target: {target_column_allen}', "L6", "iSst")
                                     ])
    
    new_index = pd.MultiIndex.from_tuples([(f'Source: {source_column_allen}', "L1", "iHtr3a"), 
                                      (f'Source: {source_column_allen}', "L2/3", "E"), 
                                      (f'Source: {source_column_allen}', "L2/3", "iHtr3a"),
                                      (f'Source: {source_column_allen}', "L2/3", "iPvalb"),
                                      (f'Source: {source_column_allen}', "L2/3", "iSst"),
                                      (f'Source: {source_column_allen}', "L4", "E"),
                                      (f'Source: {source_column_allen}', "L4", "iHtr3a"),
                                      (f'Source: {source_column_allen}', "L4", "iPvalb"),
                                      (f'Source: {source_column_allen}', "L4", "iSst"),
                                      (f'Source: {source_column_allen}', "L5", "E"),
                                      (f'Source: {source_column_allen}', "L5", "iHtr3a"),
                                      (f'Source: {source_column_allen}', "L5", "iPvalb"),
                                      (f'Source: {source_column_allen}', "L5", "iSst"),
                                      (f'Source: {source_column_allen}', "L6", "E"),
                                      (f'Source: {source_column_allen}', "L6", "iHtr3a"),
                                      (f'Source: {source_column_allen}', "L6", "iPvalb"),
                                      (f'Source: {source_column_allen}', "L6", "iSst")
                                     ])
        
    interarea_connectivity = pd.DataFrame(interarea_connectivity.values, index=new_index, columns=cols)
    interarea_connectivity = interarea_connectivity.loc[~(interarea_connectivity==0).all(axis=1)] # Remove sources with zero connections
    
    df = interarea_connectivity.style.pipe(make_pretty, min_value=interarea_connectivity.min().min(), max_value=interarea_connectivity.max().max())
        
    interarea_connectivity_table_path = interarea_connectivity_path.split('.')[0] + '_prettytable.png'
    dfi.export(df, interarea_connectivity_table_path, dpi=300)
    

def v1_comparison():
    v1_bbp_path = 'Source_VISp_Target_VISp/VISp_to_VISp_connection_probability_matrix.csv'
    bbp_v1 = pd.read_csv(v1_bbp_path, sep='\t', decimal=',', index_col='Source')
    
    v1_billeh_path = 'Source_billeh_Target_billeh/billeh_to_billeh_connection_probability_matrix.csv'
    billeh_v1 = pd.read_csv(v1_billeh_path, sep='\t', decimal=',', index_col='Source')
    
    models_ratio = billeh_v1/bbp_v1
    # models_ratio.to_csv('v1_connection_probability_ratio.csv', sep='\t', decimal=',')
    
    cols = pd.MultiIndex.from_tuples([('Target:', "L1", "iHtr3a"), 
                                      ('Target:', "L2/3", "E"), 
                                      ('Target:', "L2/3", "iHtr3a"),
                                      ('Target:', "L2/3", "iPvalb"),
                                      ('Target:', "L2/3", "iSst"),
                                      ('Target:', "L4", "E"),
                                      ('Target:', "L4", "iHtr3a"),
                                      ('Target:', "L4", "iPvalb"),
                                      ('Target:', "L4", "iSst"),
                                      ('Target:', "L5", "E"),
                                      ('Target:', "L5", "iHtr3a"),
                                      ('Target:', "L5", "iPvalb"),
                                      ('Target:', "L5", "iSst"),
                                      ('Target:', "L6", "E"),
                                      ('Target:', "L6", "iHtr3a"),
                                      ('Target:', "L6", "iPvalb"),
                                      ('Target:', "L6", "iSst")
                                     ])
    
    new_index = pd.MultiIndex.from_tuples([('Source:', "L1", "iHtr3a"), 
                                      ('Source:', "L2/3", "E"), 
                                      ('Source:', "L2/3", "iHtr3a"),
                                      ('Source:', "L2/3", "iPvalb"),
                                      ('Source:', "L2/3", "iSst"),
                                      ('Source:', "L4", "E"),
                                      ('Source:', "L4", "iHtr3a"),
                                      ('Source:', "L4", "iPvalb"),
                                      ('Source:', "L4", "iSst"),
                                      ('Source:', "L5", "E"),
                                      ('Source:', "L5", "iHtr3a"),
                                      ('Source:', "L5", "iPvalb"),
                                      ('Source:', "L5", "iSst"),
                                      ('Source:', "L6", "E"),
                                      ('Source:', "L6", "iHtr3a"),
                                      ('Source:', "L6", "iPvalb"),
                                      ('Source:', "L6", "iSst")
                                     ])
    
    models_ratio = pd.DataFrame(models_ratio.values, index=new_index, columns=cols)
    # models_ratio = models_ratio.style.pipe(make_pretty, min_value=models_ratio.min().min(), max_value=models_ratio.max().max(),
    #                                        format_float="%.1f", two_color_palette=True)
    models_ratio = models_ratio.style.pipe(make_pretty, min_value=1/5, max_value=5,
                                           format_float="%.1f", two_color_palette=True)
        
    models_ratio_table_path = v1_bbp_path.split('.')[0] + 'ratio_prettytable.png'
    dfi.export(models_ratio, models_ratio_table_path, dpi=300)
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_area",
                        default='VISp',
                        help="Brain area which acts as source")
    
    parser.add_argument("--target_area",
                        default='VISl',
                        help="Brain area which acts as target")
    
    parser.add_argument('--v1_comparison', 
                        action='store_true')

    
    args = parser.parse_args()
    csv_to_pretty_png_table(target_column_name=args.target_area, source_column_name=args.source_area)
    
    try:
        v1_comparison()
    except:
        pass
    
else:
    target_column_name = 'VISli'; source_column_name = 'VISli'
    csv_to_pretty_png_table(target_column_name, source_column_name)
