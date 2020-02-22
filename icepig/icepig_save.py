import numpy as np
import pandas as pd


def icepig_excel(path,array, colums_name, index_name=None, sheet_name='sheet1'):
    path_name = path
    data_df = pd.DataFrame( data=array,columns=colums_name)

    data_df.columns = colums_name
    data_df.index = index_name

    data_df.to_excel(path_name)
    #data_df.to_csv(path_or_buf=path_name,  float_format='%.4f')  # float_format 控制精度


    return