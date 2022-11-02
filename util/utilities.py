from sklearn.decomposition import PCA
import pandas as pd
import os

def get_project_name():
    return 'Fantasy_Football'

def get_project_dir():
    dir_name = get_project_name()
    cwd = os.getcwd()
    ind = cwd.index(dir_name)
    return cwd[:ind+len(dir_name)]+'/'

def gen_year_list(start_year,end_year):
    input_year_list = [start_year]
    for year in range(start_year+1,end_year+1):
        input_year_list.append(year)
    return input_year_list

'''
Looks under input path for file starting with input string, that contains an input year
Combine all and output as dataframe
'''
def aggregate_read_raw_data_files(inp_str,inp_path,inp_year_list):
    file_list=[]
    for fn in os.listdir(inp_path):
        if (fn.startswith(inp_str)):
            for year in inp_year_list:
                if (str(year) in fn):
                    file_list.append(pd.read_pickle(inp_path+fn))
    output_df = pd.concat(file_list)
    output_df = output_df.loc[
        ~output_df['posteam'].isnull()&
        ~output_df['defteam'].isnull()&
        (output_df['posteam']!='')&
        (output_df['defteam']!='')
     ]
    return output_df

def aggregate_read_data_files(inp_str,inp_path,inp_year_list):
    file_list=[]
    for fn in os.listdir(inp_path):
        if (fn.startswith(inp_str)):
            for year in inp_year_list:
                if (str(year) in fn):
                    file_list.append(pd.read_pickle(inp_path+fn))
    output_df = pd.concat(file_list).sort_values(['season','week','team']).drop_duplicates().reset_index(drop=True)
    return output_df

def filter_df_year(inp_df,start_year,end_year):
    return inp_df.loc[
        (inp_df['season']>=start_year) &
        (inp_df['season']<=  end_year)
    ]
