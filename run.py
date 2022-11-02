#sys.path.append('/home/sean/Documents/import_test/util')
from import_raw import run_raw_nfl_import
from normalize import normalize_raw_input
from model import model_generation
from analyze import explore_results
from util import utilities
from util import argument_validation

import pandas as pd
import argparse
import datetime
import os


def read_args():
    
    parser = argparse.ArgumentParser(description='Read and save data from nfl_data_py requests using input years')

    
    parser.add_argument('--ingest', action='store_true',
        help='Whether to ingest raw input'
    )
    parser.add_argument('--ingest_start_year', type=int, nargs='?', default=1999,
        help='The starting year for ingestion data (Earliest available data is from 1999)'
    )
    parser.add_argument('--ingest_end_year', type=int, nargs='?',default=datetime.date.today().year,
        help='The ending year for ingestion data (Latest is current year)'
    )

    
    parser.add_argument('--normalize', action='store_true',
        help='Whether to normalize input'
    )
    parser.add_argument('--norm_start_year', type=int, nargs='?', default=1999,
        help='The starting year for normalization data (Earliest available data is from 1999)'
    )
    parser.add_argument('--norm_end_year', type=int, nargs='?',default=datetime.date.today().year,
        help='The ending year for normalization data (Latest is current year)'
    )

    
    parser.add_argument('--generate_model', action='store_true',
        help='Whether to create a model or not'
    )
    parser.add_argument('--input_models_file_name', type=str, nargs='?',
        help='Optional ML file to load/use'
    )
    parser.add_argument('--output_models_file_name', type=str, nargs='?',
        help='Optional ML file to save, othwerwise uses date'
    )
    parser.add_argument('--process_start_year', type=int, nargs='?', default=1999,
        help='The starting year for model processing (Earliest available data is from 1999)'
    )
    parser.add_argument('--process_end_year', type=int, nargs='?',default=datetime.date.today().year,
        help='The ending year for model processing (Latest is current year)'
    )
    parser.add_argument('--n_rolling', type=int, nargs='?',default=3,
        help='The number of months for model lookback'
    )
    parser.add_argument('--n_components_team', type=int, nargs='?',default=24,
        help='The number of PCA components for the model team data'
    )
    parser.add_argument('--n_components_opp', type=int, nargs='?',default=15,
        help='The number of PCA components for the model opposing team data'
    )

    
    parser.add_argument('--input_scaler_file_name', type=str, nargs='?',
        help='Optional scaler file to load/use'
    )
    parser.add_argument('--output_scaler_file_name', type=str, nargs='?',
        help='Optional scaler file to save, otherwise uses date'
    )

    
    parser.add_argument('--predict_values', action='store_true',
        help='Whether to forecast values for predict years or not'
    )
    parser.add_argument('--prediction_file_name', type=str, nargs='?',
        help='Optional save data with prediction'
    )
    parser.add_argument('--predict_start_year', type=int, nargs='?', default=1999,
        help='The starting year for predicting data (Earliest available data is from 1999)'
    )
    parser.add_argument('--predict_end_year', type=int, nargs='?',default=datetime.date.today().year,
        help='The ending year for predicting data (Latest is current year)'
    )


    parser.add_argument('--analyze', action='store_true',
        help='Whether to generate output plots of predictions or not'
    )
    parser.add_argument('--top_stats_list', nargs='+',
        help='Prediction file name to load/use'
    )

    parser.add_argument('--input_version', type=str, nargs='?',
        default=run_raw_nfl_import.__import_version__,
        help='The version to use for import'
    )
    parser.add_argument('--normalization_version', type=str, nargs='?',
        default=normalize_raw_input.__normalization_version__,
        help='The version to use for normalization'
    )
    parser.add_argument('--model_version', type=str, nargs='?',
        default=model_generation.__model_version__,
        help='The version to use for models'
    )
    parser.add_argument('--analyze_version', type=str, nargs='?',
        default=explore_results.__analyze_version__,
        help='The version to use for analyzing results'
    )

    args = parser.parse_args()

    input_arguments = vars(args)

    argument_validation.run_argument_validation(input_arguments)

    return input_arguments

'''
Get arguments
Run ingest, normalize, generate model, <predict results>, explore results
'''
def main():

    input_args = read_args()

    print("Input arguments-")
    for arg in input_args:
        print("\t{:30s}: {:30s}".format(arg,str(input_args[arg])))
    input("Press enter to run the program...")

    if (input_args['ingest']):
        run_save_import(input_args)

    
    
if __name__ == "__main__":
    main()
