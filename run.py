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
import pickle as pkl

def __read_args__():
    
    return argument_validation.build_args([
        'ingest',
        'input_version',
        'ingest_start_year',
        'ingest_end_year',
        'normalize',
        'normalization_version',
        'norm_start_year',
        'norm_end_year',
        'generate_model',
        'model_version',
        'input_models_file_name',
        'output_models_file_name',
        'process_start_year',
        'process_end_year',
        'reg_model_type',
        'clf_model_type',
        'n_rolling',
        'n_components_team',
        'n_components_opp',
        'predict_values',
        'prediction_file_name',
        'predict_start_year',
        'predict_end_year',
        'analyze',
        'analyze_version',
        'top_stats_list',
    ])

'''
Get arguments
Run ingest, normalize, generate model, <predict results>, explore results
'''
def main(input_args):
    print("Input arguments-")
    for arg in input_args:
        print("\t{:30s}: {:30s}".format(arg,str(input_args[arg])))
    input("Press enter to run the program...")

    if (input_args['ingest']):
        print("Running ingestion job")
        run_raw_nfl_import.run_save_import(input_args)
        print("Finished ingestion job")

    if (input_args['normalize']):
        print("Running normalization job")
        normalize_raw_input.normalize(input_args)
        print("Finished normalization job")

    if (input_args['generate_model'] or input_args['predict_values']):
        print("Running model/predict job")
        model_generation.run_model_gen_prediction(input_args)
        print("Finished model/predict job")

    if (input_args['analyze']):
        print("Running analyze job")
        explore_results.analyze(input_args)
        print("Finished analyze job")

if __name__ == "__main__":
    input_args = __read_args__()
    main(input_args)
