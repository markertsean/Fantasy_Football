import argparse
import datetime
import os
import sys

#TODO: proper implementation
sys.path.append('/home/sean/Documents/Fantasy_Football/')

from model import model_generation


def run_input_year_check(input_arguments,start_str,end_str):
    assert input_arguments[start_str]>=1999, start_str+" must be greater than 1999!"
    assert input_arguments[end_str]<=datetime.date.today().year, end_str+" must be less than "+str(datetime.date.today().year)+"!"
    assert input_arguments[start_str]<=input_arguments[end_str], end_str+" must be greater than "+start_str

def run_input_gt_0_check(input_arguments,arg_str):
    assert (input_arguments[arg_str]>0), arg_str+" must be greater than 0!"

def run_argument_validation(input_arguments):

    if (
        ('ingest' in input_arguments) and
        input_arguments['ingest']
    ):
        run_input_year_check(input_arguments,'ingest_start_year','ingest_end_year')

    if (
        ('normalize' in input_arguments) and
        (input_arguments['normalize'])
    ):
        run_input_year_check(input_arguments,'norm_start_year','norm_end_year')

    if (
        ('generate_model' in input_arguments) and
        (input_arguments['generate_model'])
    ):
        run_input_year_check(input_arguments,'norm_start_year','norm_end_year')
        run_input_year_check(input_arguments,'process_start_year','process_end_year')
        run_input_gt_0_check(input_arguments,'n_rolling')
        run_input_gt_0_check(input_arguments,'n_components_team')
        run_input_gt_0_check(input_arguments,'n_components_opp')

        if (input_arguments['input_scaler_file_name'] is not None):
            fn = model_generation.get_model_path(
                input_arguments['model_version']
            )+input_arguments['input_scaler_file_name']
            if ( not os.path.exists(fn) ):
                raise IOError("File does not exist: "+fn)
    
        
    if (
        ('predict_values' in input_arguments) and
        (input_arguments['predict_values'])
    ):
        run_input_year_check(input_arguments,'predict_start_year','predict_end_year')
        
        assert (
            (
                ('generate_model' in input_arguments) and
                (input_arguments['generate_model'])
            ) or
            (
                ('input_models_file_name' in input_arguments) and
                (input_arguments['input_models_file_name'] is not None)
            )
        ), "If predict_values is set, either generate_model or input_models_file_name must be set"
        run_input_year_check(input_arguments,'process_start_year','process_end_year')
        
        if (
            ('input_models_file_name' in input_arguments) and
            (input_arguments['input_models_file_name'] is not None)
        ):
            fn = model_generation.get_model_path(
                input_arguments['model_version']
            )+input_arguments['input_models_file_name']
            if ( not os.path.exists(fn) ):
                raise IOError("File does not exist: "+fn)

        if (
            ('prediction_file_name' in input_arguments) and
            (input_arguments['prediction_file_name'] is None)
        ):
            input_arguments['prediction_file_name'] = 'predictions_{}_{}_{}.pkl'.format(
                input_arguments['predict_start_year'],
                input_arguments['predict_end_year'],
                datetime.date.today().strftime('%Y%m%d')
            )

    if (
        ('analyze' in input_arguments) and
        (input_arguments['analyze'])
    ):
        fn = "X"
        if (
            ('prediction_file_name' in input_arguments) and
            (input_arguments['prediction_file_name'] is not None)
        ):
            fn = model_generation.get_prediction_path(
                input_arguments['model_version']
            )+input_arguments['prediction_file_name']
        assert (
            (
                ('predict_values' in input_arguments) and
                (input_arguments['predict_values'])
            ) or
            (
                os.path.exists(fn)
            )
        ), "If analyze flag set, must either have predict_values set or a valid prediction file!"

        if (input_arguments['top_stats_list'] is None):
            input_arguments['top_stats_list'] = [1]

        for i in range(0,len(input_arguments['top_stats_list'])):
            input_arguments['top_stats_list'][i] = int(input_arguments['top_stats_list'][i])
