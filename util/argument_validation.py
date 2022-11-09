import argparse
import datetime
import os
import sys

#TODO: proper implementation
sys.path.append('/home/sean/Documents/Fantasy_Football/')

from model import model_generation

from import_raw import run_raw_nfl_import
from normalize import normalize_raw_input
from analyze import explore_results


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

        valid_reg_models = ['Linear','SVM','Forest','MLP','KNN']
        assert (input_arguments['reg_model_type'] in valid_reg_models), "reg_model_type must be one of "+' '.join(valid_reg_models)

        valid_clf_models = ['Logistic','SVM','Forest','MLP','KNN']
        assert (input_arguments['clf_model_type'] in valid_clf_models), "clf_model_type must be one of "+' '.join(valid_clf_models)


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


def build_args(
    in_list
):
    parser = argparse.ArgumentParser(description='Read and save data from nfl_data_py requests using input years')

    if("ingest" in in_list):
        in_list.remove("ingest")
        parser.add_argument('--ingest', action='store_true',
            help='Whether to ingest raw input'
        )
    if("input_version" in in_list):
        in_list.remove("input_version")
        parser.add_argument('--input_version', type=str, nargs='?',
            default=run_raw_nfl_import.__import_version__,
            help='The version to use for import'
        )
    if("ingest_start_year" in in_list):
        in_list.remove("ingest_start_year")
        parser.add_argument('--ingest_start_year', type=int, nargs='?', default=1999,
            help='The starting year for ingestion data (Earliest available data is from 1999)'
        )
    if("ingest_end_year" in in_list):
        in_list.remove("ingest_end_year")
        parser.add_argument('--ingest_end_year', type=int, nargs='?',default=datetime.date.today().year,
            help='The ending year for ingestion data (Latest is current year)'
        )


    if("normalize" in in_list):
        in_list.remove("normalize")
        parser.add_argument('--normalize', action='store_true',
            help='Whether to normalize input'
        )
    if("normalization_version" in in_list):
        in_list.remove("normalization_version")
        parser.add_argument('--normalization_version', type=str, nargs='?',
            default=normalize_raw_input.__normalization_version__,
            help='The version to use for normalization'
        )
    if("norm_start_year" in in_list):
        in_list.remove("norm_start_year")
        parser.add_argument('--norm_start_year', type=int, nargs='?', default=1999,
            help='The starting year for normalization data (Earliest available data is from 1999)'
        )
    if("norm_end_year" in in_list):
        in_list.remove("norm_end_year")
        parser.add_argument('--norm_end_year', type=int, nargs='?',default=datetime.date.today().year,
            help='The ending year for normalization data (Latest is current year)'
        )


    if("generate_model" in in_list):
        in_list.remove("generate_model")
        parser.add_argument('--generate_model', action='store_true',
            help='Whether to create a model or not'
        )
    if("model_version" in in_list):
        in_list.remove("model_version")
        parser.add_argument('--model_version', type=str, nargs='?',
            default=model_generation.__model_version__,
            help='The version to use for models'
        )
    if("input_models_file_name" in in_list):
        in_list.remove("input_models_file_name")
        parser.add_argument('--input_models_file_name', type=str, nargs='?',
            help='Optional ML file to load/use'
        )
    if("output_models_file_name" in in_list):
        in_list.remove("output_models_file_name")
        parser.add_argument('--output_models_file_name', type=str, nargs='?',
            help='Optional ML file to save, othwerwise uses date'
        )
    if("process_start_year" in in_list):
        in_list.remove("process_start_year")
        parser.add_argument('--process_start_year', type=int, nargs='?', default=1999,
            help='The starting year for model processing (Earliest available data is from 1999)'
        )
    if("process_end_year" in in_list):
        in_list.remove("process_end_year")
        parser.add_argument('--process_end_year', type=int, nargs='?',default=datetime.date.today().year,
            help='The ending year for model processing (Latest is current year)'
        )
    if("reg_model_type" in in_list):
        in_list.remove("reg_model_type")
        parser.add_argument('--reg_model_type', type=str, nargs='?',
            default='Linear',
            help='The ML model to use for regression [Linear,SVM,Forest,MLP]'
        )
    if("clf_model_type" in in_list):
        in_list.remove("clf_model_type")
        parser.add_argument('--clf_model_type', type=str, nargs='?',
            default='Logistic',
            help='The ML model to use for classification [Logistic,SVM,Forest,MLP]'
        )
    if("n_rolling" in in_list):
        in_list.remove("n_rolling")
        parser.add_argument('--n_rolling', type=int, nargs='?',default=3,
            help='The number of months for model lookback'
        )
    if("n_components_team" in in_list):
        in_list.remove("n_components_team")
        parser.add_argument('--n_components_team', type=int, nargs='?',default=24,
            help='The number of PCA components for the model team data'
        )
    if("n_components_opp" in in_list):
        in_list.remove("n_components_opp")
        parser.add_argument('--n_components_opp', type=int, nargs='?',default=15,
            help='The number of PCA components for the model opposing team data'
        )


    if("predict_values" in in_list):
        in_list.remove("predict_values")
        parser.add_argument('--predict_values', action='store_true',
            help='Whether to forecast values for predict years or not'
        )
    if("prediction_file_name" in in_list):
        in_list.remove("prediction_file_name")
        parser.add_argument('--prediction_file_name', type=str, nargs='?',
            help='Optional save data with prediction'
        )
    if("predict_start_year" in in_list):
        in_list.remove("predict_start_year")
        parser.add_argument('--predict_start_year', type=int, nargs='?', default=1999,
            help='The starting year for predicting data (Earliest available data is from 1999)'
        )
    if("predict_end_year" in in_list):
        in_list.remove("predict_end_year")
        parser.add_argument('--predict_end_year', type=int, nargs='?',default=datetime.date.today().year,
            help='The ending year for predicting data (Latest is current year)'
        )


    if("analyze" in in_list):
        in_list.remove("analyze")
        parser.add_argument('--analyze', action='store_true',
            help='Whether to generate output plots of predictions or not'
        )
    if("analyze_version" in in_list):
        in_list.remove("analyze_version")
        parser.add_argument('--analyze_version', type=str, nargs='?',
            default=explore_results.__analyze_version__,
            help='The version to use for analyzing results'
        )
    if("top_stats_list" in in_list):
        in_list.remove("top_stats_list")
        parser.add_argument('--top_stats_list', nargs='+',
            help='Prediction file name to load/use'
        )

    if (len(in_list)>0):
        raise ValueError("Unrecognized arguments: "+" ".join(in_list))

    args = parser.parse_args()

    input_arguments = vars(args)

    run_argument_validation(input_arguments)

    return input_arguments
