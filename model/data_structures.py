import pandas as pd
from sklearn.decomposition import PCA

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.multioutput import MultiOutputClassifier

import numpy as np

def get_index_resampled_categories_reduce_largest(inp_df,col,balance_sample):
    max_val = inp_df.shape[0]
    max_vals = []
    for case in inp_df[col].unique():
        max_vals.append(inp_df.loc[inp_df[col]==case].shape[0])
    sorted_vals = sorted(max_vals)
    sorted_vals = sorted_vals[:-1]
    if (len(sorted_vals)>0):
        max_val = sorted_vals[-1]

    # Resample the majority group
    concat_df_list = []
    for case in inp_df[col].unique():
        case_df = inp_df.loc[inp_df[col]==case]
        if ( int(balance_sample*max_val) < case_df.shape[0] ):
            case_df = case_df.sample(int(balance_sample*max_val))#,replace=True).drop_duplicates()
        concat_df_list.append(
            case_df
        )

    concat_df = pd.concat(concat_df_list)
    return concat_df.index


class ZScaler:
    def __init__(self,inp_df,columns=None):
        self.scale_dict = {}
        assert isinstance(inp_df,pd.DataFrame) # Input must be a dataframe
        if (columns is None):
            self.add(inp_df)
        else:
            if (not isinstance(columns,list)):
                columns = [columns]
            self.add(inp_df[columns])

    def __repr__(self):
        out_str = ""
        for key in self.scale_dict:
            out_str += "\tField: {:50s}\tMean: {:10.6f}\tStd: {:10.6f}\n".format(
                key,
                self.scale_dict[key]['mean'],
                self.scale_dict[key]['std'],
            )
        return out_str

    def __str__(self):
        return "Member of ZScaler\n"+self.__repr__()

    def add(self,inp_df):
        for col in inp_df.columns.values:
            self.scale_dict[col] = {}
            self.scale_dict[col]['mean'] = inp_df[col].mean()
            self.scale_dict[col]['std'] = inp_df[col].std()

    def remove(self,col):
        del self.scale_dict[col]

    def get_dict(self):
        return self.scale_dict

    def get(self,col,kind=None):
        if (kind is None):
            return self.scale_dict[col]
        elif((kind=='mean') or (kind=='std')):
            return self.scale_dict
        else:
            raise ValueError("kind must be 'mean' or 'std'")

    def scale_cols(self,inp_df,cols):
        out_df = inp_df.copy()
        for col in cols:
            assert col in self.scale_dict # Can only scale columns in the scaler!
            out_df[col]=(out_df[col]-self.scale_dict[col]['mean'])/self.scale_dict[col]['std']
            out_df[col]=out_df[col].fillna(0.0)
            out_df.rename(columns={col:col+'_z'},inplace=True)
        return out_df

#TODO: move w/ ^
class PCACols:
    def __init__(self,inp_df,columns,n_components):
        self.columns = columns
        self.n_components = n_components
        self.model_pca = PCA(n_components=n_components)
        self.model_pca.fit(inp_df[self.columns].dropna())

    def __repr__(self):
        out_str = ""
        out_str+= "\tN components = "+str(self.n_components)+"\n"
        out_str+= "\tN columns    = "+str(len(self.columns))+"\n"
        out_str+= "\tColumns = [\n"
        for col in self.columns:
            out_str+="\t\t"+str(col)+",\n"
        out_str+= "\t]\n"
        out_str+= "\tCumulative Explained Variance = [\n"
        for var in self.model_pca.explained_variance_ratio_.cumsum():
            out_str+="\t\t"+str(var)+",\n"
        out_str+= "\t]\n"
        return out_str

    def __str__(self):
        return "Member of PLCACols\n"+self.__repr__()

    def PCA(self):
        return self.model_pca

    def transform(self,inp_df):
        return self.model_pca.transform(inp_df[self.columns])

class PCATeamOpp:
    def __init__(
        self,
        scaler,
        team_df,
        team_cols,
        n_team,
        opp_df,
        opp_cols,
        n_opp,
    ):
        self.scaler      = scaler

        self.team_cols   = team_cols
        scaled_team_df   = scaler.scale_cols(team_df[self.team_cols],self.team_cols)
        self.team_cols_s = scaled_team_df.columns
        self.team_PCA    = PCACols(scaled_team_df,self.team_cols_s,n_team)

        self.opp_cols    = opp_cols
        scaled_opp_df    = scaler.scale_cols(opp_df[opp_cols],opp_cols)
        self.opp_cols_s  = scaled_opp_df.columns
        self.opp_PCA     = PCACols(scaled_opp_df,self.opp_cols_s,n_opp)

    def __str__(self):
        return self.team_PCA.__str__() + "\n" + self.opp_PCA.__str__()

    def team_transform(
        self,
        team_df,
    ):
        team_scaled_df = self.scaler.scale_cols(team_df[self.team_cols],self.team_cols)
        return self.team_PCA.transform(team_scaled_df)

    def opp_transform(
        self,
        opp_df,
    ):
        opp_scaled_df = self.scaler.scale_cols(opp_df[self.opp_cols],self.opp_cols)
        return self.opp_PCA.transform(opp_scaled_df)

    def transform(
        self,
        team_df,
        opp_df
    ):
        return np.concatenate([self.team_transform(team_df),self.opp_transform(opp_df)],axis=1)

class ModelWrapper:
    def __init__(
        self,
        input_x_df,
        input_y_df,
        scaler,
        pca_obj,
        key_fields=['season','week','team','opponent']
    ):
        self.x_df = input_x_df.dropna()
        self.y_df = input_y_df.loc[
            input_y_df.index.intersection(self.x_df.index)
        ].dropna()

        assert self.x_df.shape[0]==self.y_df.shape[0] # X and Y must have same dimension
        self.key_fields = key_fields

        assert isinstance(scaler,ZScaler), "input scalar must be of class ZScaler"
        self.scaler = scaler

        assert isinstance(pca_obj,PCATeamOpp), "input pca_obj must be of class PCATeamOpp"
        self.pca_obj = pca_obj
        
        self.model_dict = {}
        self.col_dict = {}
        self.cv_dict = {}

    def __get_values__(self,input_df,cols):
        return input_df.drop(columns=self.key_fields)[cols].values

    def get_model_dict(self):
        return self.model_dict

    def get_model_predicted_fields(self):
        return self.col_dict.keys()

    def get_model_predictor_fields(self):
        return self.col_dict

    def get_cv_dict(self):
        return self.cv_dict

    def model(self,name):
        assert name in self.model_dict
        return self.model_dict[name]

    def get_scaler(self):
        return self.scaler

    def scale_cols(self,inp_df,cols):
        return self.scaler.scale_cols(inp_df,cols)

    def get_pca_obj(self):
        return self.pca_obj

    def transform(self,team_df,opp_df):
        return self.pca_obj.transform(team_df,opp_df)

    def train_model(
        self,
        names,
        model,
        parameters=None,
        use_cols=None,
        test_size=0.20,
        n_jobs=1,
        scoring=None,
        cv=3,
        balance_sample=None,
        multiclass=False,
    ):
        if isinstance(names,str):
            names=[names]
        assert isinstance(names,list)
        for name in names:
            assert name in self.y_df.columns.values

        for name in names:
            if (use_cols is None):
                use_cols = self.x_df.drop(columns=self.key_fields).columns.values
            self.col_dict[name] = use_cols

            x_df = self.x_df
            y_df = self.y_df
            if (balance_sample is not None):
                assert isinstance(balance_sample,float), "balance_sample must be float"

                valid_indexes = get_index_resampled_categories_reduce_largest(self.y_df,name,balance_sample)
                x_df = self.x_df.loc[valid_indexes]
                y_df = self.y_df.loc[valid_indexes]

            features = self.__get_values__(x_df,use_cols)
            values   = self.__get_values__(y_df,name)

            x_shuf, y_shuf = shuffle( features, values )
            x_train, x_test, y_train, y_test = train_test_split( x_shuf, y_shuf, test_size=test_size )

            this_model = model
            if multiclass:
                this_model = MultiOutputClassifier(this_model)

            if (parameters is None):
                self.model_dict[name] = model.fit(x_train,y_train)
            else:
                gscv = GridSearchCV( model, parameters, n_jobs=n_jobs, scoring=scoring,cv=cv )
                gscv.fit(x_train, y_train)
                self.cv_dict[name] = gscv
                self.model_dict[name] = gscv.best_estimator_.fit(x_train,y_train)

            print("Fit model for "+name+", test data score=",str(self.model_dict[name].score(x_test,y_test)))

            if (parameters is not None):
                print(gscv.best_params_)

    def predict(self,name,inp_df):
        assert isinstance(name,str)
        assert name in self.model_dict

        features = self.__get_values__(inp_df,self.col_dict[name])
        return self.model_dict[name].predict(features)

    def predict_proba(self,name,inp_df):
        assert isinstance(name,str)
        assert name in self.model_dict

        features = self.__get_values__(inp_df,self.col_dict[name])
        return self.model_dict[name].predict_proba(features)

class MLTrainingHelper:
    #TODO:PCA DF
    def __init__(
        self,
        joined_df,
        scaled_joined_df,
        pca_df,
        reg_model,
        clf_model,
        key_fields = ['season','week','team','opponent'],
    ):
        assert reg_model in ['Linear','Forest','MLP','SVM','KNN']
        assert clf_model in ['Logistic','Forest','MLP','SVM','KNN']

        self.reg_model = reg_model
        self.clf_model = clf_model

        self.regressor_input_x_dict = {
            'Linear': scaled_df,
            'Forest': joined_df,
            'MLP'   : joined_df,
            'SVM'   : scaled_joined_df,
            'KNN'   : scaled_joined_df,
        }

        self.classifier_input_x_dict = {
            'Logistic': scaled_df,
            'Forest'  : joined_df,
            'MLP'     : scaled_joined_df,
            'SVM'     : scaled_joined_df,
            'KNN'     : scaled_joined_df,
        }
        #self.regressor_input_x_dict = {
        #    'Linear': joined_df,
        #    'Forest': joined_df,
        #    'MLP'   : joined_df,
        #    'SVM'   : scaled_joined_df,
        #    'KNN'   : scaled_joined_df,
        #}
        #
        #self.classifier_input_x_dict = {
        #    'Logistic': scaled_joined_df,
        #    'Forest'  : joined_df,
        #    'MLP'     : scaled_joined_df,
        #    'SVM'     : scaled_joined_df,
        #    'KNN'     : scaled_joined_df,
        #}

        self.regressor_model_dict = {
            'Linear': LinearRegression(),
            'Forest':RandomForestRegressor(),
            'MLP':MLPRegressor(
                activation='identity', # This is consistently best
                solver='adam',
                max_iter=1000,
            ),
            'SVM':SVR(),
            'KNN':KNeighborsRegressor(),
        }

        self.classifier_model_dict = {
            'Logistic': LogisticRegression(max_iter=1000),
            'Forest':RandomForestClassifier(),
            'MLP':MLPClassifier(
                activation='identity', # This is consistently best
                solver='adam',
                max_iter=1000,
            ),
            'SVM':SVC(),
            'KNN':KNeighborsClassifier(),
        }

        # For NN models
        n_features = (self.regressor_input_x_dict[self.reg_model]).drop(columns=key_fields).shape[1]
        n_   = n_features
        n_23 = int(n_features*2./3)
        n_2  = int(n_features/2.)
        n_3  = int(n_features/3.)

        self.regressor_cv_param_dict = {
            'Linear':{
                'fit_intercept':[True,False],
            },
            'Forest':{
                'n_estimators':[100,300,500],
                'max_depth':[None,10,15],
                #'max_features':[None,'sqrt'],
                #'min_samples_leaf':[1,0.1,0.05,0.01],
            },
            'MLP':{
                'hidden_layer_sizes': [
                    (n_  ,),
                    (n_23,),
                    (n_2 ,),
                    (n_3 ,),

                    (n_  , n_23,),
                    (n_  , n_2 ,),
                    (n_23, n_  ,),
                    (n_23, n_2 ,),
                    (n_2 , n_23,),
                    (n_2 , n_3 ,),
                    (n_3 , n_2 ,),

                    (n_  , n_23, n_  ,),
                    (n_  , n_23, n_23,),
                    (n_  , n_23, n_2 ,),
                    (n_  , n_23, n_3 ,),

                    (n_23, n_  , n_  ,),
                    (n_23, n_23, n_23,),
                    (n_23, n_2 , n_2 ,),
                    (n_23, n_3 , n_3 ,),
                ],
            },
            'SVM':{
                'C':10.**(np.arange(-2, 1, 1.0)),
                'kernel':['poly','rbf','sigmoid'],
                'gamma':['scale','auto'],
            },
            'KNN':{
                'n_neighbors': [10,15,20,25],
                'weights': ['uniform','distance'],
            },
        }

        n_features = (self.classifier_input_x_dict[self.clf_model]).drop(columns=key_fields).shape[1]
        n_   = n_features
        n_23 = int(n_features*2./3)
        n_2  = int(n_features/2.)
        n_3  = int(n_features/3.)

        self.classifier_cv_param_dict = {
            'Logistic':{
                'fit_intercept':[True,False],
                'C':10.**(np.arange(-4, 1, 0.5))
            },
            'Forest':{
                'n_estimators':[3,10,30,100],
                'max_depth':[None,10,15],
                #'max_features':[None,'sqrt'],
                #'min_samples_leaf':[1,0.1,0.05,0.01],
            },
            'MLP':{
                'hidden_layer_sizes': [
                    (n_  ,),
                    (n_23,),
                    (n_2 ,),
                    (n_3 ,),

                    (n_  , n_23,),
                    (n_  , n_2 ,),
                    (n_23, n_  ,),
                    (n_23, n_2 ,),
                    (n_2 , n_23,),
                    (n_2 , n_3 ,),
                    (n_3 , n_2 ,),

                    (n_  , n_23, n_  ,),
                    (n_  , n_23, n_23,),
                    (n_  , n_23, n_2 ,),
                    (n_  , n_23, n_3 ,),

                    (n_23, n_  , n_  ,),
                    (n_23, n_23, n_23,),
                    (n_23, n_2 , n_2 ,),
                    (n_23, n_3 , n_3 ,),
                ],
            },
            'SVM':{
                'C':10.**(np.arange(-2, 1, 1.0)),
                'kernel':['poly','rbf','sigmoid'],
                'gamma':['scale','auto'],
            },
            'KNN':{
                'n_neighbors': [10,15,20,25],
                'weights': ['uniform','distance'],
            },
        }


    def get_regressor_model(self):
        return self.regressor_model_dict[self.reg_model]

    def get_regressor_model_cv_params(self):
        return self.regressor_cv_param_dict[self.reg_model]

    def get_regressor_input_x(self):
        return self.regressor_input_x_dict[self.reg_model]

    def get_classifier_model(self):
        return self.classifier_model_dict[self.clf_model]

    def get_classifier_model_cv_params(self):
        return self.classifier_cv_param_dict[self.clf_model]

    def get_classifier_input_x(self):
        return self.classifier_input_x_dict[self.clf_model]
