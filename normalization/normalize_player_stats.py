import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from scipy.stats import boxcox




# The QB stuff is split between stuff we can z-scale,
#  and stuff we can min-max normalize
def qb_normalization( 
                        inp_df   , 
                        norm_cols ,
                        scale_cols,
                        norm_dict
                     ):
    
    # Do z-scaling
    qb_scale_feats = smart_scale_columns( 
                                            inp_df, 
                                            scale_cols,
                                            norm_dict
                                        )
    
    # Do normalization
    qb_norm_feats = min_max_norm_columns( 
                                            inp_df, 
                                            norm_cols,
                                            norm_dict
                                        )

    return pd.concat( [qb_scale_feats,qb_norm_feats], axis=1 )




# Opposing team stuff
# Pretty much same for everyone
def opp_normalization( 
                        inp_df   , 
                        inp_cols ,
                        norm_dict,
                        team_fg_miss_str=None
                     ):
    
    n_comp = 4
    
    # Do z-scaling
    opp_norm_feats = smart_scale_columns( 
                                            inp_df, 
                                            inp_cols,
                                            norm_dict
                                        )

    
    # Do PCA if not in dict
    # Captures ~80%
    if ( 'opp_pca' in norm_dict.keys() ):
        opp_pca_norm = run_pca( 
                                    opp_norm_feats, 
                                    n_comp, 
                                    'opp',
                                    inp_pca=norm_dict['opp_pca'],
                                    return_pca=False
                               )
        
    else:
        opp_pca_norm, my_pca = run_pca( 
                                        opp_norm_feats, 
                                        n_comp, 
                                        'opp'
                                      )
        norm_dict['opp_pca']=my_pca

    out_df = pd.DataFrame()

    # Variable number of pca components
    for i in range( 0, n_comp ):

        st_i = str(i)

        # Pca values normal, zscale 'em
        col =  'opp_pca_'+st_i

        # Generate normalizations if they don't exist
        if not ( ( col+'_mean' in norm_dict.keys() ) &
                 ( col+'_std'  in norm_dict.keys() ) ):
            norm_dict[col+'_mean'] = opp_pca_norm[col].mean()
            norm_dict[col+'_std' ] = opp_pca_norm[col].std ()


        out_df[col+'_scaled'] = z_scale( 
                                         opp_pca_norm[col],  
                                         norm_dict[col+'_mean'],
                                         norm_dict[col+'_std' ]
                                       )
    return out_df


# Normalization for team stuff,
# Pretty much same for everyone
def team_normalization( 
                        inp_df   , 
                        inp_cols ,
                        norm_dict,
                        team_fg_miss_str=None
                      ):

    # Do z-scaling
    team_norm_feats = min_max_norm_columns( 
                                            inp_df, 
                                            inp_cols,
                                            norm_dict
                                          )
    
    # Do PCA if not in dict
    # Captures ~85%
    if ( 'team_pca' in norm_dict.keys() ):
        team_pca_norm = run_pca( 
                                        team_norm_feats, 
                                        2, 
                                        'team',
                                        inp_pca=norm_dict['team_pca'],
                                        return_pca=False
                               )
        
    else:
        team_pca_norm, my_pca = run_pca( 
                                        team_norm_feats, 
                                        2, 
                                        'team'
                                       )
        norm_dict['team_pca']=my_pca


        
        
    # Return these values, zscale of pca's
    col =  'team_pca_0'

    # Generate normalizations if they don't exist
    if not ( ( col+'_mean' in norm_dict.keys() ) &
             ( col+'_std'  in norm_dict.keys() ) ):
        norm_dict[col+'_mean'] = team_pca_norm[col].mean()
        norm_dict[col+'_std' ] = team_pca_norm[col].std ()

        
    # Our output
    output_df = pd.DataFrame()
        
    # Min/max normalize
    output_df['team_pca_0_scaled'] = z_scale( 
                                                team_pca_norm[col], 
                                                norm_dict[col+'_mean'],
                                                norm_dict[col+'_std']
                                            )


    col =  'team_pca_1'

    # Generate normalizations if they don't exist
    if not ( ( col+'_mean' in norm_dict.keys() ) &
             ( col+'_std'  in norm_dict.keys() ) ):
        norm_dict[col+'_mean'] = team_pca_norm[col].mean()
        norm_dict[col+'_std' ] = team_pca_norm[col].std ()


    output_df['team_pca_1_scaled'] = z_scale( 
                                              team_pca_norm[col],  
                                              norm_dict[col+'_mean'],
                                              norm_dict[col+'_std' ]
                                            )
    
    
    # If this variable is present, poissonian, mm norm
    if ( isinstance( team_fg_miss_str, str ) ):
        col = team_fg_miss_str
        # Generate normalizations if they don't exist
        if not ( ( col+'_min' in norm_dict.keys() ) &
                 ( col+'_max' in norm_dict.keys() ) ):
            norm_dict[col+'_min'] = inp_df[col].min()
            norm_dict[col+'_max'] = inp_df[col].max()

        output_df[col+'_scaled'] = min_max_norm( 
                                                inp_df[col],  
                                                norm_dict[col+'_min'],
                                                norm_dict[col+'_max' ]
                                               )
    
    
    return output_df


# Normalization of fumble stuff
# Overall the same process for everyone
def fumb_normalization( 
                        inp_df   , 
                        inp_cols ,
                        norm_dict
                      ):
    
    # Do min max normalization
    # Roughly poissonian
    fumb_norm_feats = min_max_norm_columns( 
                                            inp_df, 
                                            inp_cols,
                                            norm_dict
                                          )
    # Do PCA if not in dict
    # Covers about 80%
    if ( 'fumb_pca' in norm_dict.keys() ):
        fumb_pca_norm = run_pca( 
                                        fumb_norm_feats, 
                                        2, 
                                        'fumb',
                                        inp_pca=norm_dict['fumb_pca'],
                                        return_pca=False
                               )
        
    else:
        fumb_pca_norm, my_pca = run_pca( 
                                        fumb_norm_feats, 
                                        2, 
                                        'fumb'
                                       )
        norm_dict['fumb_pca']=my_pca

    # Perform box cox transformation, 
    #  to make the distribution more normal
    fumb_pca_norm['fumb_pca_1_bc'] = boxcox( fumb_pca_norm['fumb_pca_1']+1., -1 )




    # Return these values, min/max norm of pca 0, zscale of pca 1
    col =  'fumb_pca_0'

    # Generate normalizations if they don't exist
    if not ( ( col+'_min' in norm_dict.keys() ) &
             ( col+'_max' in norm_dict.keys() ) ):
        norm_dict[col+'_min'] = fumb_pca_norm[col].min()
        norm_dict[col+'_max'] = fumb_pca_norm[col].max()

        
    # Our output
    output_df = pd.DataFrame()
        
    # Min/max normalize
    output_df['fumb_pca_0_norm'] = min_max_norm( 
                                                fumb_pca_norm[col], 
                                                norm_dict[col+'_min'],
                                                norm_dict[col+'_max']
                                               )


    col =  'fumb_pca_1_bc'

    # Generate normalizations if they don't exist
    if not ( ( col+'_mean' in norm_dict.keys() ) &
             ( col+'_std'  in norm_dict.keys() ) ):
        norm_dict[col+'_mean'] = fumb_pca_norm[col].mean()
        norm_dict[col+'_std' ] = fumb_pca_norm[col].std()


    output_df['fumb_pca_1_bc_scaled'] = z_scale( 
                                                fumb_pca_norm[col],  
                                                norm_dict[col+'_mean'],
                                                norm_dict[col+'_std' ]
                                               )

    return output_df






# Some assertions to make sure the right variables are passed
def __run_scale_assertions( inp_df, cols, inp_dict, mod_name ):
    # Make sure we are dealing with a column list
    if ( not ( type(cols) is list ) ):
        cols = [cols]
        
    # Make sure we have an input dictionary
    assert ( type(inp_dict) is dict ), 'inp_dict must be a dict'
    
    # Make sure we have a string to modify the column names
    assert ( type(mod_name) is str  ), 'mod_name must be a string'
    
    # Make sure we have a dataframe
    assert ( isinstance(inp_df, pd.DataFrame) ), 'inp_df must be a pandas dataframe'

# Simple z scaling
def z_scale( inp_df, mu, sigma ):
    return ( inp_df - mu ) / sigma

# Simple min/max normalization
def min_max_norm( inp_df, min_val, max_val ):
    return 2. * ( ( inp_df  - min_val ) /
                  ( max_val - min_val ) ) - 1.

# Do z-scaling on columns
def z_scale_columns( inp_df, cols, inp_dict, mod_name='' ):

    # Make sure our data types are good
    __run_scale_assertions( inp_df, cols, inp_dict, mod_name )
        
    # Our output frame
    out_frame = pd.DataFrame()
    
    for col in cols:
        
        # Get a few names we will use
        col_norm = col+mod_name+'_scaled'
        col_mean = col_norm+'_mean'
        col_std  = col_norm+'_std'
        
        # Mean and standard deviation
        # If in dictionary, use, if not, set
        if ( ( col_mean in inp_dict.keys() ) & 
             ( col_std  in inp_dict.keys() ) ):
            pass
        else:
            inp_dict[ col_mean ] = inp_df[col].mean() 
            inp_dict[ col_std  ] = inp_df[col].std()
            
        mu  = inp_dict[ col_mean ]
        std = inp_dict[ col_std  ]
        
        out_frame[col_norm] = z_scale( inp_df[col], mu, std )
        
    return out_frame

# Do z-scaling on columns
def smart_scale_columns( inp_df, cols, inp_dict, mod_name='', n_sigma=2.0, tolerance=0.1 ):

    # Make sure our data types are good
    __run_scale_assertions( inp_df, cols, inp_dict, mod_name )
        
    # Our output frame
    out_frame = pd.DataFrame()
    
    for col in cols:
        
        # Get a few names we will use
        col_norm = col+mod_name+'_scaled'
        col_mean = col_norm+'_mean'
        col_std  = col_norm+'_std'
        
        # Mean and standard deviation
        # If in dictionary, use, if not, set
        if ( ( col_mean in inp_dict.keys() ) & 
             ( col_std  in inp_dict.keys() ) ):
            pass
        else:
            inp_dict[ col_mean ], inp_dict[ col_std  ] = \
                smart_scale( inp_df, col, n_sigma=n_sigma, tolerance=tolerance, ret_params=True )
            
        mu  = inp_dict[ col_mean ]
        std = inp_dict[ col_std  ]
        
        out_frame[col_norm] = z_scale( inp_df[col], mu, std )
        
    return out_frame

# Do min-max on columns
def min_max_norm_columns( inp_df, cols, inp_dict, mod_name='' ):

    # Make sure our data types are good
    __run_scale_assertions( inp_df, cols, inp_dict, mod_name )
        
    # Our output frame
    out_frame = pd.DataFrame()
    
    for col in cols:
        
        # Get a few names we will use
        col_norm = col+mod_name+'_norm'
        col_min  = col_norm+'_min'
        col_max  = col_norm+'_max'
        
        # Min and max
        # If in dictionary, use, if not, set
        if ( ( col_min in inp_dict.keys() ) & 
             ( col_max in inp_dict.keys() ) ):
            pass
        else:
            inp_dict[ col_min ] = inp_df[col].min() 
            inp_dict[ col_max ] = inp_df[col].max()
            
        min_val = inp_dict[ col_min ]
        max_val = inp_dict[ col_max ]
        
        out_frame[col_norm] = min_max_norm( inp_df[col], min_val, max_val )
        
    return out_frame


# Run pca analysis, can return pca, frame, or both
def run_pca( inp_df, n_comp, str_start, inp_pca=None, cols=None, return_pca=True, return_frame=True ):

    # Default to using all of the input columns
    if ( cols==None ):
        cols=inp_df.columns.values
    
    __run_scale_assertions( inp_df, cols, {'test':0},str_start)
    
    if ( inp_pca==None ):
        # PCA since lots of overlapping data
        my_pca = PCA( n_components=n_comp )
        my_pca.fit( inp_df[cols] )

        if ( not return_frame ):
            return my_pca
    else:
        my_pca = inp_pca
        
    # Get the transformed values
    pca_vals = my_pca.transform( inp_df[cols] )
    
    # Output frame
    out_df = pd.DataFrame()
    
    # Save transformed into our frame
    for i in range( 0, n_comp ):
        st_i = str(i)
        # Pca values normal, zscale 'em
        col =  str_start+'_pca_'+st_i
        out_df[col] = pca_vals[:,i]

    if ( return_pca & return_frame ):
        return out_df, my_pca
    return out_df





# Find z scale, shifting the distribution to zscale the underlying gaussian distribution, and ignore outliers
def smart_scale( inp_df              ,  # Data frame
                 column              ,  # Column of interest
                 lower_bound = True  ,  # Whether to trim lower boundary
                 upper_bound = True  ,  # Whether to trim upper boundary
                 n_sigma     =   2.0 ,  # Number of stds to use for trimming
                 tolerance   =   0.1 ,  # Tolerance for measuring convergence
                 max_steps   =  20   ,  # Maximum number of iterations
                 max_sigma   =   3.0 ,  # When done, number of stds to cut when calculating new distribution
                 ret_params  = False ,  # Whether to return the parameters used in scaling
                 show_plot   = False ): # Whether to show plots

    converged  = False  # Whether the series has converged
    counter    = 0      # Tracks so doesn't loop infinitely

    old_mean   = inp_df[column].mean() # Starting mean and standard deviation
    old_std    = inp_df[column].std ()

    new_column = inp_df[column].copy() # Make copy of df column

    low_lim    = new_column.min()      # For plotting
    hi_lim     = new_column.max()
    
    
    # Iterate until we converge
    while ( not converged and counter < max_steps ):


        counter = counter + 1

        # Trim column based on data points within standard deviation
        if ( lower_bound ):
            new_column =  new_column.loc[ new_column > ( old_mean - n_sigma * old_std ) ]
        
        if ( upper_bound ):
            new_column =  new_column.loc[ new_column < ( old_mean + n_sigma * old_std ) ]
            
        # Calculate new mean
        myMean = new_column.mean()
        myStd  = new_column.std()
            
        # If both mean and standard deviation not changing
        if ( abs( myMean/old_mean - 1.0 ) < tolerance and
             abs( myStd /old_std  - 1.0 ) < tolerance ):
            converged = True

        # Hold on to old means, for next iteration
        old_mean  = myMean
        old_std   = myStd

        # Plot the change
        if ( show_plot ):
            new_column.hist( bins=np.arange(low_lim, hi_lim, (hi_lim-low_lim)/20), normed=True )
            x = np.linspace( low_lim, hi_lim, 500 )
            plt.plot(x,mlab.normpdf(x, myMean, myStd))
            plt.xlim( low_lim, hi_lim )
            plt.title( 'Mean: %7.2f Std: %7.2f' % (new_column.mean(), new_column.std()) )
            plt.show()

            
    # If we failed to converge, abort
    if ( counter == max_steps ):
        print 'Z-scale failed to converge'
        return
    
    
    # Restart with original, trim off edges outside maximum sigma extent
    new_column = inp_df[column].copy()
    
    if ( lower_bound ):
        new_column =  new_column.loc[ new_column > ( myMean - max_sigma * myStd ) ]
    if ( upper_bound ):
        new_column =  new_column.loc[ new_column < ( myMean + max_sigma * myStd ) ]
    
    
    # Generate mean and std for points within the distribution
    gauss_mean = new_column.mean()
    gauss_std  = new_column.std()
    
    
    # Z-scale the distribution using the recovered gaussian mean and standard deviation
    new_column = inp_df[column].copy()
    new_column = (new_column - gauss_mean) / gauss_std
    
    
    # Plot the last thing
    if ( show_plot ):
        low_lim = -7
        hi_lim  =  7
        new_column.hist(bins=np.arange(low_lim, hi_lim, 0.5))#bins=10)
        x = np.linspace( low_lim, hi_lim, 500 )
        plt.plot(x,mlab.normpdf(x, 0, 1)*500)#200000)
        plt.xlim( low_lim, hi_lim )
        plt.title( 'Final: Mean: %7.2f Std: %7.2f' % (gauss_mean, gauss_std) )
        plt.show()

    if ( ret_params ):
        return gauss_mean, gauss_std
        
    return new_column
