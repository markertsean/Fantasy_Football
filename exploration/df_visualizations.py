import pandas  as pd
import numpy   as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.cm     as cm

# Do cool correlation plot, with scatter and histograms
def corr_plot( inp_df, exclude = None, focus = None, y_label_rotation=70, x_label_rotation=20, x_tick_rotation=45 ):

    
    col_list = inp_df.columns.values
    
    # Remove elements we are excluding
    if ( exclude != None ):
        if type( exclude ) is list:
            for element in exclude:
                index    = np.argwhere( col_list==element )
                col_list = np.delete(   col_list, index   )
        else:
            index    = np.argwhere( col_list==exclude )
            col_list = np.delete(   col_list, index   )

    # Put at the bottom, for easy comparison
    if ( focus != None ):
        index    = np.argwhere( col_list==focus )
        col_list = np.delete(   col_list, index )
#        col_list = np.insert( col_list, 0, focus )
        col_list = np.append( col_list, focus )
    
    df     = inp_df[col_list].copy()    
    
    corr   = df.corr()
    corr_v = corr.as_matrix()
    
    # Mask the upper right so it can't be seen
#    mask = np.zeros_like(corr, dtype=np.bool)
#    mask[np.triu_indices_from(mask)] = True
#    mask = np.transpose( mask )
    

    # Plot the correlation with color background in upper right
    cmap = cm.get_cmap('coolwarm')
    axes = pd.plotting.scatter_matrix( df )
    for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
        axes[i, j].cla()
        axes[i, j].set_axis_bgcolor( cmap( 0.5 * corr_v[i,j] + 0.5) ) 
        axes[i, j].annotate("%0.3f" %corr_v[i,j], (0.5, 0.5), xycoords='axes fraction', ha='center', va='center')

    # Optionally rotate the x ticks
    for i, j in zip(*plt.np.tril_indices_from(axes, k=1)):
        for tick in axes[i,j].get_xticklabels():
            tick.set_rotation( x_tick_rotation )

    # Optionally rotate x label
    for ax in plt.gcf().axes:
        plt.sca(ax)
        plt.xlabel(ax.get_xlabel(), rotation=x_label_rotation)

    # Optionally rotate y label
    for ax in plt.gcf().axes:
        plt.sca(ax)
        plt.ylabel(ax.get_ylabel(), rotation=y_label_rotation)
        
#    plt.xticks( rotation=45 )
#    plt.yticks( rotation=45 )

    plt.show()    
    
# Can do histogram with strings
def hist_plot( inp_df, col ):
    df = inp_df.groupby(col).size()
    ax = df.plot(kind='bar', title=col )
    ax.set_xlabel( ' ' )
    ax.set_ylabel( 'Count' )
    plt.show()
    
# Bar plot for column taking average for y col
def plot_avg( inp_df, x_col, y_col ):
    
    df = inp_df[[x_col,y_col]].copy()
    
    means = df.groupby( x_col ).mean()
    std   = df.groupby( x_col ).std().fillna(0)

    ax = means.plot(kind='bar' , title=x_col, yerr=std, legend=False )
    ax.set_xlabel( ' ' )
    ax.set_ylabel( y_col )
    plt.show()