import numpy  as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

area_classification_list = [ 'MSSubClass', 'MSZoning', 'Neighborhood', 'Condition1', 'Condition2', 'MoSold', 
                             'YrSold', 'SaleType', 'SaleCondition' ] 
road_list                = [ 'LotFrontage', 'Street', 'Alley', 'PavedDrive' ]
shape_list               = [ 'LotArea', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope' ]
home_list                = [ 'Utilities', 'BldgType', 'HouseStyle', 'Heating', 'CentralAir',
                             '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                             'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                             'Fireplaces', 'GarageType', 'GarageCars', 'GarageArea', 'WoodDeckSF',
                             'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                             'MiscFeature', 'MiscVal']
quality_list             = [ 'OverallQual', 'OverallCond', 'HeatingQC', 'YearBuilt', 'YearRemodAdd', 
                             'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
                             'MasVnrArea', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                             'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 
                             'TotalBsmtSF', 'Electrical', 'KitchenQual', 'FireplaceQu', 'GarageYrBlt', 'GarageQual',
                             'GarageFinish', 'GarageCond', 'PoolQC', 'Fence' ]


MSSub_map    = { 20:0, 30:1, 40:2, 45:3, 50:4, 60:5, 70:6, 75:7, 80:8, 85:9, 90:10, 120:11, 150:12, 160:13, 180:14, 190:15 }
MSZoning_map = { 'A':0, 'C': 1, 'C (all)':1, 'FV':2, 'I':3, 'RH':4, 'RL':5, 'RP':6, 'RM':7, 'na':1, 'nan':1 } # MSZoning
road_map     = { 'NA':0, 'Grvl':1, 'Pave':2, 'NaN':0, 'NAN':0 } # Street, Alley
shape_map    = { 'Reg':0, 'IR1':1, 'IR2':2, 'IR3':3 }
contour_map  = { 'Lvl':0, 'Bnk':1, 'HLS':2, 'Low':3 }
util_map     = { 'AllPub':0, 'NoSewr':1, 'NoSeWa':2, 'ELO':3 }
config_map   = { 'Inside':0, 'Corner':1, 'CulDSac':2, 'FR2':3, 'FR3':4 }
slope_map    = { 'Gtl':0, 'Mod':1, 'Sev':2 }
neigh_map    = { 'Blmngtn':24, 'Blueste':0, 'BrDale':1, 'BrkSide':2, 'ClearCr':3, 'CollgCr':4, 'Crawfor':5,
                 'Edwards':6,'Gilbert':7, 'IDOTRR':8, 'MeadowV':9, 'Mitchel':10, 'Names':11, 'NAmes':11,
                 'NoRidge':12, 'NPkVill':13, 'NridgHt':14, 'NWAmes':15, 'OldTown':16, 'SWISU':17,
                 'Sawyer':18, 'SawyerW':19, 'Somerst':20, 'StoneBr':21, 'Timber':22, 'Veenker':23 }
cond_map     = { 'Norm':0, 'Feedr':1, 'Artery':1, 'RRNn':2, 'PosN':3, 'PosA':3, 'RRNe':2, 'RRAe':2, 'RRAn':2 } # 2 Railroad, 3 park/etc
bldg_map     = { '1Fam':0, '2FmCon':1, '2fmCon':1, 'Duplex':2, 'TwnhsE':3, 'Twnhs':3, 'TwnhsI':3 }
style_map    = { '1Story':0, '1.5Fin':1, '1.5Unf':2, '2Story':3, '2.5Unf':4, '2.5Fin':5 ,'SFoyer':6, 'SLvl':7 }
roofSty_map  = { 'Flat':0, 'Gable':1, 'Gambrel':2, 'Hip':3, 'Mansard':4, 'Shed':5 }
roofMat_map  = { 'ClyTile':0, 'CompShg':0, 'Membran':0, 'Metal':0, 'Roll':0, 'Tar&Grv':1, 'WdShake':2, 'WdShngl':3 }

exter_map    = { 'AsbShng':0, 'AsphShn':0, 'BrkComm':0,	'Brk Cmn':0, 'BrkFace':0, 'CBlock':0, 'CemntBd':0, 'CmentBd':0, 'HdBoard':2, 'ImStucc':0,	
                 'MetalSd':3, 'Other':0, 'Plywood':1,  'PreCast':0,  'Stone':0, 'Stucco':0, 'VinylSd':0, 'Wd Sdng':4, 
                 'WdShing':0, 'Wd Shng':0 }

masn_map     = { 'None':0, 'BrkCmn':0, 'BrkFace':1, 'CBlock':0, 'Stone':2 }
#'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond', 'PoolQC'
qual_map     = { 'NA':0, 'nan':0, 'NaN':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5 } 
found_map    = { 'BrkTil':0, 'CBlock':1, 'PConc':2, 'Slab':3, 'Stone':4, 'Wood':5 }
expo_map     = { 'NA':0, 'No':1, 'Mn':2, 'Av':2, 'Gd':2 }
basFin_map   = { 'NA':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6 } # BsmtFinType1, BsmtFinType2
heat_map     = { 'Floor':0, 'GasA':1, 'GasW':1, 'Grav':2, 'OthW':3, 'Wall':4 } # 1 any gas
yesno_map    = { 'No':0, 'N':0, 'n':0, 'no':0,'NO':0, 'Y':1, 'y':1, 'Yes':1, 'yes':1, 'YES':1, 'P':2} # CentralAir, PavedDrive
elec_map     = { 'SBrkr':0, 'FuseA':1, 'FuseF':2, 'FuseP':3, 'Mixed':4, 'Mix':4 }
func_map     = { 'Typ':0, 'Min1':1, 'Min2':2, 'Mod':3, 'Maj1':4, 'Maj2:':5, 'Sev':6, 'Sal':7 }
garT_map     = { 'NA':0, 'Detchd':1, 'CarPort':2, 'BuiltIn':3, 'Basement':4, 'Basment':4, 'Attchd':5, '2Types':6 }
garF_map     = { 'NA':0, 'Unf':1, 'RFn':2, 'Fin':3 }
fence_map    = { 'NA':0, 'MnWw':1, 'GdWo':2, 'MnPrv':3, 'GdPrv':4 }
feat_map     = { 'NA':0, 'NaN':0, 'TenC':1, 'Shed':1, 'Elev':1, 'Gar2':1, 'Othr':1 }
salType_map  = { 'WD':0, 'CWD':0, 'New':1, 'COD':2, 'VWD':3, 'Con':3, 'ConLw':3, 'ConLI':3, 'ConLD':3, 'Oth':3, 'na':3, 'nan':3 }
salCond_map  = { 'Normal':0, 'Abnorml':1, 'AdjLand':0, 'Alloca':2, 'Family':3, 'Partial':4 }

bsmtquall    = { 'NA':0, 'nan':0, 'NaN':0, 'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4 }
grgquall     = { 'NA':0, 'nan':0, 'NaN':0, 'Po':1, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':3 }
externcond   = { 'NA':0, 'nan':0, 'NaN':0, 'Po':0, 'Fa':0, 'TA':1, 'Gd':2, 'Ex':3 }


# Fill nans with most common values
def fill_nans( inp_df, column ):
    modeVal =  inp_df[ column ].mode()
    return     inp_df[ column ].fillna( modeVal )

# Remap all the quality stuff to integers
def remap_quality( quality_df, binary ):
    foo = quality_df[ ['ExterQual', 'FireplaceQu', 'KitchenQual'#, 'GarageQual','PoolQC'
                      ] ].replace( qual_map ).fillna(0).copy()

    foo.fillna( 0, inplace=True )
    
    # Drop BsmtCond
    # Drop bsmtfintype
    # Remove 0,1,5 from externcond
    # Drop poolQC
    # Drop roofstyle
    # Roofmatl all in 1, also leave 5 6 7
    # Drop Exterior2nd
    # BsmtExposure, change to 0 (none), 1 (no exposure), 2 (exposure)
    # Drop BsmtFinType2
    # Drop GarageCond
    
    foo['RoofStyle']    = fill_nans( quality_df, 'RoofStyle' )
#    foo['RoofMatl']     = fill_nans( foo, 'RoofMatl'  )
    
    foo['FireplaceQu']  = foo['FireplaceQu'].replace( {1:1, 2:1, 3:1, 4:2, 5:2} )
    foo['ExterQual']    = foo['ExterQual'].replace( {1:0, 2:0, 3:0, 4:1, 5:1 } )
    #foo['GaragerQual']  = foo['GarageQual'].replace( {2:0, 0:1, 3:2, 4:2, 5:2} )
    
    foo['BsmtQual']     = quality_df['BsmtQual']    .replace(   bsmtquall ).fillna( 0 ).replace( {1:0, 2:0, 3:1, 4:2} )
    foo['ExterCond']    = quality_df['ExterCond']   .replace(  externcond ).fillna( 0 ).replace( {1:0, 2:1, 3:1} )
    foo['HeatingQC']    = quality_df['HeatingQC']   .replace(  externcond ).fillna( 0 )
    foo['RoofStyle']    = quality_df['RoofStyle']   .replace( roofSty_map ).replace( {0:1,2:1,4:1,5:1} )
#    foo['RoofMatl']     = quality_df['RoofMatl']    .replace( roofMat_map )
    foo['Exterior1st']  = quality_df['Exterior1st'] .replace(   exter_map ).fillna( 0 )
    foo['Exterior2nd']  = quality_df['Exterior2nd'] .replace(   exter_map ).fillna( 0 )
    foo['MasVnrType']   = quality_df['MasVnrType']  .replace(    masn_map ).fillna( 0 )
    foo['Bsmt']         = quality_df['BsmtExposure'].replace(    expo_map ).fillna( 0 )
    foo['BsmtFinType1'] = quality_df['BsmtFinType1'].replace(  basFin_map ).fillna( 0 )
    foo['BsmtFinType2'] = quality_df['BsmtFinType2'].replace(  basFin_map ).fillna( 0 ).replace( {2:0,3:0,4:0,5:0,6:0,7:0} )
    foo['Electrical']   = quality_df['Electrical']  .replace(    elec_map ).fillna( 0 ).replace( {2:0, 3:0, 4:0, 5:0} )
    foo['GarageFinish'] = quality_df['GarageFinish'].replace(    garF_map ).fillna( 0 )
    foo['Fence']        = quality_df['Fence']       .replace(   fence_map ).fillna( 0 ).replace( {1:0, 2:1, 3:2, 4:3} )
    
    foo['KitchenQual']  = foo['KitchenQual'].replace( {0:foo['KitchenQual'].mode().values[0]} )


    if ( binary ):
        return binary_classification( foo )
    else:
        return foo

# Remap all the home list stuff to integers
def remap_home( quality_df, binary ):
    bigList = [ 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces' ]

    # Make total SF from 1st&2nd Floor
    # Make column multistory
    # Make column Wood Deck Open Porch Closed Porch
    # Drop lowqualfinsf
    # Combine bathrooms
    # Drop utilities
    # Drop heating
    # Drop miscfeatures
    # Drop bed above ground
    # Minimum kitchen above ground is 1
    # Drop garage cars
    # Drop pool area
    # Drop misc val
    
    foo = quality_df[ bigList ].copy()
    
#    foo['Utilities']    = fill_nans( quality_df, 'Utilities' )
#    foo['Heating']      = fill_nans( quality_df, 'Heating'   )

    #foo['InsideSF']   = quality_df['1stFlrSF']     + quality_df['2ndFlrSF']
    foo['InsideSF']   = quality_df['GrLivArea'].astype(float)
    foo['OutsideSF']  =(quality_df['WoodDeckSF']   + quality_df['OpenPorchSF'] +\
                        quality_df['EnclosedPorch']+ quality_df['3SsnPorch']   +\
                        quality_df['ScreenPorch']).astype(float)
        
    foo['MultiStory'] =(quality_df['2ndFlrSF']>0 ).astype(int)
    
    foo['NBath']      = (quality_df['BsmtFullBath'] + quality_df['BsmtHalfBath'] +\
                         quality_df['FullBath']     + quality_df['HalfBath']).fillna(0)
    foo['NBath']      = foo['NBath'].replace( {5:4, 6:4} )
#    foo['Utilities']  = quality_df['Utilities']  .replace(  util_map )
    foo['BldgType']   = quality_df['BldgType']   .replace(  bldg_map )#.replace( {1:3, 3:1} )
    foo['HouseStyle'] = quality_df['HouseStyle'] .replace( style_map )#.replace( {3:0, 0:1, 1:2, 2:1, 4:1, 5:1, 6:3, 7:4})
#    foo['Heating']    = quality_df['Heating']    .replace(  heat_map )
    foo['CentralAir'] = quality_df['CentralAir'] .replace( yesno_map )
    foo['GarageType'] = quality_df['GarageType'] .replace(  garT_map ).fillna(0).replace({ 4:1, 1:2, 5:4, 6:4 }).astype(int)
    foo['GarageArea'] = quality_df['GarageArea']                      .fillna(0).astype(float)
    foo['MiscFeature']= quality_df['MiscFeature'].replace(  feat_map ).fillna( 0 )
    
    foo['KitchenAbvGr'] = foo['KitchenAbvGr'].fillna( 0 ).replace( {0:1, 3:2} )
    foo['Fireplaces']   = foo['Fireplaces'].fillna(0).replace( {3:2, 4:2, 5:2} )
#    foo[ foo['Fireplaces'] == 4 ] = 2
#    foo[ foo['HouseStyle'] == 7 ] = foo['HouseStyle'].mode().values[0]
#    foo['HouseStyle'] = foo['HouseStyle'].replace( {5:foo['HouseStyle'].mode().values[0]} )
   
    if ( binary ):
        return binary_classification( foo, ignore=['NBath','TotRmsAbvGrd'] )
    else:
        return foo

def remap_area( inp_df, binary ):
    bigList = [ 'YrSold', 'MSSubClass', 'MSZoning', 'Neighborhood', 'Condition1', 'Condition2', #'MoSold', 
                'SaleType', 'SaleCondition' ]

    foo                  = inp_df[ ['YrSold','MSSubClass'] ].copy()
    foo['MSSubClass']    = inp_df['MSSubClass']   .replace(    MSSub_map )#.replace({0:0, 5:1, 11:2, 6:3, 4:4, 10:5, 1:10, 2:7, 3:0, 7:6, 8:8, 9:9, 12:0, 13:0, 14:0, 15:0 })
    foo['MSZoning']      = inp_df['MSZoning']     .replace( MSZoning_map )#.replace({0:0, 3:0, 5:0, 6:0, 8:0, 7:1, 1:2, 4:0, 2:3})
    foo['MSZoning']      = foo['MSZoning'].fillna( 0 )
    foo['Neighborhood']  = inp_df['Neighborhood'] .replace(    neigh_map )
    foo['Condition1']    = inp_df['Condition1']   .replace(     cond_map )
    foo['Condition2']    = inp_df['Condition2']   .replace(     cond_map )
    foo['SaleType']      = inp_df['SaleType']     .replace(  salType_map )
    foo['SaleType']      = foo['SaleType'].fillna( 0 )
    foo['SaleCondition'] = inp_df['SaleCondition'].replace(  salCond_map )
    foo['Condition']     = foo['Condition1']
    foo = foo.drop( ['Condition1','Condition2'], axis=1 )

#    foo[ foo['Neighborhood'] == 16 ] = foo['Neighborhood'].mode().values[0]
#    foo[ foo['MSSubClass']   == 15 ] = foo['MSSubClass']  .mode().values[0]
#    foo[ foo['MSSubClass']   == 12 ] = foo['MSSubClass']  .mode().values[0]
#    foo[ foo['MSZoning']     ==  5 ] = foo['MSZoning'  ]  .mode().values[0]
    foo['MSSubClass'] = foo['MSSubClass'].replace( {12:foo['MSSubClass'].mode().values[0]} )
    foo['MSZoning'  ] = foo['MSZoning'  ].replace( { 0:foo['MSZoning'  ].mode().values[0]} )
    
    
    if ( binary ):
        return binary_classification( foo )
    else:
        return foo

def remap_road( inp_df, binary ):
    bigList = [ 'LotFrontage', 'Alley', 'PavedDrive' ]

    foo               = inp_df[ bigList ].replace( road_map ).copy()
    foo['Alley'     ] = inp_df['Alley'     ].replace(  road_map ).fillna(0).astype(int)
#    foo['Street'    ] = inp_df['Street'    ].replace(  road_map )
    foo['PavedDrive'] = inp_df['PavedDrive'].replace( yesno_map )

    foo = foo.fillna( 0 )

    if ( binary ):
        return binary_classification( foo )
    else:
        return foo


def remap_land( inp_df, binary ):
    bigList = [ 'LotArea', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope' ]

    foo                = inp_df[ bigList ].copy()
    foo['junk']        = foo['LotShape']
    foo['LotShape']    = foo['LotShape']   .replace( shape_map ).replace( {3:2} )
    foo['LandContour'] = foo['LandContour'].replace( contour_map ).replace( {2:1,3:1,4:1,5:1} )
    foo['LotConfig']   = foo['LotConfig']  .replace(  config_map ).replace( {4:3} )  
    foo['LandSlope']   = foo['LandSlope']  .replace(   slope_map ).replace( {2:1} )
    
    foo['LotArea'  ]   = np.log10( foo['LotArea'] )
    
    foo = foo[bigList]
    
    if ( binary ):
        return binary_classification( foo )
    else:
        return foo

# Normalize index
def normalize_column( inp_df, column, maxVal=None, minVal=None ):
    
    new_column = inp_df.copy()
    
    max_value = maxVal
    min_value = minVal
    
    if( max_value == None ):
        max_value = inp_df[column].max()
    if( min_value == None ):
        min_value = inp_df[column].min()
        
    new_column[column] = ( inp_df[ column ] - float(min_value) ) / ( max_value - min_value )
    new_column.ix[ new_column[column]<0, column ] = 0.0
    new_column.ix[ new_column[column]>1, column ] = 1.0

    return 2*new_column[column]-1.

# Find normalization values within input sigma, returns normalization parameters that ignores outliers
def normalize_column_sigma( inp_df, column, lower_bound=True, upper_bound=True, n_sigma=3.0 ):
    
    new_column = inp_df[column].copy()
    
    myMean = new_column.mean()
    myStd  = new_column.std()
    
    if ( lower_bound ):
        new_column =  new_column[ new_column > ( myMean - n_sigma * myStd ) ]
        
    if ( upper_bound ):
        new_column =  new_column[ new_column < ( myMean + n_sigma * myStd ) ]
        
    return normalize_column( inp_df, column, minVal=new_column.min(), maxVal=new_column.max())

# Perform normalization on categories and areas
def normalize_homes( inp_df ):
    
    my_df = inp_df.drop( ['InsideSF', 'OutsideSF', 'GarageArea', 'LotFrontage', 'LotArea'], axis=1 )
    
    # Normalize categories from 0 to 1
    for col in my_df.columns.values:
        if ( len(my_df[col].unique()) > 2 ):
            my_df[col] = normalize_column( my_df, col )
    
    # Inside top and bot, log first
    # Outside top and bot log first
    # Garage upper
    # Frontage upper
    # Are upper and lower
    my_df['InsideSF']    = np.log10( inp_df['InsideSF']+1   )
    my_df['OutsideSF']   = np.log10( inp_df['OutsideSF']+1  )
    my_df['GarageArea']  = np.log10( inp_df['GarageArea']+1 )
    my_df['LotFrontage'] = np.log10( inp_df['LotFrontage'] / np.sqrt( inp_df['LotArea'] ) +1 )
    
    my_df['InsideSF']    = normalize_column_sigma( my_df, 'InsideSF'  )
    my_df['OutsideSF']   = normalize_column_sigma( my_df, 'OutsideSF' )

    my_df['LotArea']     = normalize_column_sigma( inp_df, 'LotArea'  )
    my_df['LotFrontage'] = normalize_column_sigma( inp_df, 'LotFrontage', lower_bound=False )
    my_df['GarageArea']  = normalize_column_sigma( inp_df, 'GarageArea' , lower_bound=False )

    return my_df

def run_kfold( clf, train_x_df, train_y_df, nf=10 ):
    
    kf = KFold( n_splits=nf )
    kf.get_n_splits( train_x_df )
    
    outcomes = []
    fold = 0
    
    # Generate indexes for training and testing data
    for train_index, test_index in kf.split( train_x_df ):
        
        fold += 1
        
        # Generate training and testing data from input sets
        x_train = train_x_df.iloc[train_index]
        y_train = train_y_df.iloc[train_index]
        x_test  = train_x_df.iloc[test_index]
        y_test  = train_y_df.iloc[test_index]
        
        
        # Foooooooo
        clf.fit( x_train, y_train )
        predictions = clf.predict( x_test )
        accuracy = r2_score( y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
        
    clf.fit( train_x_df, train_y_df )    
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0} +/- {1}".format(mean_outcome,np.std(outcomes)))

def run_kfold_arr( clf, train_x_df, train_y_df, nf=10 ):
    
    kf = KFold( n_splits=nf )
    kf.get_n_splits( train_x_df )
    
    outcomes = []
    fold = 0
    
    # Generate indexes for training and testing data
    for train_index, test_index in kf.split( train_x_df ):
        
        fold += 1
        
        # Generate training and testing data from input sets
        x_train = train_x_df[train_index]
        y_train = train_y_df[train_index]
        x_test  = train_x_df[test_index]
        y_test  = train_y_df[test_index]
        
        
        # Foooooooo
        clf.fit( x_train, y_train )
        predictions = clf.predict( x_test )
        accuracy = r2_score( y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
        
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0} +/- {1}".format(mean_outcome,np.std(outcomes)))
    
def optimize_fit( clf, train_x, train_y, grid_params, nf=10, verbose=True ):
    
    kf = KFold( n_splits=nf )
    kf.get_n_splits( train_x )
    
    outcomes = []
    clf_list = []
    fold = 0
    
    # Generate indexes for training and testing data
    for train_index, test_index in kf.split( train_x ):
        
        fold += 1
        
        # Generate training and testing data from input sets
        x_train = train_x[train_index]
        y_train = train_y[train_index]
        x_test  = train_x[test_index]
        y_test  = train_y[test_index]
        
        
        # 
        
        new_clf = GridSearchCV( clf, grid_params ) 
        
        new_clf.fit( x_train, y_train )
        predictions = new_clf.predict( x_test )
        accuracy = r2_score( y_test, predictions )
        outcomes.append(accuracy)
        
        clf_list.append( clone( new_clf.best_estimator_ ) )
        if ( verbose ):
#            print("Fold {0} accuracy: {1}".format(fold, accuracy)), ', ', new_clf.best_score_, new_clf.best_params_
            print "Fold %2i accuracy: %6.4f " % (fold, accuracy), ', ', '%6.4f '%new_clf.best_score_, new_clf.best_params_
        
    best_clf_score = 0
    best_clf_index = 0
    best_clf_acc   = 0
    
    clf_num = 0

    print ' '

    # Check each winning CLF against the group
    # and pick the best of the best
    for test_clf in clf_list:
        
        accuracies = []
        fold       = 0
        
        for train_index, test_index in kf.split( train_x ):
        
            fold += 1

            # Generate training and testing data from input sets
            x_train = train_x[train_index]
            y_train = train_y[train_index]
            x_test  = train_x[test_index]
            y_test  = train_y[test_index]
        
            test_clf.fit( x_train, y_train )

            predictions = test_clf.predict( x_test )
            accuracy = r2_score( y_test, predictions )
            accuracies.append(accuracy)
        
        mean_outcome = np.mean( accuracies )
        print "Clf %2i Mean Accuracy: %6.4f +/- %6.4f" % (clf_num,mean_outcome,np.std(accuracies))

        # Figure out which clf is the best
        if ( mean_outcome > best_clf_score ):
            best_clf_index = clf_num
            best_clf_score = mean_outcome
            
        clf_num = clf_num + 1
    
    # Fit and return best fit
    ret_clf = clf_list[ best_clf_index ]
    ret_clf.fit( train_x, train_y )
    
    if ( verbose ):
        print 'Using CLF with accuracy: %10.6f' % best_clf_score
        print 'CLF params: ', ret_clf.get_params( deep=False )
    
    
    return ret_clf

excellent_pred = ['InsideSF', 'NBath', 'GarageArea','TotRmsAbvGrd']
good_pred      = ['BsmtQual','MasVnrType','Fireplaces','Neighborhood','GarageFinish',
                  'GarageType','LotFrontage', 'OutsideSF', 'LotArea']
fair_pred      = ['FireplaceQu','BsmtFinType1','MSSubClass','Condition','BldgType','SaleType',
                  'HouseStyle','LotShape','LotConfig','Exterior2nd','Alley','PavedDrive','RoofStyle',
                  'SaleCondition','MSZoning','HeatingQC','Exterior1st','Electrical', 'ExterQual', 'KitchenQual']
poor_pred      = ['CentralAir','ExterCond','Bsmt','LandContour','LandSlope','MultiStory','KitchenAbvGr',
                  'Fence','BsmtFinType2','MiscFeature','YrSold']

def run_clean( input_df, train=False, warp=False, binary=True ):
    qual_t_df = remap_quality( input_df, binary )
    home_t_df = remap_home(    input_df, binary )
    area_t_df = remap_area(    input_df, binary )
    road_t_df = remap_road(    input_df, binary )
    land_t_df = remap_land(    input_df, binary )
    do_df     = qual_t_df.join( home_t_df ).join( area_t_df ).join( 
                                road_t_df ).join( land_t_df ).copy()
    
    do_df = normalize_homes( do_df )

    
    if ( warp ):
        ex = 3.0
        go = 2.0
        fa = 1.0
        po = 0.5
        
        do_df.ix[ :, excellent_pred ] = do_df.ix[ :, excellent_pred ] * ex
        do_df.ix[ :,      good_pred ] = do_df.ix[ :,      good_pred ] * go
        do_df.ix[ :,      poor_pred ] = do_df.ix[ :,      poor_pred ] * po

    if ( train ):
        do_df['SalePrice'] = np.log10(input_df['SalePrice'])

    
    return do_df.copy()


bin_class_array = []

# Change the classifications to multiple binary columns
def binary_classification( inp_df, ignore=None, lowerLim=2, upperLim=20 ):

    global bin_class_array
    
    bar=inp_df.copy()
    col_list = bar.columns.values
    
    if ( ignore != None ):
        if type( ignore ) is list:
            for element in ignore:
                index    = np.argwhere( col_list==element )
                col_list = np.delete(   col_list, index )
        else:
            index    = np.argwhere( col_list==ignore )
            col_list = np.delete( col_list, index )
            
    # Consider each column
    for col in col_list:

        options = sorted(bar[col].unique()) # Possible options

        # Only reformat the classification columns
        if ( len(options) > lowerLim and 
             len(options) < upperLim ):
            iterator = 0
            # Create new binary clssifier for each class column
            for item in bar[col].unique():

                new_col = col+'_'+str( item )

                bar[new_col] = 0
                bar.ix[ bar[col]==item, [new_col] ] = 1

            # Remove previous classification
            bar.drop( col, axis=1, inplace=True )
    return bar

# From the tanner kernel
def run_tan( train_df, test_df ):
    
    g_cols = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']
    b_cols = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2',
              'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']
    
    # Some specific values
    test_df.ix[666,'GarageFinish'] = test_df['GarageFinish'].mode()[0]
    test_df.ix[666,'GarageQual'  ] = test_df['GarageQual'  ].mode()[0]
    test_df.ix[666,'GarageCond'  ] = test_df['GarageCond'  ].mode()[0]
    test_df.ix[1150,'MasVnrType' ] = 'BrkFace'
    

    sp = train_df['SalePrice'].copy()
    
#    add_cols = ['NBath', 'YearSinceRemodel', 'TimeSinceSold', 'Age', 'AreaInside', 'HeatingScale', 'PartialPlan', 'NeighBin', 'HighSeason']
    new_dfs = [0,0]
    
    comb   = [train_df,test_df]
    
    run = 0
    
    for df in comb:

        new_dfs[run] = df.copy()

        # Remove pool nans
        inds = df[ df['PoolQC'].isnull() & df['PoolArea'] > 0 ].index
        if len( inds > 0 ):
            df.ix[inds,['PoolQC']] = 'Ex'
        new_dfs[run]['PoolQC'] = df['PoolQC'].fillna('None')
    
    
        # Put in garage build the home year if missing
        new_dfs[run]['YearBuilt'  ] = df['YearBuilt'  ].fillna( df['YearBuilt'].mode()[0] )
        new_dfs[run]['GarageYrBlt'] = df['GarageYrBlt'].fillna( new_dfs[0]['YearBuilt'] )
        new_dfs[run]['GarageYrBlt'] = new_dfs[run]['GarageYrBlt'].fillna( new_dfs[run]['GarageYrBlt'].mode()[0] )
        
        # Take care of garage nans
        for col in g_cols:
            if ( df[col].dtype == float ):
                new_dfs[run][col] = df[col].fillna( 0 )
            else:
                new_dfs[run][col] = df[col].fillna( 'None' )
    
        # Take care of basement nans
        for col in b_cols:
            if ( df[col].dtype == float ):
                new_dfs[run][col] = df[col].fillna( 0 )
            else:
                new_dfs[run][col] = df[col].fillna( 'None' )
    
        # Miscellaneous nans
        new_dfs[run]['KitchenQual'] = df['KitchenQual'].fillna( df['KitchenQual'].mode()[0] )
        new_dfs[run]['Electrical' ] = df['Electrical' ].fillna( df['Electrical' ].mode()[0] )
        new_dfs[run]['SaleType'   ] = df['SaleType'   ].fillna( df['SaleType'   ].mode()[0] )
        new_dfs[run]['Functional' ] = df['Functional' ].fillna( df['Functional' ].mode()[0] )
        new_dfs[run]['Utilities'  ] = df['Utilities'  ].fillna( df['Utilities'  ].mode()[0] )    
        new_dfs[run]['MSZoning'   ] = df['MSZoning'   ].fillna( df['MSZoning'   ].mode()[0] )
        new_dfs[run]['MasVnrArea' ] = df['MasVnrArea' ].fillna(       0 )
        new_dfs[run]['MasVnrType' ] = df['MasVnrType' ].fillna( 'None'  )
        new_dfs[run]['Fence'      ] = df['Fence'      ].fillna( 'None'  )
        new_dfs[run]['Alley'      ] = df['Alley'      ].fillna( 'None'  )
        new_dfs[run]['FireplaceQu'] = df['FireplaceQu'].fillna( 'None'  )
        new_dfs[run]['MiscFeature'] = df['MiscFeature'].fillna( 'None'  )
        new_dfs[run]['Exterior1st'] = df['Exterior1st'].fillna( 'Other' )
        new_dfs[run]['Exterior2nd'] = df['Exterior2nd'].fillna( 'Other' )

        new_dfs[run]['LotFrontage'] = df['LotFrontage'].fillna( df['LotFrontage'].mean() )

    
    
        # Nans and inconsistencies filled, now begin to remap
        num_df = df.ix[:, df.dtypes==float ].copy()
        num_df = num_df.join( df.ix[:, df.dtypes==int ], how='inner'  ).copy()
        cat_df = df.ix[:, df.dtypes=='O'].copy()


        # Remap to numeric values
        qual_cols = ['ExterQual', 'ExterCond', 'GarageQual', 'GarageCond', 'FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtQual']
        for col in qual_cols:
            new_dfs[run][col] = cat_df[col].replace( {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5} )

        new_dfs[run]['BsmtExposure'] = cat_df['BsmtExposure'].replace( {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4} ).fillna(0).astype(int)
        new_dfs[run]['BsmtFinType1'] = cat_df['BsmtFinType1'].replace( {'None': 1, 'Unf': 1, 'LwQ': 2,'Rec': 3, 
                                                                        'BLQ': 4, 'ALQ': 5, 'GLQ': 6} ).fillna(0).astype(int)
        new_dfs[run]['BsmtFinType2'] = cat_df['BsmtFinType2'].replace( {'None': 1, 'Unf': 1, 'LwQ': 2,'Rec': 3, 
                                                                        'BLQ': 4, 'ALQ': 5, 'GLQ': 6} ).fillna(0).astype(int)
        new_dfs[run]['Functional'  ] = cat_df['Functional'  ].replace( {'None': 0, 'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 
                                                                  'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8} ).fillna(0).astype(int)
        new_dfs[run]['GarageFinish'] = cat_df['GarageFinish'].replace( {'None': 0,'Unf': 0, 'RFn': 0, 'Fin': 1} ).fillna(0).astype(int)
        new_dfs[run]['Fence'       ] = cat_df['Fence'       ].replace( {'None': 1, 'MnWw': 1, 'GdWo': 1, 
                                                                        'MnPrv': 2, 'GdPrv': 4} ).fillna(0).astype(int)
        new_dfs[run]['Street'      ] = cat_df['Street'      ].replace( {'Pave': 0, 'Grvl':1 } )
        new_dfs[run]['Alley'       ] = cat_df['Alley'       ].replace( {'None': 0, 'Pave':1, 'Grvl':2} ).fillna(0)
        new_dfs[run]['Foundation'  ] = cat_df['Foundation'  ].replace( {'Wood': 0, 'BrkTil':0, 'CBlock':1, 'PConc':1, 'Slab':1, 'Stone':2} )       
        new_dfs[run]['LotConfig'   ] = cat_df['LotConfig'   ].replace( {'Corner':0,'CulDSac':1,'FR2':2,'FR3':3,'Inside':4} )       
        new_dfs[run]['GarageYrBlt' ] = num_df['GarageYrBlt' ].replace( {0:num_df['GarageYrBlt' ].mode()[0], 
                                                                        0.0:num_df['GarageYrBlt' ].mode()[0]} )
        
        manual_cols = ['BsmtExposure','BsmtFinType1','BsmtFinType2','Functional','GarageFinish','Fence','NewerDwelling']
        binarize    = ['LotShape','LandContour','LandSlope','Electrical','GarageType','PavedDrive','MiscFeature']
        binarize2   = ['2ndFlrSF', 'MasVnrArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        
        # Alter pretty binary categories so just regular/irregular, 0 reg 1 irreg
        new_dfs[run]['LotShape']    = (cat_df['LotShape'   ] == 'Reg'   ) * 1
        new_dfs[run]['LandContour'] = (cat_df['LandContour'] == 'Lvl'   ) * 1
        new_dfs[run]['LandSlope'  ] = (cat_df['LandSlope'  ] == 'Gtl'   ) * 1
        new_dfs[run]['Electrical' ] = (cat_df['Electrical' ] == 'SBrkr' ) * 1
        new_dfs[run]['GarageType' ] = (cat_df['GarageType' ] == 'Detchd') * 1
        new_dfs[run]['PavedDrive' ] = (cat_df['PavedDrive' ] == 'Y'     ) * 1
        new_dfs[run]['MiscFeature'] = (cat_df['MiscFeature'] == 'Shed'  ) * 1    


        new_dfs[run]['NewerDwelling'] =  cat_df['MSZoning'    ].fillna( cat_df['MSZoning'].mode()[0] ).replace( 
                                                                    {'RL':0, 'RM':1, 'C (all)':2, 'FV':3, 'RH':4} ).astype(int)
        
        new_dfs[run]['HasWoodDeck']   = (num_df['WoodDeckSF'  ] > 0) * 1
        new_dfs[run]['Has2ndFlr']     = (num_df['2ndFlrSF'    ] > 0) * 1
        new_dfs[run]['HasMasVnr']     = (num_df['MasVnrArea'  ] > 0) * 1
        new_dfs[run]['Remodeled']     = (num_df['YearBuilt'   ] != num_df['YearRemodAdd']) * 1
        new_dfs[run]['RecentRemodel'] = (num_df['YearRemodAdd'] >= num_df['YrSold'      ]) * 1
        new_dfs[run]['NewHouse']      = (num_df['YearBuilt'   ] == num_df['YrSold'      ]) * 1

        new_dfs[run] = new_dfs[run].drop( ['LotShape','LandContour'], axis=1 )
        
        for col in binarize2:
            new_dfs[run]['Has'+col] = (num_df[col] != 0) * 1
            
        new_dfs[run]['HighSeason'] = ((num_df['MoSold'] >= 5) & (num_df['MoSold'] <= 7) ) * 1
        
        new_dfs[run]['NeighBin'] = cat_df['Neighborhood'].replace( {'MeadowV': 0, 'IDOTRR' : 1, 'Sawyer' : 1, 'BrDale' : 1, 'OldTown': 1, 
                                                                    'Edwards': 1, 
                                                              'BrkSide': 1, 'Blueste': 1, 'SWISU'  : 2, 'NAmes'  : 2, 'NPkVill': 2, 'Mitchel': 2,
                                                              'SawyerW': 2, 'Gilbert': 2, 'NWAmes' : 2, 'Blmngtn': 2, 'CollgCr': 2, 'ClearCr': 3, 
                                                              'Crawfor': 3, 'Veenker': 3, 'Somerst': 3, 'Timber' : 3, 'StoneBr': 4, 'NoRidge': 4, 
                                                              'NridgHt': 4} )
        
        new_dfs[run]['PartialPlan' ] = (cat_df['SaleCondition'] == 'Partial') * 1

        new_dfs[run]['HeatingScale'] = cat_df['HeatingQC'].replace( {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4} )
        new_dfs[run]['AreaInside'  ] = num_df['1stFlrSF'] + num_df['2ndFlrSF']
        
        # Make some new classes
        new_dfs[run]['Age']              =             2010 - num_df['YearBuilt']
        new_dfs[run]['TimeSinceSold']    =             2010 - num_df['YrSold']
        new_dfs[run]['YearSinceRemodel'] = num_df['YrSold'] - num_df['YearRemodAdd']
        
        
        new_dfs[run]['NBath'] = num_df['BsmtFullBath'].fillna(0) + num_df['BsmtHalfBath'].fillna(0)*0.5 + num_df['FullBath'].fillna(0) + num_df['HalfBath'].fillna(0)*0.5
                
        run = run + 1

    
    # Set up some binary classifications
    # Already binary
    binary_class = ['GarageFinish','LotShape','LandContour','LandSlope','Electrical','GarageType','PavedDrive',
                    'MiscFeature','HasWoodDeck','Has2ndFlr','HasMasVnr','Remodeled','RecentRemodel','NewHouse',
                    'Has2ndFlrSF','HasMasVnrArea','HasWoodDeckSF','HasOpenPorchSF','HasEnclosedPorch',
                    'Has3SsnPorch','HasScreenPorch','HighSeason','PartialPlan']
    
    to_bin_class = ['BsmtExposure','BsmtFinType1','BsmtFinType2','Functional','Fence','NewerDwelling','NeighBin','HeatingScale',
                    'LotConfig','Alley','Foundation','MSSubClass']
    
    drop_list = ['Id','MSZoning','Utilities','Condition1','Condition2','HeatingQC','ExterQual', 'ExterCond', 'GarageQual', 
                 'GarageCond', 'FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtQual','RoofStyle','RoofMatl',
                 'Exterior1st','Exterior2nd','MasVnrType','BsmtCond','Heating','CentralAir','PoolQC',
                 'SaleType','SaleCondition','LotConfig','Foundation','Alley',
                 #ARE WE SURE ^ ?
                 'BldgType','HouseStyle','Neighborhood',
                 #Bathrooms
                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',
                 'BsmtExposure','BsmtFinType1','BsmtFinType2','Functional','Fence','NewerDwelling','NeighBin','HeatingScale',
                 # From continuous
                 '3SsnPorch','LowQualFinSF','MSSubClass', 'PoolArea','MiscVal','BsmtFinSF2','ScreenPorch','MoSold'
                ]
    
    big_df = new_dfs[0].append( new_dfs[1], ignore_index=True).copy()
    
    big_df['GarageYrBlt'] = big_df['GarageYrBlt'].fillna( big_df['YearBuilt'] )
    
    big_df = big_df.join( binary_classification( big_df.ix[:,to_bin_class] ), how='inner' )
    big_df = big_df.drop( drop_list, axis = 1 )

    nn = train_df.shape[0]
    
    train_df = big_df[:nn].copy()
    test_df  = big_df[nn:].copy()

    train_df['SalePrice'] = sp
    
    return train_df, test_df
    
    
# From the tanner kernel
def norm_tan( train_df, test_df ):
    
    #none
    lin_norm = ['Age', 'BedroomAbvGr', 'Fireplaces', 'GarageCars', 'GarageYrBlt', 'KitchenAbvGr', 'NBath', 'LotArea', 
                'OverallCond', 'OverallQual', 'TimeSinceSold', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'YearSinceRemodel', 'YrSold']
    #l +1
    log_norm = ['1stFlrSF','AreaInside','GrLivArea','LotFrontage']
    log_norm_fix_lower_lim = ['2ndFlrSF','BsmtFinSF1','BsmtUnfSF','EnclosedPorch','GarageArea','MasVnrArea',
                              'OpenPorchSF','TotalBsmtSF','WoodDeckSF']
    
    
    comb_df = train_df.drop( 'SalePrice', axis=1).append( test_df, ignore_index=True).copy()

    # List containing continuous variables
    comb_df.ix[:,log_norm              ] = np.log10( comb_df[log_norm              ] + 1 )
    comb_df.ix[:,log_norm_fix_lower_lim] = np.log10( comb_df[log_norm_fix_lower_lim] + 1 )
    
    
    # All log_norm do sigma normalization
    # '2ndFlrSF'      2.3-3.2
    # 'BsmtFinSF1'    1.2-3.4
    # 'BsmtUnfSF'     1.6-3.3
    # 'EnclosedPorch' 1.2-2.8
    # 'GarageArea'    2.1-3.1
    # 'MasVnrArea'    1.2-3.1
    # 'OpenPorchSF'   1.0-2.6
    # 'TotalBsmtSF'   2.2-3.5
    # 'WoodDeckSF'    1.1-3.1
    lower_norm_lim = [2.3,1.2,1.6,1.3,2.1,1.2,1.0,2.3,1.1]
    upper_norm_lim = [3.2,3.4,3.3,2.8,3.1,3.0,2.6,3.4,3.0]

    for col in lin_norm:
        comb_df[col] = normalize_column( comb_df, col )
    
    for col in log_norm:
        comb_df[col] = normalize_column_sigma( comb_df, col, n_sigma=3.0 )

    for i in range( 0, len(log_norm_fix_lower_lim)):
        col          = log_norm_fix_lower_lim[i]
        comb_df[col] = normalize_column( comb_df, col, maxVal = upper_norm_lim[i], minVal = lower_norm_lim[i] )
    
    
    ret_train = comb_df[:train_df.shape[0]].copy()
    ret_test  = comb_df[train_df.shape[0]:].copy()
    
    
    ret_train['SalePrice'] = np.log10(train_df['SalePrice']).copy()
    
    return ret_train, ret_test
    
    