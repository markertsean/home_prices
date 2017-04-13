import numpy  as np
import pandas as pd


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
MSZoning_map = { 'A':0, 'C': 1, 'C (all)':1, 'FV':2, 'I':3, 'RH':4, 'RL':5, 'RP':6, 'RM':7 } # MSZoning
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
exter_map    = { 'AsbShng':0, 'AsphShn':1, 'BrkComm':2,	'Brk Cmn':2, 'BrkFace':3, 'CBlock':4, 'CemntBd':5, 'CmentBd':5, 'HdBoard':6, 'ImStucc':7,	
                 'MetalSd':8, 'Other':9, 'Plywood':10,  'PreCast':11,  'Stone':12, 'Stucco':13, 'VinylSd':14, 'Wd Sdng':15, 
                 'WdShing':16, 'Wd Shng':16 }
masn_map     = { 'None':0, 'BrkCmn':1, 'BrkFace':2, 'CBlock':3, 'Stone':4 }
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
feat_map     = { 'NA':0, 'NaN':0, 'TenC':1, 'Shed':2, 'Elev':3, 'Gar2':4, 'Othr':5 }
salType_map  = { 'WD':0, 'CWD':0, 'New':1, 'COD':2, 'VWD':3, 'Con':3, 'ConLw':3, 'ConLI':3, 'ConLD':3, 'Oth':3 }
salCond_map  = { 'Normal':0, 'Abnorml':1, 'AdjLand':0, 'Alloca':2, 'Family':3, 'Partial':4 }

bsmtquall    = { 'NA':0, 'nan':0, 'NaN':0, 'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4 }
grgquall     = { 'NA':0, 'nan':0, 'NaN':0, 'Po':1, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':3 }
externcond   = { 'NA':0, 'nan':0, 'NaN':0, 'Po':0, 'Fa':0, 'TA':1, 'Gd':2, 'Ex':3 }


# Remap all the quality stuff to integers
def remap_quality( quality_df ):
    foo = quality_df[ ['ExterQual', 'FireplaceQu', 'KitchenQual', 'GarageQual'] ].replace( qual_map ).fillna(0).copy()

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
    
    foo['BsmtQual']    = quality_df['BsmtQual']    .replace(   bsmtquall ).fillna( 0 )
    foo['ExterCond']   = quality_df['ExterCond']   .replace(  externcond )
    foo['HeatingQC']   = quality_df['HeatingQC']   .replace(  externcond )
    #foo['RoofStyle']   = quality_df['RoofStyle']   .replace( roofSty_map )
    foo['RoofMatl']    = quality_df['RoofMatl']    .replace( roofMat_map )
    foo['Exterior']    = quality_df['Exterior1st'] .replace(   exter_map )
    #foo['Exterior2nd'] = quality_df['Exterior2nd'] .replace(   exter_map )
    foo['MasVnrType']  = quality_df['MasVnrType']  .replace(    masn_map ).fillna( 0 )
    foo['Bsmt']        = quality_df['BsmtExposure'].replace(    expo_map ).fillna( 0 )
    #foo['BsmtFinType'] = quality_df['BsmtFinType1'].replace(  basFin_map ).fillna( 0 )
    #foo['BsmtFinType2']= quality_df['BsmtFinType2'].replace(  basFin_map ).fillna( 0 )
    foo['Electrical']  = quality_df['Electrical']  .replace(    elec_map ).fillna( 0 )
    foo['GarageFinish']= quality_df['GarageFinish'].replace(    garF_map ).fillna( 0 )
    foo['Fence']       = quality_df['Fence']       .replace(   fence_map ).fillna( 0 )
    
    return foo[[ 'Exterior', 'ExterQual', 'MasVnrType', 'RoofMatl', 'Fence', 
                 'Bsmt', 'BsmtQual', 'GarageQual', 'GarageFinish',
                 'FireplaceQu', 'HeatingQC', 'KitchenQual', 'Electrical']].astype(int)

# Remap all the home list stuff to integers
def remap_home( quality_df ):
    bigList = [ 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageArea' ]

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
    
    #foo['InsideSF']   = quality_df['1stFlrSF']     + quality_df['2ndFlrSF']
    foo['InsideSF']   = quality_df['GrLivArea']
    foo['OutsideSF']  = quality_df['WoodDeckSF']   + quality_df['OpenPorchSF'] +\
                        quality_df['EnclosedPorch']+ quality_df['3SsnPorch']   +\
                        quality_df['ScreenPorch']
        
    foo['MultiStory'] =(quality_df['2ndFlrSF']>0 ).astype(int)
    
    foo['NBath']      = quality_df['BsmtFullBath'] + quality_df['BsmtHalfBath'] +\
                        quality_df['FullBath']     + quality_df['HalfBath']
    
    #foo['Utilities']  = quality_df['Utilities']  .replace(  util_map )
    foo['BldgType']   = quality_df['BldgType']   .replace(  bldg_map )
    foo['HouseStyle'] = quality_df['HouseStyle'] .replace( style_map )
    #foo['Heating']    = quality_df['Heating']    .replace(  heat_map )
    foo['CentralAir'] = quality_df['CentralAir'] .replace( yesno_map )
    foo['GarageType'] = quality_df['GarageType'] .replace(  garT_map ).fillna(0)
    #foo['MiscFeature']= quality_df['MiscFeature'].replace(  feat_map ).fillna( 0 )
    
    foo[ foo['KitchenAbvGr']==0 ] = 1
    
    return foo[[ 'BldgType', 'HouseStyle', 'MultiStory', 'InsideSF', 'OutsideSF',
                 'NBath', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageType', 'GarageArea', 'Fireplaces', 'CentralAir' ]]

def remap_area( inp_df ):
    bigList = [ 'YrSold', 'MSSubClass', 'MSZoning', 'Neighborhood', 'Condition1', 'Condition2', #'MoSold', 
                'SaleType', 'SaleCondition' ]

    foo = inp_df[ ['YrSold','MSSubClass'] ].copy()
    foo['MSSubClass']    = inp_df['MSSubClass']   .replace(    MSSub_map )
    foo['MSZoning']      = inp_df['MSZoning']     .replace( MSZoning_map )
    foo['Neighborhood']  = inp_df['Neighborhood'] .replace(    neigh_map )
    foo['Condition1']    = inp_df['Condition1']   .replace(     cond_map )
    foo['Condition2']    = inp_df['Condition2']   .replace(     cond_map )
    foo['SaleType']      = inp_df['SaleType']     .replace(  salType_map )
    foo['SaleCondition'] = inp_df['SaleCondition'].replace(  salCond_map )
   
    foo['Condition']     = foo['Condition1']


    return foo.drop( ['Condition1','Condition2'], axis=1 )

def remap_road( inp_df ):
    bigList = [ 'LotFrontage', 'Alley', 'PavedDrive' ]

    foo = inp_df[ bigList ].replace( road_map ).copy()
    foo['Alley'     ] = inp_df['Alley'     ].replace(  road_map )
#    foo['Street'    ] = inp_df['Street'    ].replace(  road_map )
    foo['PavedDrive'] = inp_df['PavedDrive'].replace( yesno_map )
   
    return foo.fillna( 0 )

def remap_land( inp_df ):
    bigList = [ 'LotArea', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope' ]

    foo = inp_df[ bigList ].copy()
    foo['junk'] = foo['LotShape']
    foo['LotShape'] = foo['LotShape'].replace( shape_map )
    foo['LandContour'] = foo['LandContour'].replace( contour_map )
    foo['LotConfig']   = foo['LotConfig']  .replace(  config_map )  
    foo['LandSlope']   = foo['LandSlope']  .replace(   slope_map )
    
    foo['LotArea'  ]   = np.log10( foo['LotArea'] )
    
    return foo[bigList]