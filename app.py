from pycaret.regression import *
import pandas as pd 
import numpy as np
import streamlit as st
from PIL import Image
import os

class StreamlitApp:
    
    def __init__(self):
        self.model = load_model('model/final_model_house_prices') 
        self.save_fn = 'path.csv'     
        
    def predict(self, input_data): 
        return predict_model(self.model, data=input_data)
    
    def store_prediction(self, output_df): 
        if os.path.exists(self.save_fn):
            save_df = pd.read_csv(self.save_fn)
            save_df = save_df.append(output_df, ignore_index=True)
            save_df.to_csv(self.save_fn, index=False)
            
        else: 
            output_df.to_csv(self.save_fn, index=False)  
            
    
    def run(self):
        image = Image.open('assets/hus.jpg')
        st.image(image, use_column_width=False)
    
    
        add_selectbox = st.sidebar.selectbox('How would you like to predict?', ('Online', 'Batch'))  
        st.sidebar.info('This app is created to predict houseprices' )
        st.sidebar.success('DAT158')
        st.title('Housing Prices')
        
       
        if add_selectbox == 'Online':
            OverallQual = st.number_input('OverallQual', min_value=1, max_value=10, value=5)
            MSZoning = st.selectbox('MSZoning', ['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP','RM'])
            GrLivArea = st.number_input('GrLivArea', min_value=300, max_value=6000, value=1500)
            OverallCond = st.number_input('OverallCond', min_value=1, max_value=9, value=5)           
            GarageType = st.selectbox('GarageType', ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'NA'])
            LandSlope = st.selectbox('LandSlope', ['Gtl', 'Mod', 'Sev'])
            FullBath = st.number_input('FullBath', min_value=0, max_value=5, value=0) 
            Neighborhood = st.selectbox('Neighborhood', ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 
                                                         'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 
                                                         'MeadowV', 'Mitchel', 'Names','NoRidge', 'NPkVill', 
                                                         'NridgHt', 'NWAmes', 'OldTown','SWISU', 'Sawyer', 
                                                         'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker', 'Up', 'Down'])
            Functional = st.selectbox('Functional', ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'])
            GarageCars =st.number_input('GarageCars', min_value=0.0, max_value=10.0, value=1.0)
            GarageArea = st.number_input('GarageArea', min_value=0.0, max_value=2000.0, value=1000.0)
            TotRmsAbvGrd = st.number_input('TotRmsAbvGrd', min_value=0.0, max_value=30.0, value=6.0)
            Fireplaces = st.number_input('Fireplaces', min_value=0.0, max_value=5.0, value=1.0)
            YearBuilt = st.number_input('YearBuilt', min_value=1800.0, max_value=2021.0, value=1980.0)
            PoolArea = st.number_input('PoolArea', min_value=0.0, max_value=1000.0, value=500.0)
            PoolQC=st.selectbox('PoolQC', ['Gd', 'TA', 'Fa', 'NA'])
            Fence = st.selectbox('Fence', ['MnPrv', 'GdWo', 'MnWw', 'NA'])
            Utilities=st.selectbox('Utilities', ['AllPub','NoSewr', 'NoSeWa', 'ELO'])
            MSSubClass=st.selectbox('MSSubClass', [20,30,40,45,50,60,70,75,80,85,90,120,150,160,180,190])
            LotFrontage=st.number_input('LotFrontage', min_value=0.0, max_value=400.0, value=100.0)
            LotArea=st.number_input('LotArea', min_value=0.0, max_value=30000.0, value=10000.0)
            Street=st.selectbox('Street', ['Grvl','Pave'])
            Alley=st.selectbox('Alley', ['Grvl', 'Pave', 'NA'])
            LotShape=st.selectbox('LotShape', ['Reg', 'IR1', 'IR2', 'IR3'])
            LandContour=st.selectbox('LandContour', ['Lvl', 'Bnk', 'HLS', 'Low'])
            LotConfig=st.selectbox('LotConfig', ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'])
            Condition1=st.selectbox('Condition1', ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'])
            Condition2=st.selectbox('Condition2', ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'])
            BldgType=st.selectbox('BldgType', ['1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI'])
            HouseStyle=st.selectbox('HouseStyle', ['1Story', '1 .5Fin', '1 .5Unf', '2Story', '2 .5Fin', '2 .5Unf', 'SFoyer', 'SLvl'])
            YearRemodAdd=st.number_input('YearRemodAdd', min_value=1800.0, max_value=2021.0, value=1980.0)
            RoofStyle=st.selectbox('RoofStyle', ['Flat', 'Gabel', 'Gambrel', 'Hip', 'Mansard', 'Shed'])
            RoofMatl=st.selectbox('RoofMatl', ['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'])
            Exterior1st=st.selectbox('Exterior1st', ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco','VinylSd','Wd Sdng', 'WdShing'])
            Exterior2nd=st.selectbox('Exterior2nd', ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco','VinylSd','Wd Sdng', 'WdShing'])
            MasVnrType = st.selectbox('MasVnrType', ['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone'])
            MasVnrArea=st.number_input('MasVnrArea', min_value=0.0, value=100.0)
            ExterQual=st.selectbox('ExterQual', ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
            ExterCond=st.selectbox('ExterCond', ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
            Foundation=st.selectbox('Foundation', ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'])
            BsmtQual=st.selectbox('BsmtQual', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
            BsmtCond=st.selectbox('BsmtCond', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
            BsmtExposure=st.selectbox('BsmtExposure', ['Gd', 'Av', 'Mn', 'No', 'NA'])
            BsmtFinType1=st.selectbox('BsmtFinType1', ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'])
            BsmtFinSF1 = st.number_input('BsmtFinSF1', min_value=0.0, max_value=6000.0, value=500.0)
            BsmtFinType2 = st.selectbox('BsmtFinType2', ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'])
            BsmtFinSF2 = st.number_input('BsmtFinSF2', min_value=0.0, max_value=6000.0, value=500.0)
            BsmtUnfSF =st.number_input('BsmtUnfSF', min_value=0.0, max_value=3000.0, value=500.0)
            TotalBsmtSF = st.number_input('TotalBsmtSF', min_value=0.0, max_value=7000.0, value=1000.0)
            Heating = st.selectbox('Heating', ['Floor','GasA','GasW ','Grav','OthW','Wall'])
            HeatingQC = st.selectbox('HeatingQC', ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
            SaleCondition=st.selectbox('SaleCondition', ['Normal', 'Abnormal', 'AdjLand', 'Alloca', 'Family', 'Partial'])
            SaleType=st.selectbox('SaleType', ['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth'])
            YrSold=st.number_input('YrSold', min_value=1950, max_value=2050, value=2010)
            MoSold=st.selectbox('MoSold', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            MiscVal=st.number_input('MiscVal', min_value=0, max_value=15500, value=5000)
            MiscFeature=st.selectbox('MiscFeature', ['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'NA'])
            ScreenPorch=st.number_input('ScreenPorch', min_value=0, max_value=1000, value=250)
            SsnPorch=st.number_input('3SsnPorch', min_value=0, max_value=1000, value=250)
            EnclosedPorch=st.number_input('EnclosedPorch', min_value=0, max_value=1000, value=250)
            OpenPorchSF=st.number_input('OpenPorchSF', min_value=0, max_value=1000, value=250)
            WoodDeckSF=st.number_input('WoodDeckSF', min_value=0, max_value=1000, value=250)
            PavedDrive=st.selectbox('PavedDrive', ['Y', 'P', 'N'])
            GarageCond=st.selectbox('GarageCond', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
            GarageQual=st.selectbox('GarageQual', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
            GarageFinish=st.selectbox('GarageFinish', ['Fin', 'RFn', 'Unf', 'NA'])
            GarageYrBlt=st.number_input('GarageYrBlt', min_value=1900, max_value=2050, value=1980)
            FireplaceQu=st.selectbox('FireplaceQu', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
            KitchenQual=st.selectbox('KitchenQual', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
            KitchenAbvGr=st.number_input('KitchenAbvGr', min_value=0, max_value=10, value=2)
            BedroomAbvGr=st.number_input('BedroomAbvGr', min_value=0, max_value=20, value=4)
            HalfBath=st.number_input('HalfBath', min_value=0, max_value=10, value=2)
            BsmtHalfBath=st.number_input('BsmtHalfBath', min_value=0, max_value=10, value=1)
            BsmtFullBath=st.number_input('BsmtFullBath', min_value=0, max_value=10, value=1)
            LowQualFinSF=st.number_input('LowQualFinSF', min_value=0, max_value=1000, value=5)
            TwondFlrSF=st.number_input('2ndFlrSF', min_value=0, max_value=3000, value=300)
            OnestFlrSF=st.number_input('1stFlrSF', min_value=100, max_value=6000, value=400)
            Electrical=st.selectbox('Electrical', ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'])
            CentralAir=st.selectbox('CentralAir', ['N', 'Y'])
         
            
            
            output=''
            input_dict = {'OverallQual':OverallQual, 'MSZoning':MSZoning, 'GrLivArea':GrLivArea, 'OverallCond':OverallCond, 
                          'GarageType':GarageType, 'LandSlope':LandSlope, 'FullBath':FullBath, 'Neighborhood':Neighborhood, 
                          'Functional':Functional, 'MSSubClass':MSSubClass, 'LotFrontage':LotFrontage, 'LotArea':LotArea, 'Street':Street,'Alley':Alley,
                          'LotShape':LotShape,'LandContour':LandContour,'Utilities':Utilities,'LotConfig':LotConfig,'Condition1': Condition1,
                          'Condition2': Condition2,'BldgType':BldgType,'HouseStyle':HouseStyle,'YearBuilt':YearBuilt,'YearRemodAdd':YearRemodAdd,'RoofStyle':RoofStyle,
                          'RoofMatl':RoofMatl,'Exterior1st':Exterior1st ,'Exterior2nd':Exterior2nd,'MasVnrType':MasVnrType,'MasVnrArea':MasVnrArea,'ExterQual':ExterQual,
                          'ExterCond':ExterCond,'Foundation':Foundation,'BsmtQual':BsmtQual,'BsmtCond':BsmtCond,'BsmtExposure':BsmtExposure,'BsmtFinType1':BsmtFinType1,
                          'BsmtFinSF1':BsmtFinSF1,'BsmtFinType2':BsmtFinType2,'BsmtFinSF2':BsmtFinSF2,'BsmtUnfSF':BsmtUnfSF,'TotalBsmtSF':TotalBsmtSF,'Heating':Heating,'HeatingQC':HeatingQC, 
                          'CentralAir':CentralAir,'Electrical':Electrical,'1stFlrSF':OnestFlrSF, '2ndFlrSF':TwondFlrSF,'LowQualFinSF':LowQualFinSF,'BsmtFullBath':BsmtFullBath,'BsmtHalfBath':BsmtHalfBath,
                          'HalfBath':HalfBath,'BedroomAbvGr':BedroomAbvGr,'KitchenAbvGr':KitchenAbvGr,'KitchenQual':KitchenQual,'TotRmsAbvGrd':TotRmsAbvGrd, 'Fireplaces':Fireplaces,'FireplaceQu':FireplaceQu,
                          'GarageYrBlt':GarageYrBlt, 'GarageFinish':GarageFinish,'GarageCars':GarageCars, 'GarageArea':GarageArea, 'GarageQual':GarageQual,'GarageCond':GarageCond,'PavedDrive':PavedDrive,
                          'WoodDeckSF':WoodDeckSF,'OpenPorchSF':OpenPorchSF,'EnclosedPorch':EnclosedPorch,'3SsnPorch':SsnPorch,'ScreenPorch':ScreenPorch,'PoolArea':PoolArea,'PoolQC':PoolQC,'Fence':Fence,
                          'MiscFeature':MiscFeature,'MiscVal':MiscVal, 'MoSold':MoSold, 'YrSold':YrSold, 'SaleType':SaleType,'SaleCondition':SaleCondition}
           
            input_df = pd.DataFrame(input_dict, index=[0])
        
            if st.button('Predict'): 
                output = self.predict(input_df)
                self.store_prediction(output)
                
               
                output = output['Label'][0]
                
            
            st.success('Predicted output: ${}'.format(output))
            
        if add_selectbox == 'Batch': 
            fn = st.file_uploader("Upload csv file for predictions") #st.file_uploader('Upload csv file for predictions, type=["csv"]')
            if fn is not None: 
                input_df = pd.read_csv(fn)
                predictions = self.predict(input_df)
                st.write(predictions)
            
sa = StreamlitApp()
sa.run()