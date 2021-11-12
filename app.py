from pycaret.regression import *
import pandas as pd 
import numpy as np
import streamlit as st
from PIL import Image
import os

class StreamlitApp:
    
    def __init__(self):
        self.model = load_model('models/final_model_house_prices') 
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
        st.sidebar.info('This app is created to predict house prices in Ames, Iowa. To try with other training sets, chooce batch above' )
        st.sidebar.success('DAT158 - ML Oblig 2, by Karl-Magnus and Bernt Otto')
        st.sidebar.success('Link to kaggle where all the abbreviations are explained: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data')
        st.title('Housing Prices')
        
       
        if add_selectbox == 'Online':
            MSSubClass=st.selectbox('The building class', ['1-STORY 1946 & NEWER ALL STYLES', '1-STORY 1945 & OLDER', '1-STORY W/FINISHED ATTIC ALL AGES', 
                                                           '1-1/2 STORY - UNFINISHED ALL AGES', '1-1/2 STORY FINISHED ALL AGES', '2-STORY 1946 & NEWER',
                                                           '2-STORY 1945 & OLDER', '2-1/2 STORY ALL AGES', 'SPLIT OR MULTI-LEVEL', 'SPLIT FOYER', 'DUPLEX - ALL STYLES AND AGES',
                                                           '1-STORY PUD (Planned Unit Development) - 1946 & NEWER', '1-1/2 STORY PUD - ALL AGES', 
                                                           '2-STORY PUD - 1946 & NEWER','PUD - MULTILEVEL - INCL SPLIT LEV/FOYER', '2 FAMILY CONVERSION - ALL STYLES AND AGES'])
            GrLivArea = st.number_input('Above grade (ground) living area square feet', min_value=300, max_value=6000, value=1500)
            OverallCond = st.number_input('Overall condition rating', min_value=1, max_value=9, value=5)
            OverallQual = st.number_input('Overall material and finish quality', min_value=1, max_value=10, value=5)
            MSZoning = st.selectbox('The general zoning classification', ['Agriculture', 'Commercial', 'Floating Village Residential', 'Idustrial', 'Residential High Density', 
                                                                          'Residential Low Density', 'Residential High Density Park','Residential Medium Density'])
            LotFrontage=st.number_input('Linear feet of street connected to property', min_value=0.0, max_value=400.0, value=100.0)
            LotArea=st.number_input('Lot size in square feet', min_value=0.0, max_value=30000.0, value=10000.0)
            Street=st.selectbox('Type of road access', ['Gravel','Paved'])
            Alley=st.selectbox('Type of alley access', ['Gravel', 'Paved', 'No alley access'])
            LotShape=st.selectbox('General shape of property', ['Regular', 'Slightly irregular', 'Moderately irregular', 'Irregular'])
            LandContour=st.selectbox('Flatness of the property', ['Near flat/level', 'Banked - Quick and significant rise from street grade to building', 
                                                                  'Hillside - Significant slope from side to side', 'Depression'])
            Utilities=st.selectbox('Type of utilities available', ['All public Utilities (E,G,W,& S)','Electricity, Gas, and Water (Septic Tank)', 
                                                                   'Electricity and Gas Only', 'Electricity only'])
            LotConfig=st.selectbox('Lot configuration', ['Inside lot', 'Corner lot', 'Cul-De-Sac', 'Frontage on 2 sides of property', 'Frontage on 3 sides of property'])     
            LandSlope = st.selectbox('Slope of property', ['Gentle slope', 'Moderate slope', 'Severe slope']) 
            Neighborhood = st.selectbox('Physical locations within Ames city limits', ['Bloomington Heights', 'Bluestem', 'Briardale', 'Brookside', 'Clear Creek', 
                                                         'College Creek', 'Crawford', 'Edwards', 'Gilbert', 'Iowa DOT and Rail road', 
                                                         'Meadow Village', 'Mitchell', 'North Ames','Northridge', 'Northpark Villa', 
                                                         'Northpark Heights', 'Northwest Ames', 'Old Town','South & West of Iowa State University', 'Sawyer', 
                                                         'Sawyer West', 'Somerset', 'Stone Brook', 'Timberland', 'Veenker', 'Up', 'Down'])
            Condition1=st.selectbox('Proximity to main road or railroad', ['Adjacent to arterial street', 'Adjacent to feeder street', 'Normal', 
                                                                           'Within 200 of North-South Railroad', 'Adjacent to North-South Railroad', 
                                                                           'Near positive off-site feature--park, greenbelt, etc.', 
                                                                           'Adjacent to postive off-site feature', 'Within 200 of East-West Railroad', 
                                                                           'Adjacent to East-West Railroad'])
            Condition2=st.selectbox('Proximity to main road or railroad (if a second is present)', ['Adjacent to arterial street', 'Adjacent to feeder street', 'Normal', 
                                                                           'Within 200 of North-South Railroad', 'Adjacent to North-South Railroad', 
                                                                           'Near positive off-site feature--park, greenbelt, etc.', 
                                                                           'Adjacent to postive off-site feature', 'Within 200 of East-West Railroad', 
                                                                           'Adjacent to East-West Railroad'])
            BldgType=st.selectbox('Type of dwelling', ['Single-family Detached', 'Two-family Conversion; originally built as one-family dwelling', 
                                                       'Duplex', 'Townhouse End Unit', 'Townhouse Inside Unit'])
            HouseStyle=st.selectbox('Style of dwelling', ['One Story', 'One and one-half story: 2nd level finished', 'One and one-half story: 2nd level unfinished', 
                                                          'Two Story', 'Two and one-half story: 2nd level finished', 'Two and one-half story: 2nd level unfinished', 
                                                          'Split Foyer', 'Split Level'])
            YearBuilt = st.number_input('Original construction date', min_value=1800, max_value=2021, value=1980)
            YearRemodAdd=st.number_input('Remodel date (same as construction date if no remodeling or additions)', min_value=1800, max_value=2021, value=1980)
            RoofStyle=st.selectbox('Type of roof', ['Flat', 'Gable', 'Gabrel(Barn)', 'Hip', 'Mansard', 'Shed'])
            RoofMatl=st.selectbox('Roof material', ['Clay or Tile', 'Standard (Composite) Shingle', 'Membrane', 'Metal', 'Roll', 'Gravel & Tar', 'Wood Shakes', 'Wood Shingles'])
            Exterior1st=st.selectbox('Exterior covering on house', ['Asbestos Shingles', 'Asphalt Shingles', 'Brick Common', 'Brick Face', 'Cinder Block', 'Cement board', 
                                                                    'Hard Board', 'Imitation Stucco', 'Metal Siding', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 
                                                                    'Vinyl Siding','Wood Siding', 'Wood Shingles'])
            Exterior2nd=st.selectbox('Exterior covering on house (if more than one material)', ['Asbestos Shingles', 'Asphalt Shingles', 'Brick Common', 'Brick Face', 'Cinder Block', 'Cement board', 
                                                                    'Hard Board', 'Imitation Stucco', 'Metal Siding', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 
                                                                    'Vinyl Siding','Wood Siding', 'Wood Shingles'])
            MasVnrType = st.selectbox('Masonry veneer type', ['Brick Common', 'Brick Face', 'Cinder Block', 'None', 'Stone'])
            MasVnrArea=st.number_input('Masonry veneer area in square feet', min_value=0.0, value=100.0)
            ExterQual=st.selectbox('Exterior material quality', ['Excellent', 'Good', 'Average/Typical', 'Fair', 'Poor'])
            ExterCond=st.selectbox('Present condition of the material on the exterior', ['Excellent', 'Good', 'Average/Typical', 'Fair', 'Poor'])
            Foundation=st.selectbox('Type of foundation', ['Brick and Tile', 'Cinder Block', 'Poured Concrete', 'Slab', 'Stone', 'Wood'])
            BsmtQual=st.selectbox('Height of the basement(Inches)', ['Excellent(100+)', 'Good(90-99)', 'Average/Typical(80-89)', 'Fair(70-79)', 'Poor(<70)', 
                                                                                'No Basement'])
            BsmtCond=st.selectbox('General condition of the basement', ['Excellent', 'Good', 'Typical - slight dampness allowed', 
                                                                                'Fair - dampness or some cracking or settling', 'Poor - Severe cracking, settling, or wetness', 
                                                                                'No Basement'])
            BsmtExposure=st.selectbox('Walkout or garden level basement walls', ['Good exposure', 'Average Exposure (split levels or foyers typically score average or above)', 
                                                                                 'Mimimum Exposure', 'No Exposure', 'No basement'])
            BsmtFinType1=st.selectbox('Quality of basement finished area', ['Good Living Quarters', 'Average Living Quarters', 'Below Average Living Quarters', 
                                                                            'Average Rec Room', 'Low Quality', 'Unfinshed', 'No basement'])
            BsmtFinSF1 = st.number_input('Type 1 finished square feet', min_value=0.0, max_value=6000.0, value=500.0)
            BsmtFinType2 = st.selectbox('Quality of second finished area (if present)', ['Good Living Quarters', 'Average Living Quarters', 'Below Average Living Quarters', 
                                                                            'Average Rec Room', 'Low Quality', 'Unfinshed', 'No basement'])
            BsmtFinSF2 = st.number_input('Type 2 finished square feet', min_value=0.0, max_value=6000.0, value=500.0)
            BsmtUnfSF =st.number_input('Unfinished square feet of basement area', min_value=0.0, max_value=3000.0, value=500.0)
            TotalBsmtSF = st.number_input('Total square feet of basement area', min_value=0.0, max_value=7000.0, value=1000.0)
            Heating = st.selectbox('Type of heating', ['Floor furnace','Gas forced warm air furnace','Gas hot water or steam heat','Gravity furnace',
                                                       'Hot water or steam heat other than gas','Wall furnace'])
            HeatingQC = st.selectbox('Heating quality and condition', ['Excellent', 'Good', 'Average/typical', 'Fair', 'Poor'])
            CentralAir=st.selectbox('Central air conditioning', ['No', 'Yes'])
            Electrical=st.selectbox('Electrical system', ['Standard Circuit Breakers & Romex', 'Fuse Box over 60 AMP and all Romex wiring (Average)', 
                                                          '60 AMP Fuse Box and mostly Romex wiring (Fair)', 
                                                          '60 AMP Fuse Box and mostly knob & tube wiring (poor)', 'Mixed'])
            OnestFlrSF=st.number_input('First Floor square feet', min_value=100, max_value=6000, value=400)
            TwondFlrSF=st.number_input('Second floor square feet', min_value=0, max_value=3000, value=300)
            LowQualFinSF=st.number_input('Low quality finished square feet (all floors)', min_value=0, max_value=1000, value=5)
            FullBath = st.number_input('Full bathrooms above grade', min_value=0, max_value=5, value=0)
            HalfBath=st.number_input('Half baths above grade', min_value=0, max_value=10, value=2)
            BsmtHalfBath=st.number_input('Basement half bathrooms', min_value=0, max_value=10, value=1)
            BsmtFullBath=st.number_input('Basement full bathrooms', min_value=0, max_value=10, value=1)
            KitchenAbvGr=st.number_input('Number of kitchens', min_value=0, max_value=10, value=2)
            KitchenQual=st.selectbox('Kitchen quality', ['Excellent', 'Good', 'Average/typical', 'Fair', 'Poor'])
            TotRmsAbvGrd = st.number_input('Total rooms above grade (does not include bathrooms)', min_value=0, max_value=30, value=6)
            BedroomAbvGr=st.number_input('Number of bedrooms above basement level', min_value=0, max_value=20, value=4)
            Functional = st.selectbox('Home functionality rating', ['Typical Functionality', 'Minor Deductions 1', 'Minor Deductions 2', 'Moderate Deductions', 
                                                                    'Major Deductions 1', 'Major Deductions 2', 'Severly damaged', 'Salvage only'])
            Fireplaces = st.number_input('Number of fireplaces(0-5)', min_value=0, max_value=5, value=1)
            FireplaceQu=st.selectbox('Fireplace quality', ['Excellent - Exceptional Masonry Fireplace', 'Good - Masonry Fireplace in main level', 
                                                           'Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement', 
                                                           'Fair - Prefabricated Fireplace in basement', 
                                                           'Poor - Ben Franklin Stove', 'No fireplace'])
            GarageType = st.selectbox('Garage location', ['More than one type of garage', 'Attached to home', 'Basement garage', 
                                                          'Built-In (Garage part of house - typically has room above garage)', 'Car Port', 'Detached from home', 'No garage'])
            GarageCond=st.selectbox('Garage condition', ['Excellent', 'Good', 'Average/typical', 'Fair', 'Poor', 'No garage'])
            GarageQual=st.selectbox('Garage quality', ['Excellent', 'Good', 'Average/typical', 'Fair', 'Poor', 'No garage'])
            GarageFinish=st.selectbox('Interior finish of the garage', ['Finished', 'Rough Finished', 'Unfinished', 'No garage'])
            GarageYrBlt=st.number_input('Year garage was built', min_value=1900, max_value=2050, value=1980)
            GarageCars =st.number_input('Size of garage in car capacity', min_value=0, max_value=10, value=1)
            GarageArea = st.number_input('Size of garage in square feet', min_value=0.0, max_value=2000.0, value=1000.0)
            PavedDrive=st.selectbox('Paved driveway', ['Paved', 'Partial pavement', 'Dirt/gravel'])
            WoodDeckSF=st.number_input('Wood deck area in square feet', min_value=0, max_value=1000, value=250)
            OpenPorchSF=st.number_input('Open porch area in square feet', min_value=0, max_value=1000, value=250)
            EnclosedPorch=st.number_input('Enclosed porch area in square feet', min_value=0, max_value=1000, value=250)
            ScreenPorch=st.number_input('Screen porch area in square feet', min_value=0, max_value=1000, value=250)
            SsnPorch=st.number_input('Three season porch area in square feet', min_value=0, max_value=1000, value=250)
            PoolArea = st.number_input('Pool area in square feet', min_value=0.0, max_value=1000.0, value=500.0)
            PoolQC=st.selectbox('Pool quality', ['Excellent', 'Good', 'Average/typical', 'Fair', 'No pool'])
            Fence = st.selectbox('Fence quality', ['Good Privacy', 'Minimum Privacy', 'Good Wood', 'Minimum Wood/Wire', 'No fence'])
            MiscFeature=st.selectbox('Miscellaneous feature not covered in other categories', ['Elevator', '2nd Garage (if not described in garage section)', 
                                                                                               'Other', 'Shed(over 100 SF)', 'Tennis Court', 'None'])
            MiscVal=st.number_input('$ Value of miscellaneous feature', min_value=0, max_value=15500, value=5000)
            YrSold=st.number_input('Year Sold', min_value=1950, max_value=2050, value=2010)
            MoSold=st.selectbox('Month Sold', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            SaleType=st.selectbox('Type of sale', ['Warranty Deed - Conventional', 'Warranty Deed - Cash', 'Warranty Deed - VA Loan', 'Home just constructed and sold', 
                                                   'Court Officer Deed/Estate', 'Contract 15% Down payment regular terms', 'Contract Low Down payment and low interest', 
                                                   'Contract Low Interest', 'Contract Low Down', 'Other'])
            SaleCondition=st.selectbox('Condition of sale', ['Normal', 'Abnormal Sale -  trade, foreclosure, short sale', 
                                                             'Adjoining Land Purchase', 'Allocation - two linked properties with separate deeds, typically condo with a garage unit', 
                                                             'Sale between family members', 'Home was not completed when last assessed (associated with New Homes)'])
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
           
            
            
           
            
            
         
            
            
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
