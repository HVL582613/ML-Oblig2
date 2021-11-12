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
        st.sidebar.info('This app is created to predict house prices in Ames, Iowa' )
        st.sidebar.success('DAT158 - ML Oblig 2, by Karl-Magnus and Bernt Otto')
        st.sidebar.success('
        MSSubClass: Identifies the type of dwelling involved in the sale.	

        20	1-STORY 1946 & NEWER ALL STYLES
        30	1-STORY 1945 & OLDER
        40	1-STORY W/FINISHED ATTIC ALL AGES
        45	1-1/2 STORY - UNFINISHED ALL AGES
        50	1-1/2 STORY FINISHED ALL AGES
        60	2-STORY 1946 & NEWER
        70	2-STORY 1945 & OLDER
        75	2-1/2 STORY ALL AGES
        80	SPLIT OR MULTI-LEVEL
        85	SPLIT FOYER
        90	DUPLEX - ALL STYLES AND AGES
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
       150	1-1/2 STORY PUD - ALL AGES
       160	2-STORY PUD - 1946 & NEWER
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES

MSZoning: Identifies the general zoning classification of the sale.
		
       A	Agriculture
       C	Commercial
       FV	Floating Village Residential
       I	Industrial
       RH	Residential High Density
       RL	Residential Low Density
       RP	Residential Low Density Park 
       RM	Residential Medium Density
	
LotFrontage: Linear feet of street connected to property

LotArea: Lot size in square feet

Street: Type of road access to property

       Grvl	Gravel	
       Pave	Paved
       	
Alley: Type of alley access to property

       Grvl	Gravel
       Pave	Paved
       NA 	No alley access
		
LotShape: General shape of property

       Reg	Regular	
       IR1	Slightly irregular
       IR2	Moderately Irregular
       IR3	Irregular
       
LandContour: Flatness of the property

       Lvl	Near Flat/Level	
       Bnk	Banked - Quick and significant rise from street grade to building
       HLS	Hillside - Significant slope from side to side
       Low	Depression
		
Utilities: Type of utilities available
		
       AllPub	All public Utilities (E,G,W,& S)	
       NoSewr	Electricity, Gas, and Water (Septic Tank)
       NoSeWa	Electricity and Gas Only
       ELO	Electricity only	
	
LotConfig: Lot configuration

       Inside	Inside lot
       Corner	Corner lot
       CulDSac	Cul-de-sac
       FR2	Frontage on 2 sides of property
       FR3	Frontage on 3 sides of property
	
LandSlope: Slope of property
		
       Gtl	Gentle slope
       Mod	Moderate Slope	
       Sev	Severe Slope
	
Neighborhood: Physical locations within Ames city limits

       Blmngtn	Bloomington Heights
       Blueste	Bluestem
       BrDale	Briardale
       BrkSide	Brookside
       ClearCr	Clear Creek
       CollgCr	College Creek
       Crawfor	Crawford
       Edwards	Edwards
       Gilbert	Gilbert
       IDOTRR	Iowa DOT and Rail Road
       MeadowV	Meadow Village
       Mitchel	Mitchell
       Names	North Ames
       NoRidge	Northridge
       NPkVill	Northpark Villa
       NridgHt	Northridge Heights
       NWAmes	Northwest Ames
       OldTown	Old Town
       SWISU	South & West of Iowa State University
       Sawyer	Sawyer
       SawyerW	Sawyer West
       Somerst	Somerset
       StoneBr	Stone Brook
       Timber	Timberland
       Veenker	Veenker
			
Condition1: Proximity to various conditions
	
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
	
Condition2: Proximity to various conditions (if more than one is present)
		
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
	
BldgType: Type of dwelling
		
       1Fam	Single-family Detached	
       2FmCon	Two-family Conversion; originally built as one-family dwelling
       Duplx	Duplex
       TwnhsE	Townhouse End Unit
       TwnhsI	Townhouse Inside Unit
	
HouseStyle: Style of dwelling
	
       1Story	One story
       1.5Fin	One and one-half story: 2nd level finished
       1.5Unf	One and one-half story: 2nd level unfinished
       2Story	Two story
       2.5Fin	Two and one-half story: 2nd level finished
       2.5Unf	Two and one-half story: 2nd level unfinished
       SFoyer	Split Foyer
       SLvl	Split Level
	
OverallQual: Rates the overall material and finish of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor
	
OverallCond: Rates the overall condition of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average	
       5	Average
       4	Below Average	
       3	Fair
       2	Poor
       1	Very Poor
		
YearBuilt: Original construction date

YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)

RoofStyle: Type of roof

       Flat	Flat
       Gable	Gable
       Gambrel	Gabrel (Barn)
       Hip	Hip
       Mansard	Mansard
       Shed	Shed
		
RoofMatl: Roof material

       ClyTile	Clay or Tile
       CompShg	Standard (Composite) Shingle
       Membran	Membrane
       Metal	Metal
       Roll	Roll
       Tar&Grv	Gravel & Tar
       WdShake	Wood Shakes
       WdShngl	Wood Shingles
		
Exterior1st: Exterior covering on house

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast	
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
Exterior2nd: Exterior covering on house (if more than one material)

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
MasVnrType: Masonry veneer type

       BrkCmn	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       None	None
       Stone	Stone
	
MasVnrArea: Masonry veneer area in square feet

ExterQual: Evaluates the quality of the material on the exterior 
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
ExterCond: Evaluates the present condition of the material on the exterior
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
Foundation: Type of foundation
		
       BrkTil	Brick & Tile
       CBlock	Cinder Block
       PConc	Poured Contrete	
       Slab	Slab
       Stone	Stone
       Wood	Wood
		
BsmtQual: Evaluates the height of the basement

       Ex	Excellent (100+ inches)	
       Gd	Good (90-99 inches)
       TA	Typical (80-89 inches)
       Fa	Fair (70-79 inches)
       Po	Poor (<70 inches
       NA	No Basement
		
BsmtCond: Evaluates the general condition of the basement

       Ex	Excellent
       Gd	Good
       TA	Typical - slight dampness allowed
       Fa	Fair - dampness or some cracking or settling
       Po	Poor - Severe cracking, settling, or wetness
       NA	No Basement
	
BsmtExposure: Refers to walkout or garden level walls

       Gd	Good Exposure
       Av	Average Exposure (split levels or foyers typically score average or above)	
       Mn	Mimimum Exposure
       No	No Exposure
       NA	No Basement
	
BsmtFinType1: Rating of basement finished area

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement
		
BsmtFinSF1: Type 1 finished square feet

BsmtFinType2: Rating of basement finished area (if multiple types)

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement

BsmtFinSF2: Type 2 finished square feet

BsmtUnfSF: Unfinished square feet of basement area

TotalBsmtSF: Total square feet of basement area

Heating: Type of heating
		
       Floor	Floor Furnace
       GasA	Gas forced warm air furnace
       GasW	Gas hot water or steam heat
       Grav	Gravity furnace	
       OthW	Hot water or steam heat other than gas
       Wall	Wall furnace
		
HeatingQC: Heating quality and condition

       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
CentralAir: Central air conditioning

       N	No
       Y	Yes
		
Electrical: Electrical system

       SBrkr	Standard Circuit Breakers & Romex
       FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
       FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
       FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
       Mix	Mixed
		
1stFlrSF: First Floor square feet
 
2ndFlrSF: Second floor square feet

LowQualFinSF: Low quality finished square feet (all floors)

GrLivArea: Above grade (ground) living area square feet

BsmtFullBath: Basement full bathrooms

BsmtHalfBath: Basement half bathrooms

FullBath: Full bathrooms above grade

HalfBath: Half baths above grade

Bedroom: Bedrooms above grade (does NOT include basement bedrooms)

Kitchen: Kitchens above grade

KitchenQual: Kitchen quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       	
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

Functional: Home functionality (Assume typical unless deductions are warranted)

       Typ	Typical Functionality
       Min1	Minor Deductions 1
       Min2	Minor Deductions 2
       Mod	Moderate Deductions
       Maj1	Major Deductions 1
       Maj2	Major Deductions 2
       Sev	Severely Damaged
       Sal	Salvage only
		
Fireplaces: Number of fireplaces

FireplaceQu: Fireplace quality

       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level
       TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa	Fair - Prefabricated Fireplace in basement
       Po	Poor - Ben Franklin Stove
       NA	No Fireplace
		
GarageType: Garage location
		
       2Types	More than one type of garage
       Attchd	Attached to home
       Basment	Basement Garage
       BuiltIn	Built-In (Garage part of house - typically has room above garage)
       CarPort	Car Port
       Detchd	Detached from home
       NA	No Garage
		
GarageYrBlt: Year garage was built
		
GarageFinish: Interior finish of the garage

       Fin	Finished
       RFn	Rough Finished	
       Unf	Unfinished
       NA	No Garage
		
GarageCars: Size of garage in car capacity

GarageArea: Size of garage in square feet

GarageQual: Garage quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
		
GarageCond: Garage condition

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
		
PavedDrive: Paved driveway

       Y	Paved 
       P	Partial Pavement
       N	Dirt/Gravel
		
WoodDeckSF: Wood deck area in square feet

OpenPorchSF: Open porch area in square feet

EnclosedPorch: Enclosed porch area in square feet

3SsnPorch: Three season porch area in square feet

ScreenPorch: Screen porch area in square feet

PoolArea: Pool area in square feet

PoolQC: Pool quality
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       NA	No Pool
		
Fence: Fence quality
		
       GdPrv	Good Privacy
       MnPrv	Minimum Privacy
       GdWo	Good Wood
       MnWw	Minimum Wood/Wire
       NA	No Fence
	
MiscFeature: Miscellaneous feature not covered in other categories
		
       Elev	Elevator
       Gar2	2nd Garage (if not described in garage section)
       Othr	Other
       Shed	Shed (over 100 SF)
       TenC	Tennis Court
       NA	None
		
MiscVal: $Value of miscellaneous feature

MoSold: Month Sold (MM)

YrSold: Year Sold (YYYY)

SaleType: Type of sale
		
       WD 	Warranty Deed - Conventional
       CWD	Warranty Deed - Cash
       VWD	Warranty Deed - VA Loan
       New	Home just constructed and sold
       COD	Court Officer Deed/Estate
       Con	Contract 15% Down payment regular terms
       ConLw	Contract Low Down payment and low interest
       ConLI	Contract Low Interest
       ConLD	Contract Low Down
       Oth	Other
		
SaleCondition: Condition of sale

       Normal	Normal Sale
       Abnorml	Abnormal Sale -  trade, foreclosure, short sale
       AdjLand	Adjoining Land Purchase
       Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
       Family	Sale between family members
       Partial	Home was not completed when last assessed (associated with New Homes)')
        st.title('Housing Prices')
        
       
        if add_selectbox == 'Online':
            OverallQual = st.number_input('Overall material and finish quality', min_value=1, max_value=10, value=5)
            MSZoning = st.selectbox('The general zoning classification', ['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP','RM'])
            GrLivArea = st.number_input('Above grade (ground) living area square feet', min_value=300, max_value=6000, value=1500)
            OverallCond = st.number_input('Overall condition rating', min_value=1, max_value=9, value=5)           
            GarageType = st.selectbox('Garage location', ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'NA'])
            LandSlope = st.selectbox('Slope of property', ['Gtl', 'Mod', 'Sev'])
            FullBath = st.number_input('Full bathrooms above grade', min_value=0, max_value=5, value=0) 
            Neighborhood = st.selectbox('Physical locations within Ames city limits', ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 
                                                         'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 
                                                         'MeadowV', 'Mitchel', 'Names','NoRidge', 'NPkVill', 
                                                         'NridgHt', 'NWAmes', 'OldTown','SWISU', 'Sawyer', 
                                                         'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker', 'Up', 'Down'])
            Functional = st.selectbox('Home functionality rating', ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'])
            GarageCars =st.number_input('Size of garage in car capacity', min_value=0.0, max_value=10.0, value=1.0)
            GarageArea = st.number_input('Size of garage in square feet', min_value=0.0, max_value=2000.0, value=1000.0)
            TotRmsAbvGrd = st.number_input('Total rooms above grade (does not include bathrooms)', min_value=0.0, max_value=30.0, value=6.0)
            Fireplaces = st.number_input('Number of fireplaces', min_value=0.0, max_value=5.0, value=1.0)
            YearBuilt = st.number_input('Original construction date', min_value=1800, max_value=2021, value=1980)
            PoolArea = st.number_input('Pool area in square feet', min_value=0.0, max_value=1000.0, value=500.0)
            PoolQC=st.selectbox('Pool quality', ['Gd', 'TA', 'Fa', 'NA'])
            Fence = st.selectbox('Fence quality', ['MnPrv', 'GdWo', 'MnWw', 'NA'])
            Utilities=st.selectbox('Type of utilities available', ['AllPub','NoSewr', 'NoSeWa', 'ELO'])
            MSSubClass=st.selectbox('The building class', [20,30,40,45,50,60,70,75,80,85,90,120,150,160,180,190])
            LotFrontage=st.number_input('Linear feet of street connected to property', min_value=0.0, max_value=400.0, value=100.0)
            LotArea=st.number_input('Lot size in square feet', min_value=0.0, max_value=30000.0, value=10000.0)
            Street=st.selectbox('Type of road access', ['Grvl','Pave'])
            Alley=st.selectbox('Type of alley access', ['Grvl', 'Pave', 'NA'])
            LotShape=st.selectbox('General shape of property', ['Reg', 'IR1', 'IR2', 'IR3'])
            LandContour=st.selectbox('Flatness of the property', ['Lvl', 'Bnk', 'HLS', 'Low'])
            LotConfig=st.selectbox('Lot configuration', ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'])
            Condition1=st.selectbox('Proximity to main road or railroad', ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'])
            Condition2=st.selectbox('Proximity to main road or railroad (if a second is present)', ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'])
            BldgType=st.selectbox('Type of dwelling', ['1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI'])
            HouseStyle=st.selectbox('Style of dwelling', ['1Story', '1 .5Fin', '1 .5Unf', '2Story', '2 .5Fin', '2 .5Unf', 'SFoyer', 'SLvl'])
            YearRemodAdd=st.number_input('Remodel date', min_value=1800, max_value=2021, value=1980)
            RoofStyle=st.selectbox('Type of roof', ['Flat', 'Gabel', 'Gambrel', 'Hip', 'Mansard', 'Shed'])
            RoofMatl=st.selectbox('Roof material', ['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'])
            Exterior1st=st.selectbox('Exterior covering on house', ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco','VinylSd','Wd Sdng', 'WdShing'])
            Exterior2nd=st.selectbox('Exterior covering on house (if more than one material)', ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco','VinylSd','Wd Sdng', 'WdShing'])
            MasVnrType = st.selectbox('Masonry veneer type', ['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone'])
            MasVnrArea=st.number_input('Masonry veneer area in square feet', min_value=0.0, value=100.0)
            ExterQual=st.selectbox('Exterior material quality', ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
            ExterCond=st.selectbox('Present condition of the material on the exterior', ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
            Foundation=st.selectbox('Type of foundation', ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'])
            BsmtQual=st.selectbox('Height of the basement', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
            BsmtCond=st.selectbox('General condition of the basement', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
            BsmtExposure=st.selectbox('Walkout or garden level basement walls', ['Gd', 'Av', 'Mn', 'No', 'NA'])
            BsmtFinType1=st.selectbox('Quality of basement finished area', ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'])
            BsmtFinSF1 = st.number_input('Type 1 finished square feet', min_value=0.0, max_value=6000.0, value=500.0)
            BsmtFinType2 = st.selectbox('Quality of second finished area (if present)', ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'])
            BsmtFinSF2 = st.number_input('Type 2 finished square feet', min_value=0.0, max_value=6000.0, value=500.0)
            BsmtUnfSF =st.number_input('Unfinished square feet of basement area', min_value=0.0, max_value=3000.0, value=500.0)
            TotalBsmtSF = st.number_input('Total square feet of basement area', min_value=0.0, max_value=7000.0, value=1000.0)
            Heating = st.selectbox('Type of heating', ['Floor','GasA','GasW ','Grav','OthW','Wall'])
            HeatingQC = st.selectbox('Heating quality and condition', ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
            SaleCondition=st.selectbox('Condition of sale', ['Normal', 'Abnormal', 'AdjLand', 'Alloca', 'Family', 'Partial'])
            SaleType=st.selectbox('Type of sale', ['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth'])
            YrSold=st.number_input('Year Sold', min_value=1950, max_value=2050, value=2010)
            MoSold=st.selectbox('Month Sold', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            MiscVal=st.number_input('$ Value of miscellaneous feature', min_value=0, max_value=15500, value=5000)
            MiscFeature=st.selectbox('Miscellaneous feature not covered in other categories', ['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'NA'])
            ScreenPorch=st.number_input('Screen porch area in square feet', min_value=0, max_value=1000, value=250)
            SsnPorch=st.number_input('Three season porch area in square feet', min_value=0, max_value=1000, value=250)
            EnclosedPorch=st.number_input('Enclosed porch area in square feet', min_value=0, max_value=1000, value=250)
            OpenPorchSF=st.number_input('Open porch area in square feet', min_value=0, max_value=1000, value=250)
            WoodDeckSF=st.number_input('Wood deck area in square feet', min_value=0, max_value=1000, value=250)
            PavedDrive=st.selectbox('Paved driveway', ['Y', 'P', 'N'])
            GarageCond=st.selectbox('Garage condition', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
            GarageQual=st.selectbox('Garage quality', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
            GarageFinish=st.selectbox('Interior finish of the garage', ['Fin', 'RFn', 'Unf', 'NA'])
            GarageYrBlt=st.number_input('Year garage was built', min_value=1900, max_value=2050, value=1980)
            FireplaceQu=st.selectbox('Fireplace quality', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
            KitchenQual=st.selectbox('Kitchen quality', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
            KitchenAbvGr=st.number_input('Number of kitchens', min_value=0, max_value=10, value=2)
            BedroomAbvGr=st.number_input('Number of bedrooms above basement level', min_value=0, max_value=20, value=4)
            HalfBath=st.number_input('Half baths above grade', min_value=0, max_value=10, value=2)
            BsmtHalfBath=st.number_input('Basement half bathrooms', min_value=0, max_value=10, value=1)
            BsmtFullBath=st.number_input('Basement full bathrooms', min_value=0, max_value=10, value=1)
            LowQualFinSF=st.number_input('Low quality finished square feet (all floors)', min_value=0, max_value=1000, value=5)
            TwondFlrSF=st.number_input('Second floor square feet', min_value=0, max_value=3000, value=300)
            OnestFlrSF=st.number_input('First Floor square feet', min_value=100, max_value=6000, value=400)
            Electrical=st.selectbox('Electrical system', ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'])
            CentralAir=st.selectbox('Central air conditioning', ['N', 'Y'])
         
            
            
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
