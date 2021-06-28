# Clustering Project - Zillow
Project Description: develop a machine learning model that will identify logerror drivers in the Zillow dataset

    Conditions: Single Unit properties & 2017 data

    Challenges: 

        * Identify logerror drivers 

            * Incorporate Clustering methodologies in the process

        * Defining Single Unit Property and relevant useid codes

        * Untidy Data: Null Values, Duplicates, Outliers 

        * Identify County specific data using FIPS codes

            * Calculate Tax Rate by County

            * Cross correlations in data
         
Project Findings:

    * I found geolocation and the age of the single property units to be the most significant logerror drivers
        
        * Segmenting the market into a general and premium markets added valuable insights

    * Surprisingly, sqft/bed/bath did not pan out as significant drivers in my analysis
        
        * This may be due to the fact that I combined the 3 into 2 engineered features (bedsqft & bathsqft)

   Conclusion:

   * I believe that the level of granularity necessary to significantly outperform the Benchmark was outside the scope of the time allotted this project
   
        * Geolocation is a significant explanatory variable and I chose to focus my efforts at the County level. A more localized approach (e.g. City, Neighborhood) could lead to better results and better account for factors like age and market segments

    * My expectation that I'd be able to add predictive value by sorting the data out into two markets panned out in the hypothesis testing and modeling.

    * One of the clusters I utilized focused on mapping geolocation by market segment.  I believe by segmenting the market like this and coupling with a cluster focused on age I was able to explain a large portion of what is driving logerror in the Zillow model

    Model Selection: The Polynomial Regressor algorithm offered the most predictive value

    * I also modeled using LassoLars and OLS.  In general, the results were very similar but the Polynomial Regressor slightly outperformed the Benchmark on the out-of-sample test.  This continued with greater outperformance on the test sample.  

Next Steps: I remain convinced I was on to something with the market segmentation and assume that the contradicting information by County may have created some noise. E.G. market segmentation and age weren't consistent between Counties and even intra-County.

In my next steps, I would make an effort to parse geolocation in greater detail than the County level wiith the expectation that the level of granularity could yield better results


### Project Objectives

- Data Pipeline: Acquire data from SQL database and convert into a panda dataframe, prepare the data for exploration, explore the data and document key takeaways, visualize feature attributes, identify clusters and incorporate into machine learning model.

- Document progress and present results in Jupyter Notebook

- Create modules (acquire.py, prepare.py) that make process repeateable for 3rd party

- Present thought process and modeling results to cohort utilizing Jupyter Notebook as presentation material

- Be prepared to handle questions about code, process, findings, key takeaways, and model.

### Project Goals

- Find drivers for Zillow logerrors in estimating Single Unit Property Prices from the Zillow database that can be utilized to construct predictive model

- Construct a regression model that minimizes RMSE when identifying significant logerror correlations/drivers

- Document the process so that 3rd party can read like a report and easily follow along/replicate


### Audience
- Target audience for my notebook walkthrough is the Codeup Data Science team

### Deliverables
-  Jupyter Notebook Report showing process and analysis with the goal of finding drivers Single Unit Price logerror drivers. This notebook should be commented and documented well enough to be read like a report or walked through as a presentation.

- README.md file containing the project description with goals, a data dictionary, project planning (lay out your process through the data science pipeline), instructions or an explanation of how someone else can recreate your project and findings (What would someone need to be able to recreate your project on their own?), key findings, recommendations, and takeaways from your project.

- individual modules, .py files, that hold functions to acquire and prepare data.

### Pipeline Stages Breakdown ==> Plan

-Create README.md with data dictionary, project and business goals, come up with initial hypotheses.

- Set project goals and define deliverables

### Pipeline Stages Breakdown ==> Acquire

- Write function that establish connectivity to SQL Ace, run an SQL query on the zillow database and pulls relevant tables and data

- Convert imported data into dataframe

- Complete some initial data summarization (.info(), .describe(), .value_counts(), ...).

- store functions in acquire.py

### Pipeline Stages Breakdown ==> Prepare

- prepare.py module

    - Store functions that are needed to prepare data; making sure module contains the necessary imports to run code. Final function should do the following:

        - Split data into train/validate/test.

        -  Handle Missing Values.

        - Handle erroneous data and/or address outliers 

        - Encode variables as needed.

        - Create any new features, if you decided to make any for this project.

-  Notebook

    - Explore missing values and document takeaways/action plans for handling them.

    - Is 'missing' equivalent to 0 (or some other constant value) in the specific case of this variable?

    - Replace the missing values with a value it is most likely to represent, like mean/median/mode?

    - Remove the variable (column) altogether because of the percentage of missing data?

    - Remove individual observations (rows) with a missing value for that variable?

    - Explore data types and adapt types or data values as needed to have numeric represenations of each attribute.

### Pipeline Stages Breakdown ==> Explore
- Notebook

    - Answer key questions, formulate hypotheses, and figure out the drivers of Single Unit Property Prices. Run statistical tests in data exploration. 

    - Create visualizations and run statistical tests that work toward discovering variable relationships (independent with independent and independent with dependent). Goal is to identify features that are related to Single Unit Property Prices logerrors (target), identify any data integrity issues, and understand 'how the data works'. 

    - Additionally, the aim is to identify at least 4 data clusters utilizing ml algorigthms to help enhance the discovery and modeling process

    - Summarize conclusions, provide clear answers to specific questions, and summarize any takeaways/action plan from the work above..

### Pipeline Stages Breakdown ==> Modeling and Evaluation

- Notebook

    - Establish a baseline accuracy to determine if having a model is better than no model and train and compare at least 3 different models. Document steps.

    -  Train (fit, transform, evaluate) multiple models, varying the algorithm and/or hyperparameters you use.

    - Compare RMSE across all the models 

    - Based on the evaluation of your models using the train and validate datasets, choose best model that I will try with the test data, once.

    - Test the final model on your out-of-sample data (the testing dataset), summarize the performance, interpret and document your results.

### Initial thoughts and Hypothesis

- I suspect location, age and bed/bath will be primary logerror drivers 

### Repeating processes and results
- the final notebook provides a detailed step by step process that should easily be followed once the data is imported.  

- The data acquire functions and data prepare functions will do the heavy lifting but anyone looking to recreate the study will need to write their own env.py file containing their SQL access data.  

- Key Takeaways are provided at each important step to guide the reader in the though process and enable them to act on their own thoughts and interact with the process. 


## Data Dictionary
|  Single Unit Property Defined: Single Unit Properties: A housing unit is a single unit within a larger structure that can be used by an individual or household to eat, sleep, and live. The unit can be in any type of residence, such as a house, apartment, or mobile home, and may also be a single unit in a group of rooms.|


|   Feature      | Description    |
| :------------- | -----------: |
| Single Unit Property | housing unit is a single unit within a larger structure that can be used by an individual or household to eat, sleep, and live |
| airconditioningtypeid | Type of cooling system present in the home (if any) |
| architecturalstyletypeid | Architectural style of the home (i.e. ranch, colonial, split-level, etc |
| basementsqft  | Finished living area below or partially below ground level|
| bathroomcnt| Number of bathrooms in home including fractional bathrooms|
| buildingclasstypeid | the building framing type (steel frame, wood frame, concrete/brick)|
| calculatedbathnbr | Number of bathrooms in home including fractional bathroom|
| decktypeid | Type of deck (if any) present on parcel|
| transactiondate | date of property sale |
| basementsqft |	Finished living area below or partially below ground level |
| bathroomcnt |	Number of bathrooms in home including fractional bathrooms |
| bedroomcnt |	Number of bedrooms in home |
| buildingqualitytypeid |	Overall assessment of condition of the building from best (lowest) to worst (highest) |
| buildingclasstypeid |	The building framing type (steel frame, wood frame, concrete/brick) |
| calculatedbathnbr |	Number of bathrooms in home including fractional bathroom |
| decktypeid |	Type of deck (if any) present on parcel |
| threequarterbathnbr |	Number of 3/4 bathrooms in house (shower + sink + toilet) |
| finishedfloor1squarefeet |	Size of the finished living area on the first (entry) floor of the home |
| calculatedfinishedsquarefeet | Calculated total finished living area of the home |
| finishedsquarefeet6 |	Base unfinished and finished area |
| finishedsquarefeet12 | Finished living area |
| finishedsquarefeet13 |	Perimeter living area |
| finishedsquarefeet15 |	Total area |
| finishedsquarefeet50 |	Size of the finished living area on the first (entry) floor of the home |
| fips |Federal Information Processing Standard code - see https://en.wikipedia.org/wiki/FIPS_county_code for more details |
| fireplacecnt |	Number of fireplaces in a home (if any) |
| fireplaceflag |	Is a fireplace present in this home |
| fullbathcnt |	Number of full bathrooms (sink, shower + bathtub, and toilet) present in home |
| garagecarcnt | 	Total number of garages on the lot including an attached garage |
| garagetotalsqft |	Total number of square feet of all garages on lot including an attached garage |
| hashottuborspa |	Does the home have a hot tub or spa |
| heatingorsystemtypeid |	Type of home heating system |
| latitude | Latitude of the middle of the parcel multiplied by 10e6 |
| longitude |	Longitude of the middle of the parcel multiplied by 10e6 |
| lotsizesquarefeet |	Area of the lot in square feet |
| numberofstories |	Number of stories or levels the home has |
| parcelid |	Unique identifier for parcels (lots) |
| poolcnt |	Number of pools on the lot (if any) |
| poolsizesum |	Total square footage of all pools on property |
| pooltypeid10 |	Spa or Hot Tub |
| pooltypeid2 |	Pool with Spa/Hot Tub |
| pooltypeid7 |	Pool without hot tub |
| propertycountylandusecode |	County land use code i.e. it's zoning at the county level |
| propertylandusetypeid |	Type of land use the property is zoned for |
| propertyzoningdesc |	Description of the allowed land uses (zoning) for that property |
| rawcensustractandblock |	Census tract and block ID combined - also contains blockgroup assignment by extension |
| censustractandblock |	Census tract and block ID combined - also contains blockgroup assignment by extension |
| regionidcounty | County in which the property is located |
| regionidcity |	City in which the property is located (if any) |
| regionidzip |	Zip code in which the property is located |
| regionidneighborhood | Neighborhood in which the property is located |
| roomcnt |	Total number of rooms in the principal residence |
| storytypeid |	Type of floors in a multi-story house (i.e. basement and main level, split-level, attic, etc.). See tab for details. |
| typeconstructiontypeid |	What type of construction material was used to construct the home |
| unitcnt |	Number of units the structure is built into (i.e. 2 = duplex, 3 = triplex, etc...) |
| yardbuildingsqft17 |	Patio in yard |
| yardbuildingsqft26 |	Storage shed/building in yard |
| yearbuilt |	The Year the principal residence was built |
| taxvaluedollarcnt |	The total tax assessed value of the parcel |
| structuretaxvaluedollarcnt |	The assessed value of the built structure on the parcel |
| landtaxvaluedollarcnt |	The assessed value of the land area of the parcel |
| taxamount |	The total property tax assessed for that assessment year |
| assessmentyear |	The year of the property tax assessment |
| taxdelinquencyyear |	Year for which the unpaid propert taxes were due |