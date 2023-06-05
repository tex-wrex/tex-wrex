<!-- Title -->
<div style='background-color: green'>
<head>
    <h1 align='center'><b><u><i>
        Tex-Wrex - README
    </i></u></b></h1>
</head></div>





<!-- Table of Contents -->
<div style='background-color: orange'>
<head>
    <h3 align='center'><b><i>
        <a id='tableofcontents'></a>
        Table of Contents:
    </i></b></h3>
</head></div>
<h5>
<li><a href='#additional'>Additional Project Resources</a href></li>
<li><a href='#description'>Project Description</a href></li>
<li><a href='#goals'>Project Goals</a href></li>
<li><a href='#hypo'>Hypothesis/Questions</a href></li>
<li><a href='#datadict'>Data Dictionary</a href></li>
<li><a href='#planning'>Planning</a href></li>
<li><a href='#instructions'>Instruction To Replicate</a href></li>
<li><a href='#takeaways'>Takeaways</a href></li>
</h5>
<br><br><br>




<!-- Additional Project Resources -->
<div style='background-color: orange'>
<head>
    <h3 align='center'><b><i>
        <a id='additional'></a>Additional Project Resources
    </i></b></h3>
</head></div>
<a href='#tableofcontents'>Back to 'Table of Contents'</a>
<br><br>
<h5>
Hyperlink list of additional project resources:
<li><a href='https://cris.dot.state.tx.us/public/Query/app/home'>CRIS Query</a></li>
<li><a href='https://www.canva.com/design/DAFkmDnkHN0/zq2MpInKm-TME4-m1UmwRA/edit?utm_content=DA[…]m_campaign=designshare&utm_medium=link2&utm_source=sharebutton'>Canva Presentation</a></li>
</h5>
<br><br><br>





<!-- Project Description -->
<div style='background-color: orange'>
<head>
    <h3 align='center'><b><i>
        <a id='description'></a>Project Description:
    </i></b></h3>
</head></div>
<a href='#tableofcontents'>Back to 'Table of Contents'</a>
<br><br>
<h5>
Using crash conditions, vehicle, driver, and road data of single motorcycle crash incidents (Crashes where only a single motorcycle was involved) from years 2018 - 2022 in the <a href='https://cris.dot.state.tx.us/public/Query/app/home'>CRIS Query</a>, identify patterns and create a classification model to predict the level of injury severity of the motorcycle driver.
</h5>
<br><br><br>





<!-- Project Goals -->
<div style='background-color: orange'>
<head>
    <h3 align='center'><b><i>
        <a id='goals'></a>Project Goals:
    </i></b></h3>
</head></div>
<a href='#tableofcontents'>Back to 'Table of Contents'</a>
<br><br>
<h5>
<li>Create a classification type machine learning model that best predicts the level of injury of the motorcycle driver</li>
<li>Successfully acquire data from the <a href='https://cris.dot.state.tx.us/public/Query/app/home'>CRIS Query</a></li>
<li>Adequately prepare the data in order to streamline exploration and modeling with minimal issues</li>
<li>Identify pertinent features for modeling</li>
<li>Create the best classification model that can accurately predict the level of injury of the motorcycle driver</li>
<li>Ensure files and references are available and adequate for other's information gain</li>
<li>Create and deliver a professional presentation via Canva</li>
<li>Create a model that conducts a risk assessment of motorcycle drivers to better determine the insurance premium for the motorcyclist</li>
</h5>
<br><br><br>





<!-- Hypothesis/Questions -->
<div style='background-color: orange'>
<head>
    <h3 align='center'><b><i>
        <a id='hypo'></a>Hypothesis/Questions:
    </i></b></h3>
</head></div>
<a href='#tableofcontents'>Back to 'Table of Contents'</a>
<br><br>
<h4><b>
Hypothesis:
</b></h4>
<h5>
A combination of crash conditions, vehicle, personnel, and road data will dictate the motorcyclist's level of injury in a crash, but the road's curvature, the local population, and condition of the motorcycle would most likely impact the outcome the most.
</h5>
<br>
<h4><b>
Questions:
</b></h4>
<h5>
<li>Does the time of day matter in relation to injury severity?</li>
<li>Does the weather at the time of the crash relate to injury severity?</li>
<li>Does the surface condition of the roadways relate to injury severity?</li>
<li>Does the total count of inanimate objects struck by the motorcyclist relate to injury severity?</li>
<li>Does the failure to control speed relate to injury severity?</li>
<li>Does speeding relate to injury severity?</li>
<li>Does the motorcycle brand relate to injury severity?</li>
<li>Does the maintainence condition of the motorcycle relate to injury severity?</li>
<li>Which motorcycle crashes the most? Type? Brand?</li>
<li>Which age is most likely to have fatal motorcycle crashes?</li>
<li>Which age is most likely to be in a motorcycle crash?</li>
<li>Does the income salary of the motorcyclist relate to injury severity?</li>
<li>Does the curvature of the road affect the injury severity?</li>
<li>Does whether or not there is merging traffic relate to injury severity?</li>
<li>Do intersections relate to injury severity?</li>
<li>Does the length of motorcycle trips relate to injury severity?</li>
</h5>
<br><br><br>






<!-- Data Dictionary -->
<div style='background-color: orange'>
<head>
    <h3 align='center'><b><i>
        <a id='datadict'></a>Data Dictionary:
    </i></b></h3>
</head></div>
<a href='#tableofcontents'>Back to 'Table of Contents'</a>
<br><br>

Due to the extensive amount of columns (254) in the dataset, this data dictionary only includes the columns used in the modeling phase as to avoid having a massive data dictionary.

| Feature Name | Data Type | Description | Example |
| ----- | ----- | ----- | ----- |
| feature | bool | True/False | True |
| crash_id | int | ID # of the crash incident | 16189632 |
| person_age | int | Age of the individual involved in the crash | 37 |
| charge | object | Charge given to a person involved in the crash | OPERATE UNREGISTERED MOTOR VEHICLE |
| person_ethnicity | object | Ethnicity of the person involved in the crash | w - white |
| crash_date | object | (YYYY-MM-DD) Date that the crash occured on | 2018-01-01 |
| day_of_week | object | Name of the day of the week that the crash occured on | monday |
| person_gender | object | Gender of the person involved in the crash | 1 - male |
| person_helmet | object | If a helmet was worn and/or the condition of the helmet of the person involved in the crash | 1 - not worn |
| driver_license_class | object | The driver license class of the person involved in the crash | c - class c |
| driver_license_state | object | The state that gave the driver license to in relation to the person involved in the crash | tx - texas |
| driver_license_type | object | The type of driver license the person involved in the crash had | 1 - driver license |
| person_injury_severity | object | The level of injury severity sustained by the person involved in the crash (TARGET VARIABLE) | a - suspected serious injury |
| license_plate_state | object | The state of the license plate on the vehicle in the crash | tx - texas |
| vehicle_body_style | object | The type of vehicle involved in the crash | mc - motorcycle |
| vehicle_color | object | The color that the vehicle was at the crash | blu - blue |
| vehicle_defect_1 | object | Vehicle defects at the time of the crash | 5 - defective or no headlamps |
| vehicle_make | object | The make of the vehicle at the crash | suzuku |
| vehicle_model_name | object | The name of the vehicle's model at the crash | gsx-r600 (suzuki) |
| vehicle_model_year | object | The year of the vehicle make that was at the crash | 2004 |

<br><br><br>






<!-- Planning -->
<div style='background-color: orange'>
<head>
    <h3 align='center'><b><i>
        <a id='planning'></a>Planning:
    </i></b></h3>
</head></div>
<a href='#tableofcontents'>Back to 'Table of Contents'</a>
<br><br>
<h4><b>Objective</b></h4>
<li>Identify key features for a classification machine learning model to train off of in order to make the most accurate prediction in relation to the motorcyclist's level of injury from the crash.</li>
<br>
<h4><b>Methodology</b></h4>
<li>Data science pipeline</li>
<li>Explore for key features and relationships</li>
<li>Create clusters if and when necessary</li>
<li>Create models to best predict quality</li>
<li>Deliver takeaways</li>
<br>
<h4><b>Deliverables</b></h4>
<li>Canva Presentation <a href='https://www.canva.com/design/DAFkmDnkHN0/zq2MpInKm-TME4-m1UmwRA/edit?utm_content=DA[…]m_campaign=designshare&utm_medium=link2&utm_source=sharebutton'>CLICK HERE</a></li>
<li>Full and complete group repository with pertinent work.</li>
<br><br><br>






<!-- Instructions To Replicate -->
<div style='background-color: orange'>
<head>
    <h3 align='center'><b><i>
        <a id='instructions'></a>Instructions To Replicate:
    </i></b></h3>
</head></div>
<a href='#tableofcontents'>Back to 'Table of Contents'</a>
<br><br>

1. Clone this repo
2. Obtain 'master_list.csv'
    1. Go to <a href='https://cris.dot.state.tx.us/public/Query/app/query-builder'>CRIS Query Builder</a>
    2. Accept the Query Disclaimer
    3. Under 'Select Crash Date and Time', select 'Select Crashes from a range of years'
    4. 'Begin Year' == 2018
    5. 'End Year' == 2022
    6. Under 'Select Crash Location', select 'Search All of Texas'
    7. Under 'Filters Authorized by TxDOT, select ONLY 'Motorcycle Related Crashes'
    8. Click 'Save'
    9. 
3. Run a virtual environment with the 'requirements.txt' file
4. Run desired files/operations
<br><br><br>





<!-- Takeaways -->
<div style='background-color: orange'>
<head>
    <h3 align='center'><b><i>
        <a id='takeaways'></a>Takeaways:
    </i></b></h3>
</head></div>
<a href='#tableofcontents'>Back to 'Table of Contents'</a>
<br><br>
<h4><b>Summary:</b></h4>

- When strictly looking at what causes single motorcycle crash incidents, there is a pattern of being later in the evening, during warmer months, and generally grouped in areas with curves and hazards in the road.  In relation to the injury severity, that gets more complex, but a combination of the driver's background and the information of their motorcycle can lead to a fairly accurate predictive model.
<br><br>
<h4><b>Recommendations:</b></h4>

- Our model is {acc_score}% accurate in predicting the level of injury severity for a motorcyclist should they get into a single motorcycle crash.  Because of this, implementing our model or our aggregated risk score for an individual may prove useful in determining the individual's insurance premium cost.
<br><br>
<h4><b>Next Steps:</b></h4>

- Attempt to derive more in-depth location data
    - Residency (Rural/Urban)
    - Population density/sq.mile
    - What kind of roads to expect in immeadiate vicinity (Highway/non)
    - How often are there curves in the road
- Attempt to derive more information about the driver's driving habits
    - How far do they drive on average
    - What vehicles do they normally drive
    - What is their crash history
- Attempt to get a more accurate model that also reflects in the individual injury risk value for the person
<br><br>