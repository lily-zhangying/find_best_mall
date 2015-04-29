CREATE VIEW IF NOT EXISTS filtered_states AS
SELECT stateid as mallid, state as state, state_id
FROM state 
INNER JOIN states ON state.name = states.state;

CREATE VIEW IF NOT EXISTS filtered_malls AS
SELECT mall.mallid as mallid, mall.name as name, mall.address as address, mall.phone as phone, mall.latitude, mall.longitude, state_id, filtered_states.state as states
FROM mall, filtered_states
WHERE mall.mallid = filtered_states.mallid
ORDER BY filtered_states.state;


    --Creating the table that links maLLS TO COUNTIES
    
CREATE TABLE IF NOT EXISTS mall_county_pair (
    Mall   STRING,
    County STRING,
    state  STRING,
    geoid  INT
);



--CREATE VIRTUAL TABLE test USING csvfile('C:\Users\John\Dropbox\NYU COURSES\Spring 2015\Real Time Analytics and Big Data\Datasets\mall_with_county.csv');

--uploaded manually

--obtaining information about the malls and the counties they are in.

--DELETE FROM mall_county_pair
--WHERE mall_county_pair.state NOT IN (SELECT state FROM states);

--SELECT *
--FROM mall_county_pair
--WHERE state = "alaska";

--INSERT INTO mall_county_pair
--SELECT mallid FROM filtered_malls;


--Upload Geographic data 
CREATE TABLE IF NOT EXISTS demographics (
    name STRING,
    usps STRING,
    geoid INT,
    INTPTLAT INT,
    INTPTLONGPOP10 INT,
    HU10 ,
    ALAND,
    AWATER,
    ALAND_SQMI,
    AWATER_SQMI,
    Homeowner_vacancy_rate_percent,
    Median_Age_Female,
    Median_Age_Male,
    Median_Age,
    Percent_65_years_and_over,
    Percent_American_Indian_and_Alaska_Native,
    Percent_Asian,
    Percent_Black_or_African_American,
    Percent_Family_households,
    Percent_Female,
    Percent_Male,
    Percent_Nonfamily_households,
    Percent_Occupied_housing_units,
    Percent_Some_Other_Race,
    Percent_White,
    Rental_vacancy_rate_percent





);


--Join Demographics with mall_county_pair

CREATE VIEW IF NOT EXISTS mall_with_demo AS
SELECT mall_county_pair.Mall, mall_county_pair.geoid, demographics.*
FROM mall_county_pair
JOIN demographics ON mall_county_pair.geoid = demographics.geoid;


