--SELECT *
--FROM us_malls.store
--WHERE us_malls.store.name = "American Fidelity";

--SELECT * FROM store, test_store
--WHERE store.name = "joppa "
--ORDER BY name;

CREATE TABLE IF NOT EXISTS demographics (
    name STRING,
    usps STRING,
    geoid INT,
    INTPTLAT DOUBLE,
    INTPTLONG DOUBLE,
    POP10 double,
    HU10,
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

--Upload geographic data

CREATE TABLE IF NOT EXISTS mall_county_pair (
    mallid INT,
    mall   STRING,
    county STRING,
    state  STRING,
    geoid  INT
);

--upload data.

CREATE VIEW IF NOT EXISTS mall_with_demo AS
SELECT mall_county_pair.mallid as mallid,  mall_county_pair.Mall as mall, demographics.*
FROM mall_county_pair
JOIN demographics ON mall_county_pair.geoid = demographics.geoid;
