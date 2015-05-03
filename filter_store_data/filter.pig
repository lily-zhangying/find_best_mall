--Register 'udf.py' using jython as myfuncs;
Register 's3://lilybuckettest3210/pigScript/udf.py' using jython as myfuncs;
a = load 's3://lilybuckettest3210/input/store.csv' using PigStorage(',') as (line:chararray, name:chararray, mall_id:long);
b = foreach a generate flatten(myfuncs.filter(name,mall_id));
store b into 's3://lilybuckettest3210/output/pig' using PigStorage(',');


