Register 'udf.py' using jython as myfuncs;
a = load 'dataset/store.csv' using PigStorage(',') as (line:chararray, name:chararray, mall_id:long);
b = foreach a generate flatten(myfuncs.filter(name,mall_id));
dump b;
