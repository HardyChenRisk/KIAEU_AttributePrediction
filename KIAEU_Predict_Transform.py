# Databricks notebook source
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql.functions import *
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

def unionAll(dfs):
    return functools.reduce(lambda df1, df2: df1.union(df2.select(df1.columns)), dfs)

# COMMAND ----------

data1 = spark.read.options(delimiter="|", header='true', inferschema='true').csv("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/archive_0619_20200915.csv")
data1 = data1.select('vin','counttotaltrips', 'score_dt','idl','train_flag',"at_fault_bipd_all_cnt","exposure","at_fault_bipd_all_amt","bi_all_amt","bi_all_cnt","co_amt","co_cnt","first_claim_date","garage_zip","pd_all_amt","pd_all_cnt","pd_amt","pd_cnt","meanavespeed","distanceperday","permetercountaccmeterpersecsquaredge40","permetercountbrakemeterpersecsquaredge40","pctsecondsovermph80","aveovermph80speed",'pctsecondsdurationdays7time1800to1900','pctsecondsdurationdays7time1900to2000','pctsecondsdurationdays7time2000to2100','pctsecondsdurationdays1time1800to1900','pctsecondsdurationdays1time1900to2000',           'pctsecondsdurationdays1time2000to2100','pctsecondsdurationdays2time1800to1900','pctsecondsdurationdays2time1900to2000','pctsecondsdurationdays2time2000to2100','pctsecondsdurationdays3time1800to1900','pctsecondsdurationdays3time1900to2000','pctsecondsdurationdays3time2000to2100','pctsecondsdurationdays4time1800to1900','pctsecondsdurationdays4time1900to2000','pctsecondsdurationdays4time2000to2100','pctsecondsdurationdays1time0000to0100','pctsecondsdurationdays1time0100to0200','pctsecondsdurationdays1time0200to0300','pctsecondsdurationdays1time0300to0400','pctsecondsdurationdays7time2100to2200','pctsecondsdurationdays7time2200to2300','pctsecondsdurationdays7time2300to2400','pctsecondsdurationdays2time0000to0100','pctsecondsdurationdays2time0100to0200','pctsecondsdurationdays2time0200to0300','pctsecondsdurationdays2time0300to0400','pctsecondsdurationdays1time2100to2200','pctsecondsdurationdays1time2200to2300','pctsecondsdurationdays1time2300to2400','pctsecondsdurationdays3time0000to0100','pctsecondsdurationdays3time0100to0200','pctsecondsdurationdays3time0200to0300','pctsecondsdurationdays3time0300to0400','pctsecondsdurationdays2time2100to2200','pctsecondsdurationdays2time2200to2300','pctsecondsdurationdays2time2300to2400','pctsecondsdurationdays4time0000to0100','pctsecondsdurationdays4time0100to0200','pctsecondsdurationdays4time0200to0300','pctsecondsdurationdays4time0300to0400','pctsecondsdurationdays3time2100to2200','pctsecondsdurationdays3time2200to2300','pctsecondsdurationdays3time2300to2400','pctsecondsdurationdays5time0000to0100','pctsecondsdurationdays5time0100to0200','pctsecondsdurationdays5time0200to0300','pctsecondsdurationdays5time0300to0400','pctsecondsdurationdays4time2100to2200','pctsecondsdurationdays4time2200to2300','pctsecondsdurationdays4time2300to2400','pctsecondsdurationdays5time1900to2000','pctsecondsdurationdays5time2000to2100','pctsecondsdurationdays5time2100to2200','pctsecondsdurationdays6time1900to2000','pctsecondsdurationdays6time2000to2100','pctsecondsdurationdays6time2100to2200','pctsecondsdurationdays6time0000to0100','pctsecondsdurationdays6time0100to0200','pctsecondsdurationdays6time0200to0300','pctsecondsdurationdays6time0300to0400','pctsecondsdurationdays6time0400to0500','pctsecondsdurationdays6time0500to0600','pctsecondsdurationdays5time2200to2300','pctsecondsdurationdays5time2300to2400','pctsecondsdurationdays7time0000to0100','pctsecondsdurationdays7time0100to0200','pctsecondsdurationdays7time0200to0300','pctsecondsdurationdays7time0300to0400','pctsecondsdurationdays7time0400to0500','pctsecondsdurationdays7time0500to0600','pctsecondsdurationdays6time2200to2300','pctsecondsdurationdays6time2300to2400', 
'secondsaccmeterpersecsquaredge35', 'secondsaccmeterpersecsquaredge40', 'secondsaccmeterpersecsquaredge45', 'secondsaccmeterpersecsquaredge50', 'secondsbrakemeterpersecsquaredge35','secondsbrakemeterpersecsquaredge40', 'secondsbrakemeterpersecsquaredge45', 'secondsbrakemeterpersecsquaredge50', 'meterstotaldistance', 'secondsovermph80', 'metersovermph80',  'secondsdurationtotaltime', 'DistancePerTrip', 'secondsdurationdays7time1800to1900','secondsdurationdays7time1900to2000','secondsdurationdays7time2000to2100','secondsdurationdays1time1800to1900','secondsdurationdays1time1900to2000',           'secondsdurationdays1time2000to2100','secondsdurationdays2time1800to1900','secondsdurationdays2time1900to2000','secondsdurationdays2time2000to2100','secondsdurationdays3time1800to1900','secondsdurationdays3time1900to2000','secondsdurationdays3time2000to2100','secondsdurationdays4time1800to1900','secondsdurationdays4time1900to2000','secondsdurationdays4time2000to2100','secondsdurationdays1time0000to0100','secondsdurationdays1time0100to0200','secondsdurationdays1time0200to0300','secondsdurationdays1time0300to0400','secondsdurationdays7time2100to2200','secondsdurationdays7time2200to2300','secondsdurationdays7time2300to2400','secondsdurationdays2time0000to0100','secondsdurationdays2time0100to0200','secondsdurationdays2time0200to0300','secondsdurationdays2time0300to0400','secondsdurationdays1time2100to2200','secondsdurationdays1time2200to2300','secondsdurationdays1time2300to2400','secondsdurationdays3time0000to0100','secondsdurationdays3time0100to0200','secondsdurationdays3time0200to0300','secondsdurationdays3time0300to0400','secondsdurationdays2time2100to2200','secondsdurationdays2time2200to2300','secondsdurationdays2time2300to2400','secondsdurationdays4time0000to0100','secondsdurationdays4time0100to0200','secondsdurationdays4time0200to0300','secondsdurationdays4time0300to0400','secondsdurationdays3time2100to2200','secondsdurationdays3time2200to2300','secondsdurationdays3time2300to2400','secondsdurationdays5time0000to0100','secondsdurationdays5time0100to0200','secondsdurationdays5time0200to0300','secondsdurationdays5time0300to0400','secondsdurationdays4time2100to2200','secondsdurationdays4time2200to2300','secondsdurationdays4time2300to2400','secondsdurationdays5time1900to2000','secondsdurationdays5time2000to2100','secondsdurationdays5time2100to2200','secondsdurationdays6time1900to2000','secondsdurationdays6time2000to2100','secondsdurationdays6time2100to2200','secondsdurationdays6time0000to0100','secondsdurationdays6time0100to0200','secondsdurationdays6time0200to0300','secondsdurationdays6time0300to0400','secondsdurationdays6time0400to0500','secondsdurationdays6time0500to0600','secondsdurationdays5time2200to2300','secondsdurationdays5time2300to2400','secondsdurationdays7time0000to0100','secondsdurationdays7time0100to0200','secondsdurationdays7time0200to0300','secondsdurationdays7time0300to0400','secondsdurationdays7time0400to0500','secondsdurationdays7time0500to0600','secondsdurationdays6time2200to2300','secondsdurationdays6time2300to2400', 'metersbtwfirstandlasttrip', 'daysbtwfirstandlasttripinclusive')

# COMMAND ----------

import functools
data1 = spark.read.options(delimiter="|", header='true', inferschema='true').csv("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/archive_0619_20200915.csv")
data1 = data1.select('vin','counttotaltrips','score_dt','idl','train_flag',"at_fault_bipd_all_cnt","exposure","at_fault_bipd_all_amt","bi_all_amt","bi_all_cnt","co_amt","co_cnt","first_claim_date","garage_zip","pd_all_amt","pd_all_cnt","pd_amt","pd_cnt","meanavespeed","distanceperday","permetercountaccmeterpersecsquaredge40","permetercountbrakemeterpersecsquaredge40","pctsecondsovermph80","aveovermph80speed",'pctsecondsdurationdays7time1800to1900','pctsecondsdurationdays7time1900to2000','pctsecondsdurationdays7time2000to2100','pctsecondsdurationdays1time1800to1900','pctsecondsdurationdays1time1900to2000',           'pctsecondsdurationdays1time2000to2100','pctsecondsdurationdays2time1800to1900','pctsecondsdurationdays2time1900to2000','pctsecondsdurationdays2time2000to2100','pctsecondsdurationdays3time1800to1900','pctsecondsdurationdays3time1900to2000','pctsecondsdurationdays3time2000to2100','pctsecondsdurationdays4time1800to1900','pctsecondsdurationdays4time1900to2000','pctsecondsdurationdays4time2000to2100','pctsecondsdurationdays1time0000to0100','pctsecondsdurationdays1time0100to0200','pctsecondsdurationdays1time0200to0300','pctsecondsdurationdays1time0300to0400','pctsecondsdurationdays7time2100to2200','pctsecondsdurationdays7time2200to2300','pctsecondsdurationdays7time2300to2400','pctsecondsdurationdays2time0000to0100','pctsecondsdurationdays2time0100to0200','pctsecondsdurationdays2time0200to0300','pctsecondsdurationdays2time0300to0400','pctsecondsdurationdays1time2100to2200','pctsecondsdurationdays1time2200to2300','pctsecondsdurationdays1time2300to2400','pctsecondsdurationdays3time0000to0100','pctsecondsdurationdays3time0100to0200','pctsecondsdurationdays3time0200to0300','pctsecondsdurationdays3time0300to0400','pctsecondsdurationdays2time2100to2200','pctsecondsdurationdays2time2200to2300','pctsecondsdurationdays2time2300to2400','pctsecondsdurationdays4time0000to0100','pctsecondsdurationdays4time0100to0200','pctsecondsdurationdays4time0200to0300','pctsecondsdurationdays4time0300to0400','pctsecondsdurationdays3time2100to2200','pctsecondsdurationdays3time2200to2300','pctsecondsdurationdays3time2300to2400','pctsecondsdurationdays5time0000to0100','pctsecondsdurationdays5time0100to0200','pctsecondsdurationdays5time0200to0300','pctsecondsdurationdays5time0300to0400','pctsecondsdurationdays4time2100to2200','pctsecondsdurationdays4time2200to2300','pctsecondsdurationdays4time2300to2400','pctsecondsdurationdays5time1900to2000','pctsecondsdurationdays5time2000to2100','pctsecondsdurationdays5time2100to2200','pctsecondsdurationdays6time1900to2000','pctsecondsdurationdays6time2000to2100','pctsecondsdurationdays6time2100to2200','pctsecondsdurationdays6time0000to0100','pctsecondsdurationdays6time0100to0200','pctsecondsdurationdays6time0200to0300','pctsecondsdurationdays6time0300to0400','pctsecondsdurationdays6time0400to0500','pctsecondsdurationdays6time0500to0600','pctsecondsdurationdays5time2200to2300','pctsecondsdurationdays5time2300to2400','pctsecondsdurationdays7time0000to0100','pctsecondsdurationdays7time0100to0200','pctsecondsdurationdays7time0200to0300','pctsecondsdurationdays7time0300to0400','pctsecondsdurationdays7time0400to0500','pctsecondsdurationdays7time0500to0600','pctsecondsdurationdays6time2200to2300','pctsecondsdurationdays6time2300to2400', 
'secondsaccmeterpersecsquaredge35', 'secondsaccmeterpersecsquaredge40', 'secondsaccmeterpersecsquaredge45', 'secondsaccmeterpersecsquaredge50', 'secondsbrakemeterpersecsquaredge35','secondsbrakemeterpersecsquaredge40', 'secondsbrakemeterpersecsquaredge45', 'secondsbrakemeterpersecsquaredge50', 'meterstotaldistance', 'secondsovermph80', 'metersovermph80',  'secondsdurationtotaltime', 'DistancePerTrip', 'secondsdurationdays7time1800to1900','secondsdurationdays7time1900to2000','secondsdurationdays7time2000to2100','secondsdurationdays1time1800to1900','secondsdurationdays1time1900to2000',           'secondsdurationdays1time2000to2100','secondsdurationdays2time1800to1900','secondsdurationdays2time1900to2000','secondsdurationdays2time2000to2100','secondsdurationdays3time1800to1900','secondsdurationdays3time1900to2000','secondsdurationdays3time2000to2100','secondsdurationdays4time1800to1900','secondsdurationdays4time1900to2000','secondsdurationdays4time2000to2100','secondsdurationdays1time0000to0100','secondsdurationdays1time0100to0200','secondsdurationdays1time0200to0300','secondsdurationdays1time0300to0400','secondsdurationdays7time2100to2200','secondsdurationdays7time2200to2300','secondsdurationdays7time2300to2400','secondsdurationdays2time0000to0100','secondsdurationdays2time0100to0200','secondsdurationdays2time0200to0300','secondsdurationdays2time0300to0400','secondsdurationdays1time2100to2200','secondsdurationdays1time2200to2300','secondsdurationdays1time2300to2400','secondsdurationdays3time0000to0100','secondsdurationdays3time0100to0200','secondsdurationdays3time0200to0300','secondsdurationdays3time0300to0400','secondsdurationdays2time2100to2200','secondsdurationdays2time2200to2300','secondsdurationdays2time2300to2400','secondsdurationdays4time0000to0100','secondsdurationdays4time0100to0200','secondsdurationdays4time0200to0300','secondsdurationdays4time0300to0400','secondsdurationdays3time2100to2200','secondsdurationdays3time2200to2300','secondsdurationdays3time2300to2400','secondsdurationdays5time0000to0100','secondsdurationdays5time0100to0200','secondsdurationdays5time0200to0300','secondsdurationdays5time0300to0400','secondsdurationdays4time2100to2200','secondsdurationdays4time2200to2300','secondsdurationdays4time2300to2400','secondsdurationdays5time1900to2000','secondsdurationdays5time2000to2100','secondsdurationdays5time2100to2200','secondsdurationdays6time1900to2000','secondsdurationdays6time2000to2100','secondsdurationdays6time2100to2200','secondsdurationdays6time0000to0100','secondsdurationdays6time0100to0200','secondsdurationdays6time0200to0300','secondsdurationdays6time0300to0400','secondsdurationdays6time0400to0500','secondsdurationdays6time0500to0600','secondsdurationdays5time2200to2300','secondsdurationdays5time2300to2400','secondsdurationdays7time0000to0100','secondsdurationdays7time0100to0200','secondsdurationdays7time0200to0300','secondsdurationdays7time0300to0400','secondsdurationdays7time0400to0500','secondsdurationdays7time0500to0600','secondsdurationdays6time2200to2300','secondsdurationdays6time2300to2400', 'metersbtwfirstandlasttrip', 'daysbtwfirstandlasttripinclusive')

data2 = spark.read.options(delimiter="|", header='true', inferschema='true').csv("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/archive_1218_20200915.csv")
data2 = data2.select('vin','counttotaltrips','score_dt','idl','train_flag',"at_fault_bipd_all_cnt","exposure","at_fault_bipd_all_amt","bi_all_amt","bi_all_cnt","co_amt","co_cnt","first_claim_date","garage_zip","pd_all_amt","pd_all_cnt","pd_amt","pd_cnt","meanavespeed","distanceperday","permetercountaccmeterpersecsquaredge40","permetercountbrakemeterpersecsquaredge40","pctsecondsovermph80","aveovermph80speed",'pctsecondsdurationdays7time1800to1900','pctsecondsdurationdays7time1900to2000','pctsecondsdurationdays7time2000to2100','pctsecondsdurationdays1time1800to1900','pctsecondsdurationdays1time1900to2000',           'pctsecondsdurationdays1time2000to2100','pctsecondsdurationdays2time1800to1900','pctsecondsdurationdays2time1900to2000','pctsecondsdurationdays2time2000to2100','pctsecondsdurationdays3time1800to1900','pctsecondsdurationdays3time1900to2000','pctsecondsdurationdays3time2000to2100','pctsecondsdurationdays4time1800to1900','pctsecondsdurationdays4time1900to2000','pctsecondsdurationdays4time2000to2100','pctsecondsdurationdays1time0000to0100','pctsecondsdurationdays1time0100to0200','pctsecondsdurationdays1time0200to0300','pctsecondsdurationdays1time0300to0400','pctsecondsdurationdays7time2100to2200','pctsecondsdurationdays7time2200to2300','pctsecondsdurationdays7time2300to2400','pctsecondsdurationdays2time0000to0100','pctsecondsdurationdays2time0100to0200','pctsecondsdurationdays2time0200to0300','pctsecondsdurationdays2time0300to0400','pctsecondsdurationdays1time2100to2200','pctsecondsdurationdays1time2200to2300','pctsecondsdurationdays1time2300to2400','pctsecondsdurationdays3time0000to0100','pctsecondsdurationdays3time0100to0200','pctsecondsdurationdays3time0200to0300','pctsecondsdurationdays3time0300to0400','pctsecondsdurationdays2time2100to2200','pctsecondsdurationdays2time2200to2300','pctsecondsdurationdays2time2300to2400','pctsecondsdurationdays4time0000to0100','pctsecondsdurationdays4time0100to0200','pctsecondsdurationdays4time0200to0300','pctsecondsdurationdays4time0300to0400','pctsecondsdurationdays3time2100to2200','pctsecondsdurationdays3time2200to2300','pctsecondsdurationdays3time2300to2400','pctsecondsdurationdays5time0000to0100','pctsecondsdurationdays5time0100to0200','pctsecondsdurationdays5time0200to0300','pctsecondsdurationdays5time0300to0400','pctsecondsdurationdays4time2100to2200','pctsecondsdurationdays4time2200to2300','pctsecondsdurationdays4time2300to2400','pctsecondsdurationdays5time1900to2000','pctsecondsdurationdays5time2000to2100','pctsecondsdurationdays5time2100to2200','pctsecondsdurationdays6time1900to2000','pctsecondsdurationdays6time2000to2100','pctsecondsdurationdays6time2100to2200','pctsecondsdurationdays6time0000to0100','pctsecondsdurationdays6time0100to0200','pctsecondsdurationdays6time0200to0300','pctsecondsdurationdays6time0300to0400','pctsecondsdurationdays6time0400to0500','pctsecondsdurationdays6time0500to0600','pctsecondsdurationdays5time2200to2300','pctsecondsdurationdays5time2300to2400','pctsecondsdurationdays7time0000to0100','pctsecondsdurationdays7time0100to0200','pctsecondsdurationdays7time0200to0300','pctsecondsdurationdays7time0300to0400','pctsecondsdurationdays7time0400to0500','pctsecondsdurationdays7time0500to0600','pctsecondsdurationdays6time2200to2300','pctsecondsdurationdays6time2300to2400', 
'secondsaccmeterpersecsquaredge35', 'secondsaccmeterpersecsquaredge40', 'secondsaccmeterpersecsquaredge45', 'secondsaccmeterpersecsquaredge50', 'secondsbrakemeterpersecsquaredge35','secondsbrakemeterpersecsquaredge40', 'secondsbrakemeterpersecsquaredge45', 'secondsbrakemeterpersecsquaredge50', 'meterstotaldistance', 'secondsovermph80', 'metersovermph80',  'secondsdurationtotaltime', 'DistancePerTrip', 'secondsdurationdays7time1800to1900','secondsdurationdays7time1900to2000','secondsdurationdays7time2000to2100','secondsdurationdays1time1800to1900','secondsdurationdays1time1900to2000',           'secondsdurationdays1time2000to2100','secondsdurationdays2time1800to1900','secondsdurationdays2time1900to2000','secondsdurationdays2time2000to2100','secondsdurationdays3time1800to1900','secondsdurationdays3time1900to2000','secondsdurationdays3time2000to2100','secondsdurationdays4time1800to1900','secondsdurationdays4time1900to2000','secondsdurationdays4time2000to2100','secondsdurationdays1time0000to0100','secondsdurationdays1time0100to0200','secondsdurationdays1time0200to0300','secondsdurationdays1time0300to0400','secondsdurationdays7time2100to2200','secondsdurationdays7time2200to2300','secondsdurationdays7time2300to2400','secondsdurationdays2time0000to0100','secondsdurationdays2time0100to0200','secondsdurationdays2time0200to0300','secondsdurationdays2time0300to0400','secondsdurationdays1time2100to2200','secondsdurationdays1time2200to2300','secondsdurationdays1time2300to2400','secondsdurationdays3time0000to0100','secondsdurationdays3time0100to0200','secondsdurationdays3time0200to0300','secondsdurationdays3time0300to0400','secondsdurationdays2time2100to2200','secondsdurationdays2time2200to2300','secondsdurationdays2time2300to2400','secondsdurationdays4time0000to0100','secondsdurationdays4time0100to0200','secondsdurationdays4time0200to0300','secondsdurationdays4time0300to0400','secondsdurationdays3time2100to2200','secondsdurationdays3time2200to2300','secondsdurationdays3time2300to2400','secondsdurationdays5time0000to0100','secondsdurationdays5time0100to0200','secondsdurationdays5time0200to0300','secondsdurationdays5time0300to0400','secondsdurationdays4time2100to2200','secondsdurationdays4time2200to2300','secondsdurationdays4time2300to2400','secondsdurationdays5time1900to2000','secondsdurationdays5time2000to2100','secondsdurationdays5time2100to2200','secondsdurationdays6time1900to2000','secondsdurationdays6time2000to2100','secondsdurationdays6time2100to2200','secondsdurationdays6time0000to0100','secondsdurationdays6time0100to0200','secondsdurationdays6time0200to0300','secondsdurationdays6time0300to0400','secondsdurationdays6time0400to0500','secondsdurationdays6time0500to0600','secondsdurationdays5time2200to2300','secondsdurationdays5time2300to2400','secondsdurationdays7time0000to0100','secondsdurationdays7time0100to0200','secondsdurationdays7time0200to0300','secondsdurationdays7time0300to0400','secondsdurationdays7time0400to0500','secondsdurationdays7time0500to0600','secondsdurationdays6time2200to2300','secondsdurationdays6time2300to2400', 'metersbtwfirstandlasttrip', 'daysbtwfirstandlasttripinclusive')

data3 = spark.read.options(delimiter="|", header='true', inferschema='true').csv("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/archive_201903_normalized.csv")
data3 = data3.select('vin','counttotaltrips','score_dt','idl','train_flag',"at_fault_bipd_all_cnt","exposure","at_fault_bipd_all_amt","bi_all_amt","bi_all_cnt","co_amt","co_cnt","first_claim_date","garage_zip","pd_all_amt","pd_all_cnt","pd_amt","pd_cnt","meanavespeed","distanceperday","permetercountaccmeterpersecsquaredge40","permetercountbrakemeterpersecsquaredge40","pctsecondsovermph80","aveovermph80speed",'pctsecondsdurationdays7time1800to1900','pctsecondsdurationdays7time1900to2000','pctsecondsdurationdays7time2000to2100','pctsecondsdurationdays1time1800to1900','pctsecondsdurationdays1time1900to2000',           'pctsecondsdurationdays1time2000to2100','pctsecondsdurationdays2time1800to1900','pctsecondsdurationdays2time1900to2000','pctsecondsdurationdays2time2000to2100','pctsecondsdurationdays3time1800to1900','pctsecondsdurationdays3time1900to2000','pctsecondsdurationdays3time2000to2100','pctsecondsdurationdays4time1800to1900','pctsecondsdurationdays4time1900to2000','pctsecondsdurationdays4time2000to2100','pctsecondsdurationdays1time0000to0100','pctsecondsdurationdays1time0100to0200','pctsecondsdurationdays1time0200to0300','pctsecondsdurationdays1time0300to0400','pctsecondsdurationdays7time2100to2200','pctsecondsdurationdays7time2200to2300','pctsecondsdurationdays7time2300to2400','pctsecondsdurationdays2time0000to0100','pctsecondsdurationdays2time0100to0200','pctsecondsdurationdays2time0200to0300','pctsecondsdurationdays2time0300to0400','pctsecondsdurationdays1time2100to2200','pctsecondsdurationdays1time2200to2300','pctsecondsdurationdays1time2300to2400','pctsecondsdurationdays3time0000to0100','pctsecondsdurationdays3time0100to0200','pctsecondsdurationdays3time0200to0300','pctsecondsdurationdays3time0300to0400','pctsecondsdurationdays2time2100to2200','pctsecondsdurationdays2time2200to2300','pctsecondsdurationdays2time2300to2400','pctsecondsdurationdays4time0000to0100','pctsecondsdurationdays4time0100to0200','pctsecondsdurationdays4time0200to0300','pctsecondsdurationdays4time0300to0400','pctsecondsdurationdays3time2100to2200','pctsecondsdurationdays3time2200to2300','pctsecondsdurationdays3time2300to2400','pctsecondsdurationdays5time0000to0100','pctsecondsdurationdays5time0100to0200','pctsecondsdurationdays5time0200to0300','pctsecondsdurationdays5time0300to0400','pctsecondsdurationdays4time2100to2200','pctsecondsdurationdays4time2200to2300','pctsecondsdurationdays4time2300to2400','pctsecondsdurationdays5time1900to2000','pctsecondsdurationdays5time2000to2100','pctsecondsdurationdays5time2100to2200','pctsecondsdurationdays6time1900to2000','pctsecondsdurationdays6time2000to2100','pctsecondsdurationdays6time2100to2200','pctsecondsdurationdays6time0000to0100','pctsecondsdurationdays6time0100to0200','pctsecondsdurationdays6time0200to0300','pctsecondsdurationdays6time0300to0400','pctsecondsdurationdays6time0400to0500','pctsecondsdurationdays6time0500to0600','pctsecondsdurationdays5time2200to2300','pctsecondsdurationdays5time2300to2400','pctsecondsdurationdays7time0000to0100','pctsecondsdurationdays7time0100to0200','pctsecondsdurationdays7time0200to0300','pctsecondsdurationdays7time0300to0400','pctsecondsdurationdays7time0400to0500','pctsecondsdurationdays7time0500to0600','pctsecondsdurationdays6time2200to2300','pctsecondsdurationdays6time2300to2400', 
'secondsaccmeterpersecsquaredge35', 'secondsaccmeterpersecsquaredge40', 'secondsaccmeterpersecsquaredge45', 'secondsaccmeterpersecsquaredge50', 'secondsbrakemeterpersecsquaredge35','secondsbrakemeterpersecsquaredge40', 'secondsbrakemeterpersecsquaredge45', 'secondsbrakemeterpersecsquaredge50', 'meterstotaldistance', 'secondsovermph80', 'metersovermph80',  'secondsdurationtotaltime', 'DistancePerTrip', 'secondsdurationdays7time1800to1900','secondsdurationdays7time1900to2000','secondsdurationdays7time2000to2100','secondsdurationdays1time1800to1900','secondsdurationdays1time1900to2000',           'secondsdurationdays1time2000to2100','secondsdurationdays2time1800to1900','secondsdurationdays2time1900to2000','secondsdurationdays2time2000to2100','secondsdurationdays3time1800to1900','secondsdurationdays3time1900to2000','secondsdurationdays3time2000to2100','secondsdurationdays4time1800to1900','secondsdurationdays4time1900to2000','secondsdurationdays4time2000to2100','secondsdurationdays1time0000to0100','secondsdurationdays1time0100to0200','secondsdurationdays1time0200to0300','secondsdurationdays1time0300to0400','secondsdurationdays7time2100to2200','secondsdurationdays7time2200to2300','secondsdurationdays7time2300to2400','secondsdurationdays2time0000to0100','secondsdurationdays2time0100to0200','secondsdurationdays2time0200to0300','secondsdurationdays2time0300to0400','secondsdurationdays1time2100to2200','secondsdurationdays1time2200to2300','secondsdurationdays1time2300to2400','secondsdurationdays3time0000to0100','secondsdurationdays3time0100to0200','secondsdurationdays3time0200to0300','secondsdurationdays3time0300to0400','secondsdurationdays2time2100to2200','secondsdurationdays2time2200to2300','secondsdurationdays2time2300to2400','secondsdurationdays4time0000to0100','secondsdurationdays4time0100to0200','secondsdurationdays4time0200to0300','secondsdurationdays4time0300to0400','secondsdurationdays3time2100to2200','secondsdurationdays3time2200to2300','secondsdurationdays3time2300to2400','secondsdurationdays5time0000to0100','secondsdurationdays5time0100to0200','secondsdurationdays5time0200to0300','secondsdurationdays5time0300to0400','secondsdurationdays4time2100to2200','secondsdurationdays4time2200to2300','secondsdurationdays4time2300to2400','secondsdurationdays5time1900to2000','secondsdurationdays5time2000to2100','secondsdurationdays5time2100to2200','secondsdurationdays6time1900to2000','secondsdurationdays6time2000to2100','secondsdurationdays6time2100to2200','secondsdurationdays6time0000to0100','secondsdurationdays6time0100to0200','secondsdurationdays6time0200to0300','secondsdurationdays6time0300to0400','secondsdurationdays6time0400to0500','secondsdurationdays6time0500to0600','secondsdurationdays5time2200to2300','secondsdurationdays5time2300to2400','secondsdurationdays7time0000to0100','secondsdurationdays7time0100to0200','secondsdurationdays7time0200to0300','secondsdurationdays7time0300to0400','secondsdurationdays7time0400to0500','secondsdurationdays7time0500to0600','secondsdurationdays6time2200to2300','secondsdurationdays6time2300to2400', 'metersbtwfirstandlasttrip', 'daysbtwfirstandlasttripinclusive')

data4 = spark.read.options(delimiter="|", header='true', inferschema='true').csv("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/archive_201909_correct_normalized.csv")
data4 = data4.select('vin','counttotaltrips','score_dt','idl','train_flag',"at_fault_bipd_all_cnt","exposure","at_fault_bipd_all_amt","bi_all_amt","bi_all_cnt","co_amt","co_cnt","first_claim_date","garage_zip","pd_all_amt","pd_all_cnt","pd_amt","pd_cnt","meanavespeed","distanceperday","permetercountaccmeterpersecsquaredge40","permetercountbrakemeterpersecsquaredge40","pctsecondsovermph80","aveovermph80speed",'pctsecondsdurationdays7time1800to1900','pctsecondsdurationdays7time1900to2000','pctsecondsdurationdays7time2000to2100','pctsecondsdurationdays1time1800to1900','pctsecondsdurationdays1time1900to2000',           'pctsecondsdurationdays1time2000to2100','pctsecondsdurationdays2time1800to1900','pctsecondsdurationdays2time1900to2000','pctsecondsdurationdays2time2000to2100','pctsecondsdurationdays3time1800to1900','pctsecondsdurationdays3time1900to2000','pctsecondsdurationdays3time2000to2100','pctsecondsdurationdays4time1800to1900','pctsecondsdurationdays4time1900to2000','pctsecondsdurationdays4time2000to2100','pctsecondsdurationdays1time0000to0100','pctsecondsdurationdays1time0100to0200','pctsecondsdurationdays1time0200to0300','pctsecondsdurationdays1time0300to0400','pctsecondsdurationdays7time2100to2200','pctsecondsdurationdays7time2200to2300','pctsecondsdurationdays7time2300to2400','pctsecondsdurationdays2time0000to0100','pctsecondsdurationdays2time0100to0200','pctsecondsdurationdays2time0200to0300','pctsecondsdurationdays2time0300to0400','pctsecondsdurationdays1time2100to2200','pctsecondsdurationdays1time2200to2300','pctsecondsdurationdays1time2300to2400','pctsecondsdurationdays3time0000to0100','pctsecondsdurationdays3time0100to0200','pctsecondsdurationdays3time0200to0300','pctsecondsdurationdays3time0300to0400','pctsecondsdurationdays2time2100to2200','pctsecondsdurationdays2time2200to2300','pctsecondsdurationdays2time2300to2400','pctsecondsdurationdays4time0000to0100','pctsecondsdurationdays4time0100to0200','pctsecondsdurationdays4time0200to0300','pctsecondsdurationdays4time0300to0400','pctsecondsdurationdays3time2100to2200','pctsecondsdurationdays3time2200to2300','pctsecondsdurationdays3time2300to2400','pctsecondsdurationdays5time0000to0100','pctsecondsdurationdays5time0100to0200','pctsecondsdurationdays5time0200to0300','pctsecondsdurationdays5time0300to0400','pctsecondsdurationdays4time2100to2200','pctsecondsdurationdays4time2200to2300','pctsecondsdurationdays4time2300to2400','pctsecondsdurationdays5time1900to2000','pctsecondsdurationdays5time2000to2100','pctsecondsdurationdays5time2100to2200','pctsecondsdurationdays6time1900to2000','pctsecondsdurationdays6time2000to2100','pctsecondsdurationdays6time2100to2200','pctsecondsdurationdays6time0000to0100','pctsecondsdurationdays6time0100to0200','pctsecondsdurationdays6time0200to0300','pctsecondsdurationdays6time0300to0400','pctsecondsdurationdays6time0400to0500','pctsecondsdurationdays6time0500to0600','pctsecondsdurationdays5time2200to2300','pctsecondsdurationdays5time2300to2400','pctsecondsdurationdays7time0000to0100','pctsecondsdurationdays7time0100to0200','pctsecondsdurationdays7time0200to0300','pctsecondsdurationdays7time0300to0400','pctsecondsdurationdays7time0400to0500','pctsecondsdurationdays7time0500to0600','pctsecondsdurationdays6time2200to2300','pctsecondsdurationdays6time2300to2400', 
'secondsaccmeterpersecsquaredge35', 'secondsaccmeterpersecsquaredge40', 'secondsaccmeterpersecsquaredge45', 'secondsaccmeterpersecsquaredge50', 'secondsbrakemeterpersecsquaredge35','secondsbrakemeterpersecsquaredge40', 'secondsbrakemeterpersecsquaredge45', 'secondsbrakemeterpersecsquaredge50', 'meterstotaldistance', 'secondsovermph80', 'metersovermph80',  'secondsdurationtotaltime', 'DistancePerTrip', 'secondsdurationdays7time1800to1900','secondsdurationdays7time1900to2000','secondsdurationdays7time2000to2100','secondsdurationdays1time1800to1900','secondsdurationdays1time1900to2000',           'secondsdurationdays1time2000to2100','secondsdurationdays2time1800to1900','secondsdurationdays2time1900to2000','secondsdurationdays2time2000to2100','secondsdurationdays3time1800to1900','secondsdurationdays3time1900to2000','secondsdurationdays3time2000to2100','secondsdurationdays4time1800to1900','secondsdurationdays4time1900to2000','secondsdurationdays4time2000to2100','secondsdurationdays1time0000to0100','secondsdurationdays1time0100to0200','secondsdurationdays1time0200to0300','secondsdurationdays1time0300to0400','secondsdurationdays7time2100to2200','secondsdurationdays7time2200to2300','secondsdurationdays7time2300to2400','secondsdurationdays2time0000to0100','secondsdurationdays2time0100to0200','secondsdurationdays2time0200to0300','secondsdurationdays2time0300to0400','secondsdurationdays1time2100to2200','secondsdurationdays1time2200to2300','secondsdurationdays1time2300to2400','secondsdurationdays3time0000to0100','secondsdurationdays3time0100to0200','secondsdurationdays3time0200to0300','secondsdurationdays3time0300to0400','secondsdurationdays2time2100to2200','secondsdurationdays2time2200to2300','secondsdurationdays2time2300to2400','secondsdurationdays4time0000to0100','secondsdurationdays4time0100to0200','secondsdurationdays4time0200to0300','secondsdurationdays4time0300to0400','secondsdurationdays3time2100to2200','secondsdurationdays3time2200to2300','secondsdurationdays3time2300to2400','secondsdurationdays5time0000to0100','secondsdurationdays5time0100to0200','secondsdurationdays5time0200to0300','secondsdurationdays5time0300to0400','secondsdurationdays4time2100to2200','secondsdurationdays4time2200to2300','secondsdurationdays4time2300to2400','secondsdurationdays5time1900to2000','secondsdurationdays5time2000to2100','secondsdurationdays5time2100to2200','secondsdurationdays6time1900to2000','secondsdurationdays6time2000to2100','secondsdurationdays6time2100to2200','secondsdurationdays6time0000to0100','secondsdurationdays6time0100to0200','secondsdurationdays6time0200to0300','secondsdurationdays6time0300to0400','secondsdurationdays6time0400to0500','secondsdurationdays6time0500to0600','secondsdurationdays5time2200to2300','secondsdurationdays5time2300to2400','secondsdurationdays7time0000to0100','secondsdurationdays7time0100to0200','secondsdurationdays7time0200to0300','secondsdurationdays7time0300to0400','secondsdurationdays7time0400to0500','secondsdurationdays7time0500to0600','secondsdurationdays6time2200to2300','secondsdurationdays6time2300to2400', 'metersbtwfirstandlasttrip', 'daysbtwfirstandlasttripinclusive')
def unionAll(dfs):
    return functools.reduce(lambda df1, df2: df1.union(df2.select(df1.columns)), dfs)
data_all = unionAll([data1, data2, data3, data4])

# COMMAND ----------

# cnt = data_all.count()
# print(cnt)
# data_all = data_all.withColumn('KMDistanceTraveled', data_all.meterstotaldistance / 1000)
# data_all = data_all.filter(data_all.KMDistanceTraveled < 10000) 
# data_all = data_all.filter(data_all.KMDistanceTraveled > 100)
# cnt_ = data_all.count()
# print(cnt_)
# print(cnt_/cnt)

# COMMAND ----------

data_all = data_all.withColumn('KMDistanceTraveled', data_all.meterstotaldistance / 1000)
#create Per KM attrs
data_all = data_all.withColumn('PerKMSecondsAccMeterPerSecSquaredGe35', data_all.secondsaccmeterpersecsquaredge35 * 1000/ data_all.KMDistanceTraveled)
data_all = data_all.withColumn('PerKMSecondsAccMeterPerSecSquaredGe40', data_all.secondsaccmeterpersecsquaredge40 * 1000/ data_all.KMDistanceTraveled)
data_all = data_all.withColumn('PerKMSecondsAccMeterPerSecSquaredGe45', data_all.secondsaccmeterpersecsquaredge45 * 1000/ data_all.KMDistanceTraveled)
data_all = data_all.withColumn('PerKMSecondsAccMeterPerSecSquaredGe50', data_all.secondsaccmeterpersecsquaredge50 * 1000/ data_all.KMDistanceTraveled)

data_all = data_all.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe35', data_all.secondsbrakemeterpersecsquaredge35 * 1000/ data_all.KMDistanceTraveled)
data_all = data_all.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe40', data_all.secondsbrakemeterpersecsquaredge40 * 1000/ data_all.KMDistanceTraveled)
data_all = data_all.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe45', data_all.secondsbrakemeterpersecsquaredge45 * 1000/ data_all.KMDistanceTraveled)
data_all = data_all.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe50', data_all.secondsbrakemeterpersecsquaredge50 * 1000/ data_all.KMDistanceTraveled)

data_all = data_all.withColumn('PerKMSecondsOverMph80', data_all.secondsovermph80 / data_all.KMDistanceTraveled)

data_all = data_all.withColumn('PerKMMetersOverMph80', data_all.metersovermph80 / data_all.KMDistanceTraveled)

# COMMAND ----------

# MAGIC %md # Pred Acc3.61 using Acc3.5 & Acc4.0

# COMMAND ----------

modelAcc361_Large = LinearRegressionModel.load("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/Acc361_Large/")
modelAcc361_Small = LinearRegressionModel.load("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/Acc361_Small/")

# COMMAND ----------

# cnt = data_all.count()

# COMMAND ----------

# zero % 0.16519949592270022
# match % 0.026372169147259327
# Large % 0.028516962891003514
# Small % 0.9714830371089965
# out range % 0.048124739679623676

#Rule based set 1
data_zero = data_all.filter(data_all.PerKMSecondsAccMeterPerSecSquaredGe35==0)
data = data_all.filter(data_all.PerKMSecondsAccMeterPerSecSquaredGe35>0)
# print('zero % ' + str(data_zero.count()/cnt))
#Rule based set 2
data_match = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe35==data.PerKMSecondsAccMeterPerSecSquaredGe40)
data = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe35 != data.PerKMSecondsAccMeterPerSecSquaredGe40)
# print('match % ' + str(data_match.count()/cnt))
#2 Model based set
data_Large = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe35 > 42.3)
data_Small = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe35 <= 42.3)

# cnt_modeling = data.count()
# print('Large % ' + str(data_Large.count()/cnt_modeling))
# print('Small % ' + str(data_Small.count()/cnt_modeling))
#Assembler
assembler=VectorAssembler().setInputCols(['PerKMSecondsAccMeterPerSecSquaredGe35','PerKMSecondsAccMeterPerSecSquaredGe40']).setOutputCol('features')
#Model Prediction for Large value set
data_Large = assembler.transform(data_Large)
# data_Large = data_Large.select('features')
data_Large = modelAcc361_Large.transform(data_Large)
#Model Prediction for Small value set
data_Small = assembler.transform(data_Small)
# data_Small = data_Small.select('features')
data_Small = modelAcc361_Small.transform(data_Small)

data_Large_Small = unionAll([data_Large, data_Small])
# if prediction out of range
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
first=F.udf(lambda v:float(v[0]),DoubleType())
second=F.udf(lambda v:float(v[1]),DoubleType())
data_Large_Small = data_Large_Small.withColumn("ceiling", first("features")).withColumn("floor", second("features"))

# print('out range % ' + str(data_Large_Small.filter((data_Large_Small.prediction > data_Large_Small.ceiling) | (data_Large_Small.prediction < data_Large_Small.floor)).count() / data_Large_Small.count()))

from pyspark.sql.functions import when

data_Large_Small = data_Large_Small.withColumn("prediction_temp", \
              when(data_Large_Small["prediction"] > data_Large_Small["ceiling"], data_Large_Small["ceiling"]).otherwise(data_Large_Small["prediction"]))
data_Large_Small = data_Large_Small.withColumn("prediction_temp", \
              when(data_Large_Small["prediction"] < data_Large_Small["floor"], data_Large_Small["floor"]).otherwise(data_Large_Small["prediction_temp"]))
data_Large_Small = data_Large_Small.drop('prediction')
data_Large_Small = data_Large_Small.withColumnRenamed('prediction_temp', 'prediction')
data_Large_Small = data_Large_Small.drop('features', 'ceiling', 'floor')

#Combine Rule based set and Model based set
data_zero = data_zero.withColumn("prediction", data_zero.PerKMSecondsAccMeterPerSecSquaredGe35)
data_match = data_match.withColumn("prediction", data_match.PerKMSecondsAccMeterPerSecSquaredGe35)
data_back1 = unionAll([data_zero, data_match])

data_all_predicted = unionAll([data_Large_Small, data_back1])
data_all_predicted = data_all_predicted.withColumnRenamed('prediction', 'PerKMSecondsAccMeterPerSecSquaredGe361')
# data_all_predicted.persist()

# COMMAND ----------

# MAGIC %md # Pred Acc4.72 using Acc4.5 & Acc5.0

# COMMAND ----------

modelAcc472_Large = LinearRegressionModel.load("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/Acc472_Large/")
modelAcc472_Small = LinearRegressionModel.load("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/Acc472_Small/")

# COMMAND ----------

# zero % 0.5616147315798286
# match % 0.0654485624186992
# Large % 0.021804805981849428
# Small % 0.9781951940181506
# out range % 0.008000589608814652

#Rule based set 1
data_zero = data_all_predicted.filter(data_all_predicted.PerKMSecondsAccMeterPerSecSquaredGe45==0)
data = data_all_predicted.filter(data_all_predicted.PerKMSecondsAccMeterPerSecSquaredGe45>0)
# print('zero % ' + str(data_zero.count()/cnt))
#Rule based set 2
data_match = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe45==data.PerKMSecondsAccMeterPerSecSquaredGe50)
data = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe45 != data.PerKMSecondsAccMeterPerSecSquaredGe50)
# print('match % ' + str(data_match.count()/cnt))
#2 Model based set
data_Large = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe50 > 11.05)
data_Small = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe50 <= 11.05)

# cnt_modeling = data.count()
# print('Large % ' + str(data_Large.count()/cnt_modeling))
# print('Small % ' + str(data_Small.count()/cnt_modeling))
#Assembler
assembler=VectorAssembler().setInputCols(['PerKMSecondsAccMeterPerSecSquaredGe45','PerKMSecondsAccMeterPerSecSquaredGe50']).setOutputCol('features')
#Model Prediction for Large value set
data_Large = assembler.transform(data_Large)
# data_Large = data_Large.select('features')
data_Large = modelAcc472_Large.transform(data_Large)
#Model Prediction for Small value set
data_Small = assembler.transform(data_Small)
# data_Small = data_Small.select('features')
data_Small = modelAcc472_Small.transform(data_Small)

data_Large_Small = unionAll([data_Large, data_Small])
# if prediction out of range
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
first=F.udf(lambda v:float(v[0]),DoubleType())
second=F.udf(lambda v:float(v[1]),DoubleType())
data_Large_Small = data_Large_Small.withColumn("ceiling", first("features")).withColumn("floor", second("features"))

# print('out range % ' + str(data_Large_Small.filter((data_Large_Small.prediction > data_Large_Small.ceiling) | (data_Large_Small.prediction < data_Large_Small.floor)).count() / data_Large_Small.count()))

from pyspark.sql.functions import when

data_Large_Small = data_Large_Small.withColumn("prediction_temp", \
              when(data_Large_Small["prediction"] > data_Large_Small["ceiling"], data_Large_Small["ceiling"]).otherwise(data_Large_Small["prediction"]))
data_Large_Small = data_Large_Small.withColumn("prediction_temp", \
              when(data_Large_Small["prediction"] < data_Large_Small["floor"], data_Large_Small["floor"]).otherwise(data_Large_Small["prediction_temp"]))
data_Large_Small = data_Large_Small.drop('prediction')
data_Large_Small = data_Large_Small.withColumnRenamed('prediction_temp', 'prediction')
data_Large_Small = data_Large_Small.drop('features', 'ceiling', 'floor')

#Combine Rule based set and Model based set
data_zero = data_zero.withColumn("prediction", data_zero.PerKMSecondsAccMeterPerSecSquaredGe45)
data_match = data_match.withColumn("prediction", data_match.PerKMSecondsAccMeterPerSecSquaredGe45)
data_back1 = unionAll([data_zero, data_match])

data_all_predicted = unionAll([data_Large_Small, data_back1])
data_all_predicted = data_all_predicted.withColumnRenamed('prediction', 'PerKMSecondsAccMeterPerSecSquaredGe472')
# data_all_predicted.persist()

# COMMAND ----------

# MAGIC %md # Pred Brake3.61 using Brake3.5 & Brake4.0

# COMMAND ----------

modelBrake361_Large = LinearRegressionModel.load("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/Brake361_Large/")
modelBrake361_Small = LinearRegressionModel.load("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/Brake361_Small/")

# COMMAND ----------

# zero % 0.008192494518852867
# match % 0.001484343188665394
# Large % 0.03867160427160708
# Small % 0.9613283957283929
# out range % 0.0001741444214315073

#Rule based set 1
data_zero = data_all_predicted.filter(data_all_predicted.PerKMSecondsBrakeMeterPerSecSquaredGe35==0)
data = data_all_predicted.filter(data_all_predicted.PerKMSecondsBrakeMeterPerSecSquaredGe35>0)
# print('zero % ' + str(data_zero.count()/cnt))
#Rule based set 2
data_match = data.filter(data.PerKMSecondsBrakeMeterPerSecSquaredGe35==data.PerKMSecondsBrakeMeterPerSecSquaredGe40)
data = data.filter(data.PerKMSecondsBrakeMeterPerSecSquaredGe35 != data.PerKMSecondsBrakeMeterPerSecSquaredGe40)
# print('match % ' + str(data_match.count()/cnt))
#2 Model based set
data_Large = data.filter(data.PerKMSecondsBrakeMeterPerSecSquaredGe35 > 60.18)
data_Small = data.filter(data.PerKMSecondsBrakeMeterPerSecSquaredGe35 <= 60.18)

# cnt_modeling = data.count()
# print('Large % ' + str(data_Large.count()/cnt_modeling))
# print('Small % ' + str(data_Small.count()/cnt_modeling))
#Assembler
assembler=VectorAssembler().setInputCols(['PerKMSecondsBrakeMeterPerSecSquaredGe35','PerKMSecondsBrakeMeterPerSecSquaredGe40']).setOutputCol('features')
#Model Prediction for Large value set
data_Large = assembler.transform(data_Large)
# data_Large = data_Large.select('features')
data_Large = modelBrake361_Large.transform(data_Large)
#Model Prediction for Small value set
data_Small = assembler.transform(data_Small)
# data_Small = data_Small.select('features')
data_Small = modelBrake361_Small.transform(data_Small)

data_Large_Small = unionAll([data_Large, data_Small])
# if prediction out of range
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
first=F.udf(lambda v:float(v[0]),DoubleType())
second=F.udf(lambda v:float(v[1]),DoubleType())
data_Large_Small = data_Large_Small.withColumn("ceiling", first("features")).withColumn("floor", second("features"))

# print('out range % ' + str(data_Large_Small.filter((data_Large_Small.prediction > data_Large_Small.ceiling) | (data_Large_Small.prediction < data_Large_Small.floor)).count() / data_Large_Small.count()))

from pyspark.sql.functions import when

data_Large_Small = data_Large_Small.withColumn("prediction_temp", \
              when(data_Large_Small["prediction"] > data_Large_Small["ceiling"], data_Large_Small["ceiling"]).otherwise(data_Large_Small["prediction"]))
data_Large_Small = data_Large_Small.withColumn("prediction_temp", \
              when(data_Large_Small["prediction"] < data_Large_Small["floor"], data_Large_Small["floor"]).otherwise(data_Large_Small["prediction_temp"]))
data_Large_Small = data_Large_Small.drop('prediction')
data_Large_Small = data_Large_Small.withColumnRenamed('prediction_temp', 'prediction')
data_Large_Small = data_Large_Small.drop('features', 'ceiling', 'floor')

#Combine Rule based set and Model based set
data_zero = data_zero.withColumn("prediction", data_zero.PerKMSecondsBrakeMeterPerSecSquaredGe35)
data_match = data_match.withColumn("prediction", data_match.PerKMSecondsBrakeMeterPerSecSquaredGe35)
data_back1 = unionAll([data_zero, data_match])

data_all_predicted = unionAll([data_Large_Small, data_back1])
data_all_predicted = data_all_predicted.withColumnRenamed('prediction', 'PerKMSecondsBrakeMeterPerSecSquaredGe361')
# data_all_predicted.persist()

# COMMAND ----------

data_all_predicted.write.parquet("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/Modeling_Set_temp_updated.parquet")

# COMMAND ----------

df = spark.read.parquet('dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/Modeling_Set_temp_updated.parquet')

# COMMAND ----------

# MAGIC %md # Pred Brake4.72 using Brake4.5 & Brake5.0

# COMMAND ----------

modelBrake472_Large = LinearRegressionModel.load("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/Brake472_Large/")
modelBrake472_Small = LinearRegressionModel.load("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/Brake472_Small/")

# COMMAND ----------

# zero % 0.09949904346180634
# match % 0.03475364311763091
# Large % 0.0036229699048826816
# Small % 0.9963770300951174
# out range % 0.0010863740607301193

#Rule based set 1
data_zero = df.filter(df.PerKMSecondsBrakeMeterPerSecSquaredGe45==0)
data = df.filter(df.PerKMSecondsBrakeMeterPerSecSquaredGe45>0)
# print('zero % ' + str(data_zero.count()/cnt))
#Rule based set 2
data_match = data.filter(data.PerKMSecondsBrakeMeterPerSecSquaredGe45==data.PerKMSecondsBrakeMeterPerSecSquaredGe50)
data = data.filter(data.PerKMSecondsBrakeMeterPerSecSquaredGe45 != data.PerKMSecondsBrakeMeterPerSecSquaredGe50)
# print('match % ' + str(data_match.count()/cnt))
#2 Model based set
data_Large = data.filter(data.PerKMSecondsBrakeMeterPerSecSquaredGe50 > 10.69)
data_Small = data.filter(data.PerKMSecondsBrakeMeterPerSecSquaredGe50 <= 10.69)

# cnt_modeling = data.count()
# print('Large % ' + str(data_Large.count()/cnt_modeling))
# print('Small % ' + str(data_Small.count()/cnt_modeling))
#Assembler
assembler=VectorAssembler().setInputCols(['PerKMSecondsBrakeMeterPerSecSquaredGe45','PerKMSecondsBrakeMeterPerSecSquaredGe50']).setOutputCol('features')
#Model Prediction for Large value set
data_Large = assembler.transform(data_Large)
# data_Large = data_Large.select('features')
data_Large = modelBrake472_Large.transform(data_Large)
#Model Prediction for Small value set
data_Small = assembler.transform(data_Small)
# data_Small = data_Small.select('features')
data_Small = modelBrake472_Small.transform(data_Small)

data_Large_Small = unionAll([data_Large, data_Small])
# if prediction out of range
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
first=F.udf(lambda v:float(v[0]),DoubleType())
second=F.udf(lambda v:float(v[1]),DoubleType())
data_Large_Small = data_Large_Small.withColumn("ceiling", first("features")).withColumn("floor", second("features"))

# print('out range % ' + str(data_Large_Small.filter((data_Large_Small.prediction > data_Large_Small.ceiling) | (data_Large_Small.prediction < data_Large_Small.floor)).count() / data_Large_Small.count()))

from pyspark.sql.functions import when

data_Large_Small = data_Large_Small.withColumn("prediction_temp", \
              when(data_Large_Small["prediction"] > data_Large_Small["ceiling"], data_Large_Small["ceiling"]).otherwise(data_Large_Small["prediction"]))
data_Large_Small = data_Large_Small.withColumn("prediction_temp", \
              when(data_Large_Small["prediction"] < data_Large_Small["floor"], data_Large_Small["floor"]).otherwise(data_Large_Small["prediction_temp"]))
data_Large_Small = data_Large_Small.drop('prediction')
data_Large_Small = data_Large_Small.withColumnRenamed('prediction_temp', 'prediction')
data_Large_Small = data_Large_Small.drop('features', 'ceiling', 'floor')

#Combine Rule based set and Model based set
data_zero = data_zero.withColumn("prediction", data_zero.PerKMSecondsBrakeMeterPerSecSquaredGe45)
data_match = data_match.withColumn("prediction", data_match.PerKMSecondsBrakeMeterPerSecSquaredGe45)
data_back1 = unionAll([data_zero, data_match])

data_all_predicted = unionAll([data_Large_Small, data_back1])
data_all_predicted = data_all_predicted.withColumnRenamed('prediction', 'PerKMSecondsBrakeMeterPerSecSquaredGe472')
# data_all_predicted.persist()

# COMMAND ----------

# MAGIC %md # Pred speeding

# COMMAND ----------

modelSecond = LinearRegressionModel.load("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/SecondOver75/")
modelMeter = LinearRegressionModel.load("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/MeterOver75/")

# COMMAND ----------

#Assembler
assembler=VectorAssembler().setInputCols(['PerKMSecondsOverMph80','DistancePerTrip', 'meanavespeed']).setOutputCol('features')
#Model Prediction
data_all_predicted = assembler.transform(data_all_predicted)
data_all_predicted = modelSecond.transform(data_all_predicted)
data_all_predicted = data_all_predicted.withColumnRenamed('prediction', 'PerKMSecondsOverMph75')
data_all_predicted = data_all_predicted.drop('features')

# COMMAND ----------

#Assembler
assembler=VectorAssembler().setInputCols(['PerKMMetersOverMph80','DistancePerTrip', 'meanavespeed']).setOutputCol('features')
#Model Prediction
data_all_predicted = assembler.transform(data_all_predicted)
data_all_predicted = modelMeter.transform(data_all_predicted)
data_all_predicted = data_all_predicted.withColumnRenamed('prediction', 'PerKMMetersOverMph75')
data_all_predicted = data_all_predicted.drop('features')

# COMMAND ----------

#data_all_predicted = data_all_predicted.withColumn("prediction_temp", \
#              when(data_all_predicted["prediction"] > data_all_predicted["ceiling"], data_all_predicted["ceiling"]).#otherwise(data_all_predicted["prediction"]))

# COMMAND ----------

data_all_predicted = data_all_predicted.withColumn('secondsaccmeterpersecsquaredge361',  ((data_all_predicted.PerKMSecondsAccMeterPerSecSquaredGe361 * data_all_predicted.KMDistanceTraveled) / 1000))

data_all_predicted = data_all_predicted.withColumn('secondsbrakemeterpersecsquaredge361',  ((data_all_predicted.PerKMSecondsBrakeMeterPerSecSquaredGe361 * data_all_predicted.KMDistanceTraveled) / 1000))
data_all_predicted = data_all_predicted.withColumn('secondsaccmeterpersecsquaredge472',  ((data_all_predicted.PerKMSecondsAccMeterPerSecSquaredGe472 * data_all_predicted.KMDistanceTraveled) / 1000))
data_all_predicted = data_all_predicted.withColumn('secondsbrakemeterpersecsquaredge472',  ((data_all_predicted.PerKMSecondsBrakeMeterPerSecSquaredGe472 * data_all_predicted.KMDistanceTraveled) / 1000))

data_all_predicted = data_all_predicted.withColumn('secondsovermph75',  (data_all_predicted.PerKMSecondsOverMph75 * data_all_predicted.KMDistanceTraveled))
data_all_predicted = data_all_predicted.withColumn('metersovermph75',  (data_all_predicted.PerKMMetersOverMph75 * data_all_predicted.KMDistanceTraveled))

# COMMAND ----------

# data_all_predicted.write.parquet("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/Modeling_Set.parquet")

data_all_predicted.coalesce(1).write.option("sep","|").option("header","true").csv('dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/Modeling_Set_Updated0418.csv')

# COMMAND ----------

data_all_predicted.columns
