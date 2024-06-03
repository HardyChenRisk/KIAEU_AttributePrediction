# Databricks notebook source
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql.functions import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.tree import DecisionTree
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.regression import DecisionTreeRegressor
import functools

# COMMAND ----------

# loading the KIA data
data_KIA = spark.read.options(header='true', inferschema='true').csv("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/KIAEU_KIA_EntityAttr_202211.csv")
data_KIA = data_KIA.fillna(0)
print(data_KIA.count())

data_KIA = data_KIA.withColumn('KMDistanceTraveled', data_KIA.MetersDistanceTraveled / 1000)
data_KIA = data_KIA.filter(data_KIA.KMDistanceTraveled < 10000) 
data_KIA = data_KIA.filter(data_KIA.KMDistanceTraveled > 100)
print(data_KIA.count())

#create Per KM attrs
data_KIA = data_KIA.withColumn('PerKMSecondsAccMeterPerSecSquaredGe30', data_KIA.SecondsAccMeterPerSecSquaredGe3 * 1000 / data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.withColumn('PerKMSecondsAccMeterPerSecSquaredGe35', data_KIA.SecondsAccMeterPerSecSquaredGe35 * 1000/ data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.withColumn('PerKMSecondsAccMeterPerSecSquaredGe361', data_KIA.SecondsAccMeterPerSecSquaredGe361 * 1000/ data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.withColumn('PerKMSecondsAccMeterPerSecSquaredGe40', data_KIA.SecondsAccMeterPerSecSquaredGe4 * 1000/ data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.withColumn('PerKMSecondsAccMeterPerSecSquaredGe45', data_KIA.SecondsAccMeterPerSecSquaredGe45 * 1000/ data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.withColumn('PerKMSecondsAccMeterPerSecSquaredGe472', data_KIA.SecondsAccMeterPerSecSquaredGe472 * 1000/ data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.withColumn('PerKMSecondsAccMeterPerSecSquaredGe50', data_KIA.SecondsAccMeterPerSecSquaredGe5 * 1000/ data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.select("PerKMSecondsAccMeterPerSecSquaredGe30", "PerKMSecondsAccMeterPerSecSquaredGe35", 'PerKMSecondsAccMeterPerSecSquaredGe361', 'PerKMSecondsAccMeterPerSecSquaredGe40', 'PerKMSecondsAccMeterPerSecSquaredGe45','PerKMSecondsAccMeterPerSecSquaredGe472', 'PerKMSecondsAccMeterPerSecSquaredGe50')

# COMMAND ----------

# loading the 2 months data
data1 = spark.read.options(header='true', inferschema='true').csv("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/KIAEU_GM_EntityAttr_202211.csv")
data2 = spark.read.options(header='true', inferschema='true').csv("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/KIAEU_GM_EntityAttr_202306.csv")

def unionAll(dfs):
    return functools.reduce(lambda df1, df2: df1.union(df2.select(df1.columns)), dfs)
data = unionAll([data1, data2])

print(data.count()) 
print(data.dropna().count())

# COMMAND ----------

# remove data with unrealistic distance Traveled
# keep 99.3%
data = data.withColumn('KMDistanceTraveled', data.MetersDistanceTraveled / 1000)
data = data.filter(data.KMDistanceTraveled < 10000) 
data = data.filter(data.KMDistanceTraveled > 100)
data.count()

# COMMAND ----------

#create Per KM attrs
data = data.withColumn('PerKMSecondsAccMeterPerSecSquaredGe30', data.SecondsAccMeterPerSecSquaredGe30 * 1000 / data.KMDistanceTraveled)
data = data.withColumn('PerKMSecondsAccMeterPerSecSquaredGe35', data.SecondsAccMeterPerSecSquaredGe35 * 1000/ data.KMDistanceTraveled)
data = data.withColumn('PerKMSecondsAccMeterPerSecSquaredGe361', data.SecondsAccMeterPerSecSquaredGe361 * 1000/ data.KMDistanceTraveled)
data = data.withColumn('PerKMSecondsAccMeterPerSecSquaredGe40', data.SecondsAccMeterPerSecSquaredGe40 * 1000/ data.KMDistanceTraveled)
data = data.withColumn('PerKMSecondsAccMeterPerSecSquaredGe45', data.SecondsAccMeterPerSecSquaredGe45 * 1000/ data.KMDistanceTraveled)
data = data.withColumn('PerKMSecondsAccMeterPerSecSquaredGe472', data.SecondsAccMeterPerSecSquaredGe472 * 1000/ data.KMDistanceTraveled)
data = data.withColumn('PerKMSecondsAccMeterPerSecSquaredGe50', data.SecondsAccMeterPerSecSquaredGe50 * 1000/ data.KMDistanceTraveled)
data = data.select("PerKMSecondsAccMeterPerSecSquaredGe30", "PerKMSecondsAccMeterPerSecSquaredGe35", 'PerKMSecondsAccMeterPerSecSquaredGe361', 'PerKMSecondsAccMeterPerSecSquaredGe40', 'PerKMSecondsAccMeterPerSecSquaredGe45','PerKMSecondsAccMeterPerSecSquaredGe472', 'PerKMSecondsAccMeterPerSecSquaredGe50')

# COMMAND ----------

# MAGIC %md # Acc 3.61 Modeling

# COMMAND ----------

#final sample size, keeps 91.4%
#data = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe35>= 10)
#data.count()

# COMMAND ----------

#remove data have zero second in SecondsAccMeterPerSecSquaredGe35
#48% records removed
data_35_zero = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe35==0)
print(data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe35 == 0).count() / data.count())
data = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe35>0)
data.count()

# COMMAND ----------

# 8.4% PerKMSecondsAccMeterPerSecSquaredGe35 == PerKMSecondsAccMeterPerSecSquaredGe40
data_35_40_match = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe35==data.PerKMSecondsAccMeterPerSecSquaredGe40)
print(data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe35 == data.PerKMSecondsAccMeterPerSecSquaredGe40).count() / data.count())
data = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe35 != data.PerKMSecondsAccMeterPerSecSquaredGe40)
data.count()

# COMMAND ----------

test_df_pandas = data.sample(fraction=0.05, seed=3).toPandas()
ax = test_df_pandas['PerKMSecondsAccMeterPerSecSquaredGe35'].plot.hist(bins=100, alpha=0.5)
ax

# COMMAND ----------

test_df_pandas = data.sample(fraction=0.05, seed=3).toPandas()
test_df_pandas.plot.scatter(x='PerKMSecondsAccMeterPerSecSquaredGe35', y='PerKMSecondsAccMeterPerSecSquaredGe361') 

# COMMAND ----------

data_Large = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe35 > 42.3)
test_df_pandas = data_Large.sample(fraction=0.05, seed=3).toPandas()
test_df_pandas.plot.scatter(x='PerKMSecondsAccMeterPerSecSquaredGe35', y='PerKMSecondsAccMeterPerSecSquaredGe361') 

# COMMAND ----------

test_df_pandas.plot.scatter(x='PerKMSecondsAccMeterPerSecSquaredGe40', y='PerKMSecondsAccMeterPerSecSquaredGe361') 

# COMMAND ----------

train_df, test_df = data.randomSplit(weights=[0.7,0.3], seed=100)
data_Large = train_df.filter(train_df.PerKMSecondsAccMeterPerSecSquaredGe35 > 42.3)
data_Small = train_df.filter(train_df.PerKMSecondsAccMeterPerSecSquaredGe35 <= 42.3)
print(data_Large.count())
print(data_Small.count())

data_Large_test = test_df.filter(test_df.PerKMSecondsAccMeterPerSecSquaredGe35 > 42.3)
data_Small_test = test_df.filter(test_df.PerKMSecondsAccMeterPerSecSquaredGe35 <= 42.3)

print(data_Large_test.count())
print(data_Small_test.count())

# COMMAND ----------

# Modeling with 2 attrs (PerKMSecondsAccMeterPerSecSquaredGe35 and PerKMSecondsAccMeterPerSecSquaredGe40)

# COMMAND ----------

# train_df = train_df.withColumnRenamed("PerKMSecondsAccMeterPerSecSquaredGe361", "label")
# assembler=VectorAssembler().setInputCols(['PerKMSecondsAccMeterPerSecSquaredGe35','PerKMSecondsAccMeterPerSecSquaredGe40']).setOutputCol('features')
# train_df = assembler.transform(train_df)
# train_df = train_df.select("features","label")
# dt = DecisionTreeRegressor(maxDepth=2)
# model = dt.fit(train_df)
# print(model.toDebugString)

# COMMAND ----------

data_Large = data_Large.withColumnRenamed("PerKMSecondsAccMeterPerSecSquaredGe361", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsAccMeterPerSecSquaredGe35','PerKMSecondsAccMeterPerSecSquaredGe40']).setOutputCol('features')
data_Large = assembler.transform(data_Large)
data_Large = data_Large.select("features","label")
lr = LinearRegression()
model_Large = lr.fit(data_Large)
print('intercept and coefficients')
print(model_Large.intercept, model_Large.coefficients)
print('pValues')
print(model_Large.summary.pValues)

data_Small = data_Small.withColumnRenamed("PerKMSecondsAccMeterPerSecSquaredGe361", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsAccMeterPerSecSquaredGe35','PerKMSecondsAccMeterPerSecSquaredGe40']).setOutputCol('features')
data_Small = assembler.transform(data_Small)
data_Small = data_Small.select("features","label")
lr = LinearRegression()
model_Small = lr.fit(data_Small)
print('intercept and coefficients')
print(model_Small.intercept, model_Small.coefficients)
print('pValues')
print(model_Small.summary.pValues)

lr_path1 = "dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/" + "Acc361_Large"
model_Large.save(lr_path1)

lr_path2 = "dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/" + "Acc361_Small"
model_Small.save(lr_path2)


# COMMAND ----------

data_Large_test = data_Large_test.withColumnRenamed("PerKMSecondsAccMeterPerSecSquaredGe361", "label")
data_Large_test = assembler.transform(data_Large_test)
data_Large_test = data_Large_test.select('features', 'label')
data_Large_test = model_Large.transform(data_Large_test)

data_Small_test = data_Small_test.withColumnRenamed("PerKMSecondsAccMeterPerSecSquaredGe361", "label")
data_Small_test = assembler.transform(data_Small_test)
data_Small_test = data_Small_test.select('features', 'label')
data_Small_test = model_Small.transform(data_Small_test)

# COMMAND ----------

print('data_Large_test')
test_df_pandas = data_Large_test.sample(fraction=0.05, seed=3).toPandas()
test_df_pandas.plot.scatter(x='label', y='prediction') 
print('data_Small_test')
test_df_pandas = data_Small_test.sample(fraction=0.001, seed=3).toPandas()
test_df_pandas.plot.scatter(x='label', y='prediction') 

# COMMAND ----------

test_df = unionAll([data_Large_test, data_Small_test])
evaluator = RegressionEvaluator()
print(evaluator.evaluate(test_df,
{evaluator.metricName: "r2"})
)
print(evaluator.evaluate(test_df,
{evaluator.metricName: "mse"})
)
print(evaluator.evaluate(test_df,
{evaluator.metricName: "rmse"})
)
print(evaluator.evaluate(test_df,
{evaluator.metricName: "mae"})
)

# metrics from 2 model - testing dataset 
# 0.8543614513315407
# 0.5315891172351185
# 0.7291015822470271
# 0.5080625701387936

# metrics from 2 model - train dataset - no overfitting
# 0.8569671390333123
# 0.5200646207324839
# 0.7211550601170902
# 0.5030215258659366

# metrics from 2 model - 2023 dataset even better than 2022
0.8724841585916305
0.41413181188308607
0.643530738879726
0.4435985325532268

# metrics from 4 attributes model
# 0.8554213326236759
# 0.526633113815338
# 0.7256949178651715
# 0.5035343987095042

# COMMAND ----------

# if prediction out of range
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
first=F.udf(lambda v:float(v[0]),FloatType())
second=F.udf(lambda v:float(v[1]),FloatType())
test_df = test_df.withColumn("PerKMSecondsAccMeterPerSecSquaredGe35", first("features")).withColumn("PerKMSecondsAccMeterPerSecSquaredGe40", second("features"))

#0.06%
test_df.filter((test_df.prediction > test_df.PerKMSecondsAccMeterPerSecSquaredGe35) | (test_df.prediction < test_df.PerKMSecondsAccMeterPerSecSquaredGe40)).count() / test_df.count()

# COMMAND ----------

# Plot error distribution
test_df_ = test_df.withColumn('error', test_df.label - test_df.prediction)
test_df_pandas = test_df_.toPandas()
ax = test_df_pandas['error'].plot.hist(bins=50, alpha=0.5)
ax

# COMMAND ----------

print(test_df.count())
print(data_35_zero.count())
print(data_35_40_match.count())
print(test_df.count() + data_35_zero.count() + data_35_40_match.count())

# COMMAND ----------

print(test_df.count())
data_35_zero = data_35_zero.withColumn("label", data_35_zero.PerKMSecondsAccMeterPerSecSquaredGe361).withColumn("prediction", data_35_zero.PerKMSecondsAccMeterPerSecSquaredGe361)
print(data_35_zero.count())
data_35_40_match = data_35_40_match.withColumn("label", data_35_40_match.PerKMSecondsAccMeterPerSecSquaredGe361).withColumn("prediction", data_35_40_match.PerKMSecondsAccMeterPerSecSquaredGe361)
print(data_35_40_match.count())
data_back1 = unionAll([data_35_zero, data_35_40_match])
data_back1 = data_back1.select('label', 'prediction')
print(data_back1.count())

data_all_test = unionAll([test_df.select('label', 'prediction'), data_back1])
print(data_all_test.count())

# COMMAND ----------

evaluator = RegressionEvaluator()
print(evaluator.evaluate(data_all_test,
{evaluator.metricName: "r2"})
)
print(evaluator.evaluate(data_all_test,
{evaluator.metricName: "mse"})
)
print(evaluator.evaluate(data_all_test,
{evaluator.metricName: "rmse"})
)
print(evaluator.evaluate(data_all_test,
{evaluator.metricName: "mae"})
)

# COMMAND ----------

#KIA
data_35_zero = data_KIA.filter(data_KIA.PerKMSecondsAccMeterPerSecSquaredGe35==0)
print(data_KIA.filter(data_KIA.PerKMSecondsAccMeterPerSecSquaredGe35 == 0).count() / data_KIA.count())
data_KIA = data_KIA.filter(data_KIA.PerKMSecondsAccMeterPerSecSquaredGe35>0)
data_KIA.count()

# COMMAND ----------

#KIA
data_35_40_match = data_KIA.filter(data_KIA.PerKMSecondsAccMeterPerSecSquaredGe35==data_KIA.PerKMSecondsAccMeterPerSecSquaredGe40)
print(data_KIA.filter(data_KIA.PerKMSecondsAccMeterPerSecSquaredGe35 == data_KIA.PerKMSecondsAccMeterPerSecSquaredGe40).count() / data_KIA.count())
data_KIA = data_KIA.filter(data_KIA.PerKMSecondsAccMeterPerSecSquaredGe35 != data_KIA.PerKMSecondsAccMeterPerSecSquaredGe40)
data_KIA.count()

# COMMAND ----------

data_Large_KIA = data_KIA.filter(data_KIA.PerKMSecondsAccMeterPerSecSquaredGe35 > 42.3)
data_Large_KIA = data_Large_KIA.withColumnRenamed("PerKMSecondsAccMeterPerSecSquaredGe361", "label")
data_Large_KIA = assembler.transform(data_Large_KIA)
data_Large_KIA = data_Large_KIA.select('features', 'label')

data_Small_KIA = data_KIA.filter(data_KIA.PerKMSecondsAccMeterPerSecSquaredGe35 <= 42.3)
data_Small_KIA = data_Small_KIA.withColumnRenamed("PerKMSecondsAccMeterPerSecSquaredGe361", "label")
data_Small_KIA = assembler.transform(data_Small_KIA)
data_Small_KIA = data_Small_KIA.select('features', 'label')


data_Large_KIA = model_Large.transform(data_Large_KIA)
data_Small_KIA = model_Small.transform(data_Small_KIA)

data_35_zero = data_35_zero.withColumn("label", data_35_zero.PerKMSecondsAccMeterPerSecSquaredGe361).withColumn("prediction", data_35_zero.PerKMSecondsAccMeterPerSecSquaredGe361)
print(data_35_zero.count())
data_35_40_match = data_35_40_match.withColumn("label", data_35_40_match.PerKMSecondsAccMeterPerSecSquaredGe361).withColumn("prediction", data_35_40_match.PerKMSecondsAccMeterPerSecSquaredGe361)
print(data_35_40_match.count())
data_back1 = unionAll([data_35_zero, data_35_40_match])
data_back1 = data_back1.select('label', 'prediction')
print(data_back1.count())

data_all_test = unionAll([data_Small_KIA.select('label', 'prediction'), data_Large_KIA.select('label', 'prediction'), data_back1])
print(data_all_test.count())

# COMMAND ----------

print(data_Large_KIA.count())
print(data_Small_KIA.count())

# COMMAND ----------

evaluator = RegressionEvaluator()
print(evaluator.evaluate(data_all_test,
{evaluator.metricName: "r2"})
)
print(evaluator.evaluate(data_all_test,
{evaluator.metricName: "mse"})
)
print(evaluator.evaluate(data_all_test,
{evaluator.metricName: "rmse"})
)
print(evaluator.evaluate(data_all_test,
{evaluator.metricName: "mae"})
)

# COMMAND ----------

print('KIA')
test_df_pandas = data_all_test.toPandas()
test_df_pandas.plot.scatter(x='label', y='prediction') 

# COMMAND ----------

# MAGIC %md # Acc 4.72 Modeling

# COMMAND ----------

# loading the 2 months data
data1 = spark.read.options(header='true', inferschema='true').csv("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/KIAEU_GM_EntityAttr_202211.csv")
data2 = spark.read.options(header='true', inferschema='true').csv("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/KIAEU_GM_EntityAttr_202306.csv")

def unionAll(dfs):
    return functools.reduce(lambda df1, df2: df1.union(df2.select(df1.columns)), dfs)
data = unionAll([data1, data2])

print(data.count()) 
print(data.dropna().count())

# remove data with unrealistic distance Traveled
# keep 99.3%
data = data.withColumn('KMDistanceTraveled', data.MetersDistanceTraveled / 1000)
data = data.filter(data.KMDistanceTraveled < 10000) 
data = data.filter(data.KMDistanceTraveled > 100)
data.count()

#create Per KM attrs
data = data.withColumn('PerKMSecondsAccMeterPerSecSquaredGe30', data.SecondsAccMeterPerSecSquaredGe30 * 1000 / data.KMDistanceTraveled)
data = data.withColumn('PerKMSecondsAccMeterPerSecSquaredGe35', data.SecondsAccMeterPerSecSquaredGe35 * 1000/ data.KMDistanceTraveled)
data = data.withColumn('PerKMSecondsAccMeterPerSecSquaredGe361', data.SecondsAccMeterPerSecSquaredGe361 * 1000/ data.KMDistanceTraveled)
data = data.withColumn('PerKMSecondsAccMeterPerSecSquaredGe40', data.SecondsAccMeterPerSecSquaredGe40 * 1000/ data.KMDistanceTraveled)
data = data.withColumn('PerKMSecondsAccMeterPerSecSquaredGe45', data.SecondsAccMeterPerSecSquaredGe45 * 1000/ data.KMDistanceTraveled)
data = data.withColumn('PerKMSecondsAccMeterPerSecSquaredGe472', data.SecondsAccMeterPerSecSquaredGe472 * 1000/ data.KMDistanceTraveled)
data = data.withColumn('PerKMSecondsAccMeterPerSecSquaredGe50', data.SecondsAccMeterPerSecSquaredGe50 * 1000/ data.KMDistanceTraveled)
data = data.select("PerKMSecondsAccMeterPerSecSquaredGe30", "PerKMSecondsAccMeterPerSecSquaredGe35", 'PerKMSecondsAccMeterPerSecSquaredGe361', 'PerKMSecondsAccMeterPerSecSquaredGe40', 'PerKMSecondsAccMeterPerSecSquaredGe45','PerKMSecondsAccMeterPerSecSquaredGe472', 'PerKMSecondsAccMeterPerSecSquaredGe50')



# COMMAND ----------

#remove data have zero second in SecondsAccMeterPerSecSquaredGe45
#82% records removed
data_45_zero = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe45==0)
print(data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe45 == 0).count() / data.count())
data = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe45>0)
data.count()

# COMMAND ----------

# 22% PerKMSecondsAccMeterPerSecSquaredGe45 == PerKMSecondsAccMeterPerSecSquaredGe50
data_45_50_match = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe45==data.PerKMSecondsAccMeterPerSecSquaredGe50)
print(data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe45 == data.PerKMSecondsAccMeterPerSecSquaredGe50).count() / data.count())
data = data.filter(data.PerKMSecondsAccMeterPerSecSquaredGe45 != data.PerKMSecondsAccMeterPerSecSquaredGe50)
data.count()

# COMMAND ----------

train_df, test_df = data.randomSplit(weights=[0.7,0.3], seed=100)
data_Large = train_df.filter(train_df.PerKMSecondsAccMeterPerSecSquaredGe50 > 11.05)
data_Small = train_df.filter(train_df.PerKMSecondsAccMeterPerSecSquaredGe50 <= 11.05)
print(data_Large.count())
print(data_Small.count())

data_Large_test = test_df.filter(test_df.PerKMSecondsAccMeterPerSecSquaredGe35 > 11.05)
data_Small_test = test_df.filter(test_df.PerKMSecondsAccMeterPerSecSquaredGe35 <= 11.05)

print(data_Large_test.count())
print(data_Small_test.count())

# COMMAND ----------

# train_df, test_df = data.randomSplit(weights=[0.7,0.3], seed=100)
# train_df = train_df.withColumnRenamed("PerKMSecondsAccMeterPerSecSquaredGe472", "label")
# assembler=VectorAssembler().setInputCols(['PerKMSecondsAccMeterPerSecSquaredGe45','PerKMSecondsAccMeterPerSecSquaredGe50']).setOutputCol('features')
# train_df = assembler.transform(train_df)
# train_df = train_df.select("features","label")
# dt = DecisionTreeRegressor(maxDepth=2)
# model = dt.fit(train_df)
# print(model.toDebugString)

# COMMAND ----------

data_Large = data_Large.withColumnRenamed("PerKMSecondsAccMeterPerSecSquaredGe472", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsAccMeterPerSecSquaredGe45','PerKMSecondsAccMeterPerSecSquaredGe50']).setOutputCol('features')
data_Large = assembler.transform(data_Large)
data_Large = data_Large.select("features","label")
lr = LinearRegression()
model_Large = lr.fit(data_Large)
print('intercept and coefficients')
print(model_Large.intercept, model_Large.coefficients)
print('pValues')
print(model_Large.summary.pValues)

data_Small = data_Small.withColumnRenamed("PerKMSecondsAccMeterPerSecSquaredGe472", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsAccMeterPerSecSquaredGe45','PerKMSecondsAccMeterPerSecSquaredGe50']).setOutputCol('features')
data_Small = assembler.transform(data_Small)
data_Small = data_Small.select("features","label")
lr = LinearRegression()
model_Small = lr.fit(data_Small)
print('intercept and coefficients')
print(model_Small.intercept, model_Small.coefficients)
print('pValues')
print(model_Small.summary.pValues)

lr_path1 = "dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/" + "Acc472_Large"
model_Large.save(lr_path1)

lr_path2 = "dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/" + "Acc472_Small"
model_Small.save(lr_path2)

# COMMAND ----------

data_Large_test = data_Large_test.withColumnRenamed("PerKMSecondsAccMeterPerSecSquaredGe472", "label")
data_Large_test = assembler.transform(data_Large_test)
data_Large_test = data_Large_test.select('features', 'label')
data_Large_test = model_Large.transform(data_Large_test)

data_Small_test = data_Small_test.withColumnRenamed("PerKMSecondsAccMeterPerSecSquaredGe472", "label")
data_Small_test = assembler.transform(data_Small_test)
data_Small_test = data_Small_test.select('features', 'label')
data_Small_test = model_Small.transform(data_Small_test)

# COMMAND ----------

print('data_Large_test')
test_df_pandas = data_Large_test.sample(fraction=0.05, seed=3).toPandas()
test_df_pandas.plot.scatter(x='label', y='prediction') 
print('data_Small_test')
test_df_pandas = data_Small_test.sample(fraction=0.01, seed=3).toPandas()
test_df_pandas.plot.scatter(x='label', y='prediction') 

# COMMAND ----------

test_df = unionAll([data_Large_test, data_Small_test])
evaluator = RegressionEvaluator()
print(evaluator.evaluate(test_df,
{evaluator.metricName: "r2"})
)
print(evaluator.evaluate(test_df,
{evaluator.metricName: "mse"})
)
print(evaluator.evaluate(test_df,
{evaluator.metricName: "rmse"})
)
print(evaluator.evaluate(test_df,
{evaluator.metricName: "mae"})
)


# COMMAND ----------

print(test_df.count())
data_45_zero = data_45_zero.withColumn("label", data_45_zero.PerKMSecondsAccMeterPerSecSquaredGe472).withColumn("prediction", data_45_zero.PerKMSecondsAccMeterPerSecSquaredGe472)
print(data_45_zero.count())
data_45_50_match = data_45_50_match.withColumn("label", data_45_50_match.PerKMSecondsAccMeterPerSecSquaredGe472).withColumn("prediction", data_45_50_match.PerKMSecondsAccMeterPerSecSquaredGe472)
print(data_45_50_match.count())
data_back1 = unionAll([data_45_zero, data_45_50_match])
data_back1 = data_back1.select('label', 'prediction')
print(data_back1.count())

data_all_test = unionAll([test_df.select('label', 'prediction'), data_back1])
print(data_all_test.count())

# COMMAND ----------

evaluator = RegressionEvaluator()
print(evaluator.evaluate(data_all_test,
{evaluator.metricName: "r2"})
)
print(evaluator.evaluate(data_all_test,
{evaluator.metricName: "mse"})
)
print(evaluator.evaluate(data_all_test,
{evaluator.metricName: "rmse"})
)
print(evaluator.evaluate(data_all_test,
{evaluator.metricName: "mae"})
)

# COMMAND ----------

# if prediction out of range
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
first=F.udf(lambda v:float(v[0]),FloatType())
second=F.udf(lambda v:float(v[1]),FloatType())
test_df = test_df.withColumn("PerKMSecondsAccMeterPerSecSquaredGe45", first("features")).withColumn("PerKMSecondsAccMeterPerSecSquaredGe50", second("features"))

#0.03%
test_df.filter((test_df.prediction > test_df.PerKMSecondsAccMeterPerSecSquaredGe45) | (test_df.prediction < test_df.PerKMSecondsAccMeterPerSecSquaredGe50)).count() / test_df.count()

# COMMAND ----------

#KIA
data_45_zero = data_KIA.filter(data_KIA.PerKMSecondsAccMeterPerSecSquaredGe45==0)
print(data_KIA.filter(data_KIA.PerKMSecondsAccMeterPerSecSquaredGe45 == 0).count() / data_KIA.count())
data_KIA = data_KIA.filter(data_KIA.PerKMSecondsAccMeterPerSecSquaredGe45>0)
data_KIA.count()

#KIA
data_45_50_match = data_KIA.filter(data_KIA.PerKMSecondsAccMeterPerSecSquaredGe45==data_KIA.PerKMSecondsAccMeterPerSecSquaredGe50)
print(data_KIA.filter(data_KIA.PerKMSecondsAccMeterPerSecSquaredGe45 == data_KIA.PerKMSecondsAccMeterPerSecSquaredGe50).count() / data_KIA.count())
data_KIA = data_KIA.filter(data_KIA.PerKMSecondsAccMeterPerSecSquaredGe45 != data_KIA.PerKMSecondsAccMeterPerSecSquaredGe50)
data_KIA.count()


# COMMAND ----------

data_Large_KIA = data_KIA.filter(data_KIA.PerKMSecondsAccMeterPerSecSquaredGe50 > 11.05)
data_Large_KIA = data_Large_KIA.withColumnRenamed("PerKMSecondsAccMeterPerSecSquaredGe472", "label")
data_Large_KIA = assembler.transform(data_Large_KIA)
data_Large_KIA = data_Large_KIA.select('features', 'label')

data_Small_KIA = data_KIA.filter(data_KIA.PerKMSecondsAccMeterPerSecSquaredGe50 <= 11.05)
data_Small_KIA = data_Small_KIA.withColumnRenamed("PerKMSecondsAccMeterPerSecSquaredGe472", "label")
data_Small_KIA = assembler.transform(data_Small_KIA)
data_Small_KIA = data_Small_KIA.select('features', 'label')


data_Large_KIA = model_Large.transform(data_Large_KIA)
data_Small_KIA = model_Small.transform(data_Small_KIA)

data_45_zero = data_45_zero.withColumn("label", data_45_zero.PerKMSecondsAccMeterPerSecSquaredGe472).withColumn("prediction", data_45_zero.PerKMSecondsAccMeterPerSecSquaredGe472)
print(data_45_zero.count())
data_45_50_match = data_45_50_match.withColumn("label", data_45_50_match.PerKMSecondsAccMeterPerSecSquaredGe472).withColumn("prediction", data_45_50_match.PerKMSecondsAccMeterPerSecSquaredGe472)
print(data_45_50_match.count())
data_back1 = unionAll([data_45_zero, data_45_50_match])
data_back1 = data_back1.select('label', 'prediction')
print(data_back1.count())

data_all_test = unionAll([data_Small_KIA.select('label', 'prediction'), data_Large_KIA.select('label', 'prediction'), data_back1])
print(data_all_test.count())

# COMMAND ----------

print(data_Large_KIA.count())
print(data_Small_KIA.count())

# COMMAND ----------

evaluator = RegressionEvaluator()
print(evaluator.evaluate(data_all_test,
{evaluator.metricName: "r2"})
)
print(evaluator.evaluate(data_all_test,
{evaluator.metricName: "mse"})
)
print(evaluator.evaluate(data_all_test,
{evaluator.metricName: "rmse"})
)
print(evaluator.evaluate(data_all_test,
{evaluator.metricName: "mae"})
)

# COMMAND ----------

print('KIA')
test_df_pandas = data_all_test.toPandas()
test_df_pandas.plot.scatter(x='label', y='prediction') 
