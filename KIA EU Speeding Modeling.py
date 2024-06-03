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
data_KIA = spark.read.options(header='true', inferschema='true').csv("/mnt/analytics/connectedcar/chenha01/KIAEU/KIAEU_KIA_EntityAttr_202211.csv")
data_KIA = data_KIA.fillna(0)

# COMMAND ----------

data_KIA = data_KIA.withColumn('KMDistanceTraveled', data_KIA.MetersDistanceTraveled / 1000)
data_KIA = data_KIA.withColumn('DistancePerTrip', data_KIA.MetersDistanceTraveled / data_KIA.tripcount)
data_KIA = data_KIA.filter(data_KIA.KMDistanceTraveled < 10000) 
data_KIA = data_KIA.filter(data_KIA.KMDistanceTraveled > 100)
print(data_KIA.count())

# COMMAND ----------

#create Per KM attrs
data_KIA = data_KIA.withColumn('PerKMSecondsOverMph80', data_KIA.SecondsOverMph80 / data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.withColumn('PerKMSecondsOverMph75', data_KIA.SecondsOverMph75/ data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe35', data_KIA.SecondsBrakeMeterPerSecSquaredGe35/ data_KIA.KMDistanceTraveled)

# COMMAND ----------

# MAGIC %md # SecondsOverMph75

# COMMAND ----------

# Final Model
train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("PerKMSecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsOverMph80','DistancePerTrip', 'meanavespeed']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
lr = LinearRegression()
model = lr.fit(train_df)
print('intercept and coefficients')
print(model.intercept, model.coefficients)
print('pValues')
print(model.summary.pValues)

test_df = test_df.withColumnRenamed("PerKMSecondsOverMph75", "label")
test_df = assembler.transform(test_df)
test_df = test_df.select('features', 'label','KMDistanceTraveled', 'SecondsOverMph75')
test_df = model.transform(test_df)

test_df = test_df.withColumn('label', test_df.SecondsOverMph75)
test_df = test_df.withColumn('prediction', test_df.prediction * data_KIA.KMDistanceTraveled)

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

lr_path1 = "dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/" + "SecondOver75"
model.save(lr_path1)

# COMMAND ----------

data_KIA_pandas = data_KIA.toPandas()
ax = data_KIA_pandas['SecondsOverMph80'].plot.hist(bins=50, alpha=0.5)
ax = data_KIA_pandas['SecondsOverMph75'].plot.hist(bins=50, alpha=0.5)
ax
# SecondsOverMph80 blue
# SecondsOverMph75 orange

# COMMAND ----------

data_KIA_pandas = data_KIA.filter(data_KIA.PerKMSecondsOverMph80 < 1.08).toPandas()
data_KIA_pandas.plot.scatter(x='SecondsOverMph80', y='SecondsOverMph75') 

# COMMAND ----------

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("PerKMSecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsOverMph80','DistancePerTrip', 'meanavespeed']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
dt = DecisionTreeRegressor(maxDepth=2)
model = dt.fit(train_df)
print(model.toDebugString)

# COMMAND ----------

data_KIA.filter(data_KIA.PerKMSecondsOverMph80 < 1.08).count()

# COMMAND ----------

data_KIA_pandas = data_KIA.filter(data_KIA.PerKMSecondsOverMph80 < 1.08).toPandas()
data_KIA_pandas.plot.scatter(x='PerKMSecondsOverMph80', y='PerKMSecondsOverMph75') 

# COMMAND ----------

# 1. SecondsOverMph80 Predicting SecondsOverMph75

# COMMAND ----------

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("SecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['SecondsOverMph80']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
lr = LinearRegression()
model = lr.fit(train_df)
print('intercept and coefficients')
print(model.intercept, model.coefficients)
print('pValues')
print(model.summary.pValues)

test_df = test_df.withColumnRenamed("SecondsOverMph75", "label")
test_df = assembler.transform(test_df)
test_df = test_df.select('features', 'label')
test_df = model.transform(test_df)

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

# 2. PerKMSecondsOverMph80 Predicting PerKMSecondsOverMph75

# COMMAND ----------

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("PerKMSecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsOverMph80']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
lr = LinearRegression()
model = lr.fit(train_df)
print('intercept and coefficients')
print(model.intercept, model.coefficients)
print('pValues')
print(model.summary.pValues)

test_df = test_df.withColumnRenamed("PerKMSecondsOverMph75", "label")
test_df = assembler.transform(test_df)
test_df = test_df.select('features', 'label','KMDistanceTraveled', 'SecondsOverMph75')
test_df = model.transform(test_df)


# COMMAND ----------

test_df = test_df.withColumn('label', test_df.SecondsOverMph75)
test_df = test_df.withColumn('prediction', test_df.prediction * data_KIA.KMDistanceTraveled)

# COMMAND ----------

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

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("PerKMSecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsOverMph80']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
lr = LinearRegression()
model = lr.fit(train_df)
print('intercept and coefficients')
print(model.intercept, model.coefficients)
print('pValues')
print(model.summary.pValues)

test_df = test_df.withColumnRenamed("PerKMSecondsOverMph75", "label")
test_df = assembler.transform(test_df)
test_df = test_df.select('features', 'label')
test_df = model.transform(test_df)

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

# 3.  SecondsOverMph80 + KMDistanceTraveled Predicting PerKMSecondsOverMph75

# COMMAND ----------

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("SecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['SecondsOverMph80','KMDistanceTraveled']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
lr = LinearRegression()
model = lr.fit(train_df)
print('intercept and coefficients')
print(model.intercept, model.coefficients)
print('pValues')
print(model.summary.pValues)

test_df = test_df.withColumnRenamed("SecondsOverMph75", "label")
test_df = assembler.transform(test_df)
test_df = test_df.select('features', 'label')
test_df = model.transform(test_df)

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

# 3.  SecondsOverMph80 + KMDistanceTraveled + DistancePerTrip + meanavespeed Predicting PerKMSecondsOverMph75

# COMMAND ----------

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("SecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['SecondsOverMph80','KMDistanceTraveled', 'DistancePerTrip', 'meanavespeed']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
lr = LinearRegression()
model = lr.fit(train_df)
print('intercept and coefficients')
print(model.intercept, model.coefficients)
print('pValues')
print(model.summary.pValues)

test_df = test_df.withColumnRenamed("SecondsOverMph75", "label")
test_df = assembler.transform(test_df)
test_df = test_df.select('features', 'label')
test_df = model.transform(test_df)

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

# 2. PerKMSecondsOverMph80 + DistancePerTrip + meanavespeed Predicting PerKMSecondsOverMph75

# COMMAND ----------

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("PerKMSecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsOverMph80','DistancePerTrip', 'meanavespeed']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
lr = LinearRegression()
model = lr.fit(train_df)
print('intercept and coefficients')
print(model.intercept, model.coefficients)
print('pValues')
print(model.summary.pValues)

test_df = test_df.withColumnRenamed("PerKMSecondsOverMph75", "label")
test_df = assembler.transform(test_df)
test_df = test_df.select('features', 'label','KMDistanceTraveled', 'SecondsOverMph75')
test_df = model.transform(test_df)

test_df = test_df.withColumn('label', test_df.SecondsOverMph75)
test_df = test_df.withColumn('prediction', test_df.prediction * data_KIA.KMDistanceTraveled)

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

#4. Applied Decision Tree filtering SecondsOverMph80 < 2845. 82% Kept

# COMMAND ----------

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("SecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['SecondsOverMph80', 'KMDistanceTraveled']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
dt = DecisionTreeRegressor(maxDepth=2)
model = dt.fit(train_df)
print(model.toDebugString)

# COMMAND ----------

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
data_Large = train_df.filter(train_df.SecondsOverMph80 > 2845)
data_Small = train_df.filter(train_df.SecondsOverMph80 <= 2845)
print(data_Large.count())
print(data_Small.count())

data_Large_test = test_df.filter(test_df.SecondsOverMph80 > 2845)
data_Small_test = test_df.filter(test_df.SecondsOverMph80 <= 2845)

print(data_Large_test.count())
print(data_Small_test.count())

# COMMAND ----------

data_KIA_pandas = data_Small.toPandas()
data_KIA_pandas.plot.scatter(x='SecondsOverMph80', y='SecondsOverMph75')

# COMMAND ----------

data_Large = data_Large.withColumnRenamed("SecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['SecondsOverMph80', 'KMDistanceTraveled']).setOutputCol('features')
data_Large = assembler.transform(data_Large)
data_Large = data_Large.select("features","label")
lr = LinearRegression()
model_Large = lr.fit(data_Large)
print('intercept and coefficients')
print(model_Large.intercept, model_Large.coefficients)
print('pValues')
print(model_Large.summary.pValues)

data_Small = data_Small.withColumnRenamed("SecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['SecondsOverMph80', 'KMDistanceTraveled']).setOutputCol('features')
data_Small = assembler.transform(data_Small)
data_Small = data_Small.select("features","label")
lr = LinearRegression()
model_Small = lr.fit(data_Small)
print('intercept and coefficients')
print(model_Small.intercept, model_Small.coefficients)
print('pValues')
print(model_Small.summary.pValues)

# lr_path1 = "dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/" + "Acc361_Large"
# model_Large.save(lr_path1)

# lr_path2 = "dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/" + "Acc361_Small"
# model_Small.save(lr_path2)



# COMMAND ----------

data_Large_test = data_Large_test.withColumnRenamed("SecondsOverMph75", "label")
data_Large_test = assembler.transform(data_Large_test)
data_Large_test = data_Large_test.select('features', 'label')
data_Large_test = model_Large.transform(data_Large_test)

data_Small_test = data_Small_test.withColumnRenamed("SecondsOverMph75", "label")
data_Small_test = assembler.transform(data_Small_test)
data_Small_test = data_Small_test.select('features', 'label')
data_Small_test = model_Small.transform(data_Small_test)



# COMMAND ----------

def unionAll(dfs):
    return functools.reduce(lambda df1, df2: df1.union(df2.select(df1.columns)), dfs)

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

#5. Applied Decision Tree filtering SecondsOverMph80 < 2845. 82% Kept

# COMMAND ----------

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
data_Large = train_df.filter(train_df.SecondsOverMph80 > 2845.0)
data_Small_ = train_df.filter(train_df.SecondsOverMph80 <= 2845.0)
data_Medium = data_Small_.filter(data_Small_.SecondsOverMph80 > 681.0)
data_Small = data_Small_.filter(data_Small_.SecondsOverMph80 <= 681.0)

print(data_Large.count())
print(data_Medium.count())
print(data_Small.count())

data_Large_test = test_df.filter(test_df.SecondsOverMph80 > 2845.0)
data_Small_test_ = test_df.filter(test_df.SecondsOverMph80 <= 2845.0)
data_Medium_test = data_Small_test_.filter(data_Small_test_.SecondsOverMph80 > 681.0)
data_Small_test = data_Small_test_.filter(data_Small_test_.SecondsOverMph80 <= 681.0)

print(data_Large_test.count())
print(data_Medium_test.count())
print(data_Small_test.count())

# COMMAND ----------

data_Large = data_Large.withColumnRenamed("SecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['SecondsOverMph80','MetersDistanceTraveled']).setOutputCol('features')
data_Large = assembler.transform(data_Large)
data_Large = data_Large.select("features","label")
lr = LinearRegression()
model_Large = lr.fit(data_Large)
print('intercept and coefficients')
print(model_Large.intercept, model_Large.coefficients)
print('pValues')
print(model_Large.summary.pValues)

data_Medium = data_Medium.withColumnRenamed("SecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['SecondsOverMph80','MetersDistanceTraveled']).setOutputCol('features')
data_Medium = assembler.transform(data_Medium)
data_Medium = data_Medium.select("features","label")
lr = LinearRegression()
model_Medium = lr.fit(data_Medium)
print('intercept and coefficients')
print(model_Medium.intercept, model_Medium.coefficients)
print('pValues')
print(model_Medium.summary.pValues)

data_Small = data_Small.withColumnRenamed("SecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['SecondsOverMph80','MetersDistanceTraveled']).setOutputCol('features')
data_Small = assembler.transform(data_Small)
data_Small = data_Small.select("features","label")
lr = LinearRegression()
model_Small = lr.fit(data_Small)
print('intercept and coefficients')
print(model_Small.intercept, model_Small.coefficients)
print('pValues')
print(model_Small.summary.pValues)


# COMMAND ----------

data_Large_test = data_Large_test.withColumnRenamed("SecondsOverMph75", "label")
data_Large_test = assembler.transform(data_Large_test)
data_Large_test = data_Large_test.select('features', 'label')
data_Large_test = model_Large.transform(data_Large_test)

data_Medium_test = data_Medium_test.withColumnRenamed("SecondsOverMph75", "label")
data_Medium_test = assembler.transform(data_Medium_test)
data_Medium_test = data_Medium_test.select('features', 'label')
data_Medium_test = model_Medium.transform(data_Medium_test)

data_Small_test = data_Small_test.withColumnRenamed("SecondsOverMph75", "label")
data_Small_test = assembler.transform(data_Small_test)
data_Small_test = data_Small_test.select('features', 'label')
data_Small_test = model_Small.transform(data_Small_test)

# COMMAND ----------

test_df = unionAll([data_Large_test, data_Medium_test, data_Small_test])
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

print('data_Large_test')
test_df_pandas = data_Large_test.toPandas()
test_df_pandas.plot.scatter(x='label', y='prediction') 
print('data_Medium_test')
test_df_pandas = data_Medium_test.toPandas()
test_df_pandas.plot.scatter(x='label', y='prediction') 
print('data_Small_test')
test_df_pandas = data_Small_test.toPandas()
test_df_pandas.plot.scatter(x='label', y='prediction') 

# COMMAND ----------

#Applied Decision Tree filtering PerKMSecondsOverMph80 < 1.0830397184898577. 75% Kept

# COMMAND ----------

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("PerKMSecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsOverMph80', 'DistancePerTrip', 'meanavespeed']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
dt = DecisionTreeRegressor(maxDepth=2)
model = dt.fit(train_df)
print(model.toDebugString)

# COMMAND ----------

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
data_Large = train_df.filter(train_df.PerKMSecondsOverMph80 > 1.0830397184898577)
data_Small = train_df.filter(train_df.PerKMSecondsOverMph80 <= 1.0830397184898577)
print(data_Large.count())
print(data_Small.count())

data_Large_test = test_df.filter(test_df.PerKMSecondsOverMph80 > 1.0830397184898577)
data_Small_test = test_df.filter(test_df.PerKMSecondsOverMph80 <= 1.0830397184898577)

print(data_Large_test.count())
print(data_Small_test.count())

# COMMAND ----------

data_Large = data_Large.withColumnRenamed("PerKMSecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsOverMph80']).setOutputCol('features')
data_Large = assembler.transform(data_Large)
data_Large = data_Large.select("features","label")
lr = LinearRegression()
model_Large = lr.fit(data_Large)
print('intercept and coefficients')
print(model_Large.intercept, model_Large.coefficients)
print('pValues')
print(model_Large.summary.pValues)

data_Small = data_Small.withColumnRenamed("PerKMSecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsOverMph80']).setOutputCol('features')
data_Small = assembler.transform(data_Small)
data_Small = data_Small.select("features","label")
lr = LinearRegression()
model_Small = lr.fit(data_Small)
print('intercept and coefficients')
print(model_Small.intercept, model_Small.coefficients)
print('pValues')
print(model_Small.summary.pValues)


# COMMAND ----------

data_Large_test = data_Large_test.withColumnRenamed("PerKMSecondsOverMph75", "label")
data_Large_test = assembler.transform(data_Large_test)
data_Large_test = data_Large_test.select('features', 'label', 'SecondsOverMph75','KMDistanceTraveled')
data_Large_test = model_Large.transform(data_Large_test)

data_Small_test = data_Small_test.withColumnRenamed("PerKMSecondsOverMph75", "label")
data_Small_test = assembler.transform(data_Small_test)
data_Small_test = data_Small_test.select('features', 'label','SecondsOverMph75','KMDistanceTraveled')
data_Small_test = model_Small.transform(data_Small_test)

# COMMAND ----------

data_Small_test = data_Small_test.withColumn('label', data_Small_test.SecondsOverMph75)
data_Small_test = data_Small_test.withColumn('prediction', data_Small_test.prediction * data_KIA.KMDistanceTraveled)

data_Large_test = data_Large_test.withColumn('label', data_Large_test.SecondsOverMph75)
data_Large_test = data_Large_test.withColumn('prediction', data_Large_test.prediction * data_KIA.KMDistanceTraveled)

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

print('data_Large_test')
test_df_pandas = data_Large_test.toPandas()
test_df_pandas.plot.scatter(x='label', y='prediction') 
print('data_Small_test')
test_df_pandas = data_Small_test.toPandas()
test_df_pandas.plot.scatter(x='label', y='prediction') 

# COMMAND ----------

#Applied Decision Tree filtering PerKMSecondsOverMph80 < 1.0830397184898577. 75% Kept

# COMMAND ----------

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
data_Large = train_df.filter(train_df.SecondsOverMph80 > 1.0830397184898577)
data_Small_ = train_df.filter(train_df.SecondsOverMph80 <= 1.0830397184898577)
data_Medium = data_Small_.filter(data_Small_.SecondsOverMph80 > 0.21764098474412455)
data_Small = data_Small_.filter(data_Small_.SecondsOverMph80 <= 0.21764098474412455)

print(data_Large.count())
print(data_Medium.count())
print(data_Small.count())

data_Large_test = test_df.filter(test_df.SecondsOverMph80 > 1.0830397184898577)
data_Small_test_ = test_df.filter(test_df.SecondsOverMph80 <= 1.0830397184898577)
data_Medium_test = data_Small_test_.filter(data_Small_test_.SecondsOverMph80 > 0.21764098474412455)
data_Small_test = data_Small_test_.filter(data_Small_test_.SecondsOverMph80 <= 0.21764098474412455)

print(data_Large_test.count())
print(data_Medium_test.count())
print(data_Small_test.count())

# COMMAND ----------

#Last one

# COMMAND ----------

large_set = ['PerKMSecondsOverMph80']
small_set = ['PerKMSecondsOverMph80', 'meanavespeed']
train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
data_Large = train_df.filter(train_df.PerKMSecondsOverMph80 > 1.0830397184898577)
data_Small = train_df.filter(train_df.PerKMSecondsOverMph80 <= 1.0830397184898577)
# print(data_Large.count())
# print(data_Small.count())

data_Large_test = test_df.filter(test_df.PerKMSecondsOverMph80 > 1.0830397184898577)
data_Small_test = test_df.filter(test_df.PerKMSecondsOverMph80 <= 1.0830397184898577)

# print(data_Large_test.count())
# print(data_Small_test.count())

data_Large = data_Large.withColumnRenamed("PerKMSecondsOverMph75", "label")
assembler=VectorAssembler().setInputCols(large_set).setOutputCol('features')
data_Large = assembler.transform(data_Large)
data_Large = data_Large.select("features","label")
lr = LinearRegression()
model_Large = lr.fit(data_Large)
print('intercept and coefficients')
print(model_Large.intercept, model_Large.coefficients)
print('pValues')
print(model_Large.summary.pValues)

data_Small = data_Small.withColumnRenamed("PerKMSecondsOverMph75", "label")
assembler_small=VectorAssembler().setInputCols(small_set).setOutputCol('features')
data_Small = assembler_small.transform(data_Small)
data_Small = data_Small.select("features","label")
lr = LinearRegression()
model_Small = lr.fit(data_Small)
print('intercept and coefficients')
print(model_Small.intercept, model_Small.coefficients)
print('pValues')
print(model_Small.summary.pValues)

data_Large_test = data_Large_test.withColumnRenamed("PerKMSecondsOverMph75", "label")
data_Large_test = assembler.transform(data_Large_test)
data_Large_test = data_Large_test.select('features', 'label', 'SecondsOverMph75','KMDistanceTraveled')
data_Large_test = model_Large.transform(data_Large_test)

data_Small_test = data_Small_test.withColumnRenamed("PerKMSecondsOverMph75", "label")
data_Small_test = assembler_small.transform(data_Small_test)
data_Small_test = data_Small_test.select('features', 'label','SecondsOverMph75','KMDistanceTraveled')
data_Small_test = model_Small.transform(data_Small_test)

data_Small_test = data_Small_test.withColumn('label', data_Small_test.SecondsOverMph75)
data_Small_test = data_Small_test.withColumn('prediction', data_Small_test.prediction * data_KIA.KMDistanceTraveled)

data_Large_test = data_Large_test.withColumn('label', data_Large_test.SecondsOverMph75)
data_Large_test = data_Large_test.withColumn('prediction', data_Large_test.prediction * data_KIA.KMDistanceTraveled)

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

# MAGIC %md # MetersOverMph75

# COMMAND ----------

#create Per KM attrs
data_KIA = data_KIA.withColumn('PerKMMetersOverMph80', data_KIA.MetersOverMph80 / data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.withColumn('PerKMMetersOverMph75', data_KIA.MetersOverMph75/ data_KIA.KMDistanceTraveled)

# COMMAND ----------

data_KIA_pandas = data_KIA.toPandas()
ax = data_KIA_pandas['MetersOverMph80'].plot.hist(bins=50, alpha=0.5)
ax = data_KIA_pandas['MetersOverMph75'].plot.hist(bins=50, alpha=0.5)
ax

# COMMAND ----------

data_KIA_pandas = data_KIA.toPandas()
data_KIA_pandas.plot.scatter(x='MetersOverMph80', y='MetersOverMph75') 

# COMMAND ----------

data_KIA_pandas = data_KIA.toPandas()
data_KIA_pandas.plot.scatter(x='PerKMMetersOverMph80', y='PerKMMetersOverMph75') 

# COMMAND ----------

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("MetersOverMph75", "label")
assembler=VectorAssembler().setInputCols(['MetersOverMph80']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
lr = LinearRegression()
model = lr.fit(train_df)
print('intercept and coefficients')
print(model.intercept, model.coefficients)
print('pValues')
print(model.summary.pValues)

test_df = test_df.withColumnRenamed("MetersOverMph75", "label")
test_df = assembler.transform(test_df)
test_df = test_df.select('features', 'label')
test_df = model.transform(test_df)

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

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("PerKMMetersOverMph75", "label")
assembler=VectorAssembler().setInputCols(['PerKMMetersOverMph80']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
lr = LinearRegression()
model = lr.fit(train_df)
print('intercept and coefficients')
print(model.intercept, model.coefficients)
print('pValues')
print(model.summary.pValues)

test_df = test_df.withColumnRenamed("PerKMMetersOverMph75", "label")
test_df = assembler.transform(test_df)
test_df = test_df.select('features', 'label')
test_df = model.transform(test_df)

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

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("MetersOverMph75", "label")
assembler=VectorAssembler().setInputCols(['MetersOverMph80','KMDistanceTraveled']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
lr = LinearRegression()
model = lr.fit(train_df)
print('intercept and coefficients')
print(model.intercept, model.coefficients)
print('pValues')
print(model.summary.pValues)

test_df = test_df.withColumnRenamed("MetersOverMph75", "label")
test_df = assembler.transform(test_df)
test_df = test_df.select('features', 'label')
test_df = model.transform(test_df)

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

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("MetersOverMph75", "label")
assembler=VectorAssembler().setInputCols(['MetersOverMph80', 'KMDistanceTraveled']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
dt = DecisionTreeRegressor(maxDepth=2)
model = dt.fit(train_df)
print(model.toDebugString)

# COMMAND ----------

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
data_Large = train_df.filter(train_df.MetersOverMph80 > 105482.82115)
data_Small = train_df.filter(train_df.MetersOverMph80 <= 105482.82115)
print(data_Large.count())
print(data_Small.count())

data_Large_test = test_df.filter(test_df.MetersOverMph80 > 105482.82115)
data_Small_test = test_df.filter(test_df.MetersOverMph80 <= 105482.82115)

print(data_Large_test.count())
print(data_Small_test.count())

# COMMAND ----------

data_KIA_pandas = data_Small.toPandas()
data_KIA_pandas.plot.scatter(x='MetersOverMph80', y='MetersOverMph75')

# COMMAND ----------

data_Large = data_Large.withColumnRenamed("MetersOverMph75", "label")
assembler=VectorAssembler().setInputCols(['MetersOverMph80', 'KMDistanceTraveled']).setOutputCol('features')
data_Large = assembler.transform(data_Large)
data_Large = data_Large.select("features","label")
lr = LinearRegression()
model_Large = lr.fit(data_Large)
print('intercept and coefficients')
print(model_Large.intercept, model_Large.coefficients)
print('pValues')
print(model_Large.summary.pValues)

data_Small = data_Small.withColumnRenamed("MetersOverMph75", "label")
assembler=VectorAssembler().setInputCols(['MetersOverMph80', 'KMDistanceTraveled']).setOutputCol('features')
data_Small = assembler.transform(data_Small)
data_Small = data_Small.select("features","label")
lr = LinearRegression()
model_Small = lr.fit(data_Small)
print('intercept and coefficients')
print(model_Small.intercept, model_Small.coefficients)
print('pValues')
print(model_Small.summary.pValues)

# lr_path1 = "dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/" + "Acc361_Large"
# model_Large.save(lr_path1)

# lr_path2 = "dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/" + "Acc361_Small"
# model_Small.save(lr_path2)

# COMMAND ----------

data_Large_test = data_Large_test.withColumnRenamed("MetersOverMph75", "label")
data_Large_test = assembler.transform(data_Large_test)
data_Large_test = data_Large_test.select('features', 'label')
data_Large_test = model_Large.transform(data_Large_test)

data_Small_test = data_Small_test.withColumnRenamed("MetersOverMph75", "label")
data_Small_test = assembler.transform(data_Small_test)
data_Small_test = data_Small_test.select('features', 'label')
data_Small_test = model_Small.transform(data_Small_test)

def unionAll(dfs):
    return functools.reduce(lambda df1, df2: df1.union(df2.select(df1.columns)), dfs)
	
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

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
data_Large = train_df.filter(train_df.MetersOverMph80 > 105482.82115)
data_Small_ = train_df.filter(train_df.MetersOverMph80 <= 105482.82115)
data_Medium = data_Small_.filter(data_Small_.MetersOverMph80 > 25393.476564999997)
data_Small = data_Small_.filter(data_Small_.MetersOverMph80 <= 25393.476564999997)

print(data_Large.count())
print(data_Medium.count())
print(data_Small.count())

data_Large_test = test_df.filter(test_df.MetersOverMph80 > 105482.82115)
data_Small_test_ = test_df.filter(test_df.MetersOverMph80 <= 105482.82115)
data_Medium_test = data_Small_test_.filter(data_Small_test_.MetersOverMph80 > 25393.476564999997)
data_Small_test = data_Small_test_.filter(data_Small_test_.MetersOverMph80 <= 25393.476564999997)

print(data_Large_test.count())
print(data_Medium_test.count())
print(data_Small_test.count())

# COMMAND ----------

data_Large = data_Large.withColumnRenamed("MetersOverMph75", "label")
assembler=VectorAssembler().setInputCols(['MetersOverMph80','MetersDistanceTraveled']).setOutputCol('features')
data_Large = assembler.transform(data_Large)
data_Large = data_Large.select("features","label")
lr = LinearRegression()
model_Large = lr.fit(data_Large)
print('intercept and coefficients')
print(model_Large.intercept, model_Large.coefficients)
print('pValues')
print(model_Large.summary.pValues)

data_Medium = data_Medium.withColumnRenamed("MetersOverMph75", "label")
assembler=VectorAssembler().setInputCols(['MetersOverMph80','MetersDistanceTraveled']).setOutputCol('features')
data_Medium = assembler.transform(data_Medium)
data_Medium = data_Medium.select("features","label")
lr = LinearRegression()
model_Medium = lr.fit(data_Medium)
print('intercept and coefficients')
print(model_Medium.intercept, model_Medium.coefficients)
print('pValues')
print(model_Medium.summary.pValues)

data_Small = data_Small.withColumnRenamed("MetersOverMph75", "label")
assembler=VectorAssembler().setInputCols(['MetersOverMph80','MetersDistanceTraveled']).setOutputCol('features')
data_Small = assembler.transform(data_Small)
data_Small = data_Small.select("features","label")
lr = LinearRegression()
model_Small = lr.fit(data_Small)
print('intercept and coefficients')
print(model_Small.intercept, model_Small.coefficients)
print('pValues')
print(model_Small.summary.pValues)


# COMMAND ----------

data_Large_test = data_Large_test.withColumnRenamed("MetersOverMph75", "label")
data_Large_test = assembler.transform(data_Large_test)
data_Large_test = data_Large_test.select('features', 'label')
data_Large_test = model_Large.transform(data_Large_test)

data_Medium_test = data_Medium_test.withColumnRenamed("MetersOverMph75", "label")
data_Medium_test = assembler.transform(data_Medium_test)
data_Medium_test = data_Medium_test.select('features', 'label')
data_Medium_test = model_Medium.transform(data_Medium_test)

data_Small_test = data_Small_test.withColumnRenamed("MetersOverMph75", "label")
data_Small_test = assembler.transform(data_Small_test)
data_Small_test = data_Small_test.select('features', 'label')
data_Small_test = model_Small.transform(data_Small_test)

test_df = unionAll([data_Large_test, data_Medium_test, data_Small_test])
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

print('data_Large_test')
test_df_pandas = data_Large_test.toPandas()
test_df_pandas.plot.scatter(x='label', y='prediction') 
print('data_Medium_test')
test_df_pandas = data_Medium_test.toPandas()
test_df_pandas.plot.scatter(x='label', y='prediction') 
print('data_Small_test')
test_df_pandas = data_Small_test.toPandas()
test_df_pandas.plot.scatter(x='label', y='prediction') 

# COMMAND ----------

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("PerKMMetersOverMph75", "label")
assembler=VectorAssembler().setInputCols(['PerKMMetersOverMph80']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
lr = LinearRegression()
model = lr.fit(train_df)
print('intercept and coefficients')
print(model.intercept, model.coefficients)
print('pValues')
print(model.summary.pValues)

test_df = test_df.withColumnRenamed("PerKMMetersOverMph75", "label")
test_df = assembler.transform(test_df)
test_df = test_df.select('features', 'label', 'KMDistanceTraveled', 'MetersOverMph75')
test_df = model.transform(test_df)

test_df = test_df.withColumn('label', test_df.MetersOverMph75)
test_df = test_df.withColumn('prediction', test_df.prediction * data_KIA.KMDistanceTraveled)

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

train_df, test_df = data_KIA.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("PerKMMetersOverMph75", "label")
assembler=VectorAssembler().setInputCols(['PerKMMetersOverMph80', 'DistancePerTrip', 'meanavespeed']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
lr = LinearRegression()
model = lr.fit(train_df)
print('intercept and coefficients')
print(model.intercept, model.coefficients)
print('pValues')
print(model.summary.pValues)

test_df = test_df.withColumnRenamed("PerKMMetersOverMph75", "label")
test_df = assembler.transform(test_df)
test_df = test_df.select('features', 'label', 'KMDistanceTraveled', 'MetersOverMph75')
test_df = model.transform(test_df)

test_df = test_df.withColumn('label', test_df.MetersOverMph75)
test_df = test_df.withColumn('prediction', test_df.prediction * data_KIA.KMDistanceTraveled)

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

lr_path1 = "dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/" + "MeterOver75"
model.save(lr_path1)
