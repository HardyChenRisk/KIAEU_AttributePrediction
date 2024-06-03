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
data_KIA = data_KIA.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe30', data_KIA.SecondsBrakeMeterPerSecSquaredGe3 * 1000 / data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe35', data_KIA.SecondsBrakeMeterPerSecSquaredGe35 * 1000/ data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe361', data_KIA.SecondsBrakeMeterPerSecSquaredGe361 * 1000/ data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe40', data_KIA.SecondsBrakeMeterPerSecSquaredGe4 * 1000/ data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe45', data_KIA.SecondsBrakeMeterPerSecSquaredGe45 * 1000/ data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe472', data_KIA.SecondsBrakeMeterPerSecSquaredGe472 * 1000/ data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe50', data_KIA.SecondsBrakeMeterPerSecSquaredGe5 * 1000/ data_KIA.KMDistanceTraveled)
data_KIA = data_KIA.select("PerKMSecondsBrakeMeterPerSecSquaredGe30", "PerKMSecondsBrakeMeterPerSecSquaredGe35", 'PerKMSecondsBrakeMeterPerSecSquaredGe361', 'PerKMSecondsBrakeMeterPerSecSquaredGe40', 'PerKMSecondsBrakeMeterPerSecSquaredGe45','PerKMSecondsBrakeMeterPerSecSquaredGe472', 'PerKMSecondsBrakeMeterPerSecSquaredGe50')

# COMMAND ----------

# loading the 2 months data
data1 = spark.read.options(header='true', inferschema='true').csv("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/KIAEU_GM_EntityAttr_202211.csv")
data2 = spark.read.options(header='true', inferschema='true').csv("dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/KIAEU_GM_EntityAttr_202306.csv")

def unionAll(dfs):
    return functools.reduce(lambda df1, df2: df1.union(df2.select(df1.columns)), dfs)
data_all = unionAll([data1, data2])

print(data_all.count()) 
print(data_all.dropna().count())

# COMMAND ----------

# remove data with unrealistic distance Traveled
# keep 99.3%
data_all = data_all.withColumn('KMDistanceTraveled', data_all.MetersDistanceTraveled / 1000)
data_all = data_all.filter(data_all.KMDistanceTraveled < 10000) 
data_all = data_all.filter(data_all.KMDistanceTraveled > 100)
data_all.count()

# COMMAND ----------

#create Per KM attrs
data_all = data_all.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe30', data_all.SecondsBrakeMeterPerSecSquaredGe30 * 1000 / data_all.KMDistanceTraveled)
data_all = data_all.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe35', data_all.SecondsBrakeMeterPerSecSquaredGe35 * 1000/ data_all.KMDistanceTraveled)
data_all = data_all.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe361', data_all.SecondsBrakeMeterPerSecSquaredGe361 * 1000/ data_all.KMDistanceTraveled)
data_all = data_all.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe40', data_all.SecondsBrakeMeterPerSecSquaredGe40 * 1000/ data_all.KMDistanceTraveled)
data_all = data_all.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe45', data_all.SecondsBrakeMeterPerSecSquaredGe45 * 1000/ data_all.KMDistanceTraveled)
data_all = data_all.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe472', data_all.SecondsBrakeMeterPerSecSquaredGe472 * 1000/ data_all.KMDistanceTraveled)
data_all = data_all.withColumn('PerKMSecondsBrakeMeterPerSecSquaredGe50', data_all.SecondsBrakeMeterPerSecSquaredGe50 * 1000/ data_all.KMDistanceTraveled)
data_all = data_all.select("PerKMSecondsBrakeMeterPerSecSquaredGe30", "PerKMSecondsBrakeMeterPerSecSquaredGe35", 'PerKMSecondsBrakeMeterPerSecSquaredGe361', 'PerKMSecondsBrakeMeterPerSecSquaredGe40', 'PerKMSecondsBrakeMeterPerSecSquaredGe45','PerKMSecondsBrakeMeterPerSecSquaredGe472', 'PerKMSecondsBrakeMeterPerSecSquaredGe50')

# COMMAND ----------

# MAGIC %md # Brake 3.61 Modeling

# COMMAND ----------

#final sample size, keeps 93%
# data_final_outlier = data_all.filter(data_all.PerKMSecondsBrakeMeterPerSecSquaredGe35 >= 30)
# data = data_all.filter(data_all.PerKMSecondsBrakeMeterPerSecSquaredGe35 < 30)
# data.count()

# COMMAND ----------

#remove data have zero second in SecondsBrakeMeterPerSecSquaredGe35
#5% records removed
data_35_zero = data_all.filter(data_all.PerKMSecondsBrakeMeterPerSecSquaredGe35==0)
print(data_all.filter(data_all.PerKMSecondsBrakeMeterPerSecSquaredGe35 == 0).count() / data_all.count())
data = data_all.filter(data_all.PerKMSecondsBrakeMeterPerSecSquaredGe35>0)
data.count()

# COMMAND ----------

# 1.5% PerKMSecondsBrakeMeterPerSecSquaredGe35 == PerKMSecondsBrakeMeterPerSecSquaredGe40
data_35_40_match = data.filter(data.PerKMSecondsBrakeMeterPerSecSquaredGe35==data.PerKMSecondsBrakeMeterPerSecSquaredGe40)
print(data.filter(data.PerKMSecondsBrakeMeterPerSecSquaredGe35 == data.PerKMSecondsBrakeMeterPerSecSquaredGe40).count() / data.count())
data = data.filter(data.PerKMSecondsBrakeMeterPerSecSquaredGe35 != data.PerKMSecondsBrakeMeterPerSecSquaredGe40)
data.count()

# COMMAND ----------

# train_df, test_df = data.randomSplit(weights=[0.7,0.3], seed=100)
# train_df = train_df.withColumnRenamed("PerKMSecondsBrakeMeterPerSecSquaredGe361", "label")
# assembler=VectorAssembler().setInputCols(['PerKMSecondsBrakeMeterPerSecSquaredGe35','PerKMSecondsBrakeMeterPerSecSquaredGe40']).setOutputCol('features')
# train_df = assembler.transform(train_df)
# train_df = train_df.select("features","label")
# dt = DecisionTreeRegressor(maxDepth=2)
# model = dt.fit(train_df)
# print(model.toDebugString)

# COMMAND ----------

train_df, test_df = data.randomSplit(weights=[0.7,0.3], seed=100)
data_Large = train_df.filter(train_df.PerKMSecondsBrakeMeterPerSecSquaredGe35 > 60.18)
data_Small = train_df.filter(train_df.PerKMSecondsBrakeMeterPerSecSquaredGe35 <= 60.18)

print(data_Small.count())
print(data_Large.count())

data_Large_test = test_df.filter(test_df.PerKMSecondsBrakeMeterPerSecSquaredGe35 > 60.18)
data_Small_test = test_df.filter(test_df.PerKMSecondsBrakeMeterPerSecSquaredGe35 <= 60.18)

print(data_Small_test.count())
print(data_Large_test.count())


# COMMAND ----------

# train_df, test_df = data.randomSplit(weights=[0.7,0.3], seed=100)

# COMMAND ----------

# Total Testing Size
print(test_df.count())
print(data_35_zero.count())
print(data_35_40_match.count())
print(test_df.count() + data_35_zero.count() + data_35_40_match.count())

# COMMAND ----------

# Modeling with 2 attrs (PerKMSecondsBrakeMeterPerSecSquaredGe35 and PerKMSecondsBrakeMeterPerSecSquaredGe40)

# COMMAND ----------

data_Large = data_Large.withColumnRenamed("PerKMSecondsBrakeMeterPerSecSquaredGe361", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsBrakeMeterPerSecSquaredGe35','PerKMSecondsBrakeMeterPerSecSquaredGe40']).setOutputCol('features')
data_Large = assembler.transform(data_Large)
data_Large = data_Large.select("features","label", "PerKMSecondsBrakeMeterPerSecSquaredGe35")
lr = LinearRegression(featuresCol="features", labelCol="label")
# weightCol='PerKMSecondsBrakeMeterPerSecSquaredGe35')
model_Large = lr.fit(data_Large)

data_Large_test = data_Large_test.withColumnRenamed("PerKMSecondsBrakeMeterPerSecSquaredGe361", "label")
data_Large_test = assembler.transform(data_Large_test)
data_Large_test = data_Large_test.select('features', 'label')
data_Large_test = model_Large.transform(data_Large_test)

print('intercept and coefficients')
print(model_Large.intercept, model_Large.coefficients)
print('pValues')
print(model_Large.summary.pValues)

lr_path = "dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/" + "Brake361_Large"
model_Large.save(lr_path)

# COMMAND ----------

data_Small = data_Small.withColumnRenamed("PerKMSecondsBrakeMeterPerSecSquaredGe361", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsBrakeMeterPerSecSquaredGe35','PerKMSecondsBrakeMeterPerSecSquaredGe40']).setOutputCol('features')
data_Small = assembler.transform(data_Small)
data_Small = data_Small.select("features","label", "PerKMSecondsBrakeMeterPerSecSquaredGe35")
lr = LinearRegression(featuresCol="features", labelCol="label")
# weightCol='PerKMSecondsBrakeMeterPerSecSquaredGe35')
model_Small = lr.fit(data_Small)

print('intercept and coefficients')
print(model_Small.intercept, model_Small.coefficients)
print('pValues')
print(model_Small.summary.pValues)

lr_path = "dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/" + "Brake361_Small"
model_Small.save(lr_path)

# COMMAND ----------

data_Large_test = data_Large_test.withColumnRenamed("PerKMSecondsBrakeMeterPerSecSquaredGe361", "label")
data_Large_test = assembler.transform(data_Large_test)
data_Large_test = data_Large_test.select('features', 'label')
data_Large_test = model_Large.transform(data_Large_test)

data_Small_test = data_Small_test.withColumnRenamed("PerKMSecondsBrakeMeterPerSecSquaredGe361", "label")
data_Small_test = assembler.transform(data_Small_test)
data_Small_test = data_Small_test.select('features', 'label')
data_Small_test = model_Small.transform(data_Small_test)

# COMMAND ----------

test_df_pandas = data_Small_test.sample(fraction=0.001, seed=3).toPandas()
test_df_pandas.plot.scatter(x='label', y='prediction') 

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

# if prediction out of range
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
first=F.udf(lambda v:float(v[0]),FloatType())
second=F.udf(lambda v:float(v[1]),FloatType())
test_df = test_df.withColumn("PerKMSecondsAccMeterPerSecSquaredGe35", first("features")).withColumn("PerKMSecondsAccMeterPerSecSquaredGe40", second("features"))

#0.06%
test_df.filter((test_df.prediction > test_df.PerKMSecondsAccMeterPerSecSquaredGe35) | (test_df.prediction < test_df.PerKMSecondsAccMeterPerSecSquaredGe40)).count() / test_df.count()

# COMMAND ----------

print(test_df.count())
data_35_zero = data_35_zero.withColumn("label", data_35_zero.PerKMSecondsBrakeMeterPerSecSquaredGe361).withColumn("prediction", data_35_zero.PerKMSecondsBrakeMeterPerSecSquaredGe361)
print(data_35_zero.count())
data_35_40_match = data_35_40_match.withColumn("label", data_35_40_match.PerKMSecondsBrakeMeterPerSecSquaredGe361).withColumn("prediction", data_35_40_match.PerKMSecondsBrakeMeterPerSecSquaredGe361)
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
data_35_zero = data_KIA.filter(data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe35==0)
print(data_KIA.filter(data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe35 == 0).count() / data_KIA.count())
data_KIA = data_KIA.filter(data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe35>0)
data_KIA.count()

# COMMAND ----------

#KIA
data_35_40_match = data_KIA.filter(data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe35==data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe40)
print(data_KIA.filter(data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe35 == data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe40).count() / data_KIA.count())
data_KIA = data_KIA.filter(data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe35 != data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe40)
data_KIA.count()

# COMMAND ----------

data_Large_KIA = data_KIA.filter(data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe35 > 60.18)
data_Large_KIA = data_Large_KIA.withColumnRenamed("PerKMSecondsBrakeMeterPerSecSquaredGe361", "label")
data_Large_KIA = assembler.transform(data_Large_KIA)
data_Large_KIA = data_Large_KIA.select('features', 'label')

data_Small_ = data_KIA.filter(data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe35 <= 60.18)

data_Medium_KIA = data_Small_.filter(data_Small_.PerKMSecondsBrakeMeterPerSecSquaredGe35 > 18.31)
data_Medium_KIA = data_Medium_KIA.withColumnRenamed("PerKMSecondsBrakeMeterPerSecSquaredGe361", "label")
data_Medium_KIA = assembler.transform(data_Medium_KIA)
data_Medium_KIA = data_Medium_KIA.select('features', 'label')

data_Small_KIA = data_Small_.filter(data_Small_.PerKMSecondsBrakeMeterPerSecSquaredGe35 <= 18.31)
data_Small_KIA = data_Small_KIA.withColumnRenamed("PerKMSecondsBrakeMeterPerSecSquaredGe361", "label")
data_Small_KIA = assembler.transform(data_Small_KIA)
data_Small_KIA = data_Small_KIA.select('features', 'label')


data_Large_KIA = model_Large.transform(data_Large_KIA)
data_Medium_KIA = model_Medium.transform(data_Medium_KIA)
data_Small_KIA = model_Small.transform(data_Small_KIA)

data_35_zero = data_35_zero.withColumn("label", data_35_zero.PerKMSecondsBrakeMeterPerSecSquaredGe361).withColumn("prediction", data_35_zero.PerKMSecondsBrakeMeterPerSecSquaredGe361)
print(data_35_zero.count())
data_35_40_match = data_35_40_match.withColumn("label", data_35_40_match.PerKMSecondsBrakeMeterPerSecSquaredGe361).withColumn("prediction", data_35_40_match.PerKMSecondsBrakeMeterPerSecSquaredGe361)
print(data_35_40_match.count())
data_back1 = unionAll([data_35_zero, data_35_40_match])
data_back1 = data_back1.select('label', 'prediction')
print(data_back1.count())

data_all_test = unionAll([data_Small_KIA.select('label', 'prediction'), data_Large_KIA.select('label', 'prediction'), data_Medium_KIA.select('label', 'prediction'), data_back1])
print(data_all_test.count())

# COMMAND ----------

print(data_Large_KIA.count())
print(data_Medium_KIA.count())
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

# MAGIC %md # Brake 4.72 Modeling

# COMMAND ----------

#final sample size, keeps 94.4%
# data_final_outlier = data_all.filter(data_all.PerKMSecondsBrakeMeterPerSecSquaredGe45 >= 5)
# data_all = data_all.filter(data_all.PerKMSecondsBrakeMeterPerSecSquaredGe45< 5)
# data_all.count()

# COMMAND ----------

#remove data have zero second in PerKMSecondsBrakeMeterPerSecSquaredGe45
#45% records removed
data_35_zero = data_all.filter(data_all.PerKMSecondsBrakeMeterPerSecSquaredGe45==0)
print(data_all.filter(data_all.PerKMSecondsBrakeMeterPerSecSquaredGe45 == 0).count() / data_all.count())
data = data_all.filter(data_all.PerKMSecondsBrakeMeterPerSecSquaredGe45>0)
data.count()

# COMMAND ----------

# 13% PerKMSecondsBrakeMeterPerSecSquaredGe45 == PerKMSecondsBrakeMeterPerSecSquaredGe50
data_45_50_match = data.filter(data.PerKMSecondsBrakeMeterPerSecSquaredGe45==data.PerKMSecondsBrakeMeterPerSecSquaredGe50)
print(data.filter(data.PerKMSecondsBrakeMeterPerSecSquaredGe45 == data.PerKMSecondsBrakeMeterPerSecSquaredGe50).count() / data.count())
data = data.filter(data.PerKMSecondsBrakeMeterPerSecSquaredGe45 != data.PerKMSecondsBrakeMeterPerSecSquaredGe50)
data.count()

# COMMAND ----------

# Total Training + Testing Size
data.count() + data_35_zero.count() + data_45_50_match.count()
# 2457879

# COMMAND ----------

train_df, test_df = data.randomSplit(weights=[0.7,0.3], seed=100)
train_df = train_df.withColumnRenamed("PerKMSecondsBrakeMeterPerSecSquaredGe472", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsBrakeMeterPerSecSquaredGe45','PerKMSecondsBrakeMeterPerSecSquaredGe50']).setOutputCol('features')
train_df = assembler.transform(train_df)
train_df = train_df.select("features","label")
dt = DecisionTreeRegressor(maxDepth=2)
model = dt.fit(train_df)
print(model.toDebugString)

# COMMAND ----------

data.filter(data.PerKMSecondsBrakeMeterPerSecSquaredGe45 > 10.69).count()

# COMMAND ----------

data.filter(data.PerKMSecondsBrakeMeterPerSecSquaredGe45 <= 10.69).count()

# COMMAND ----------

train_df, test_df = data.randomSplit(weights=[0.7,0.3], seed=100)
data_Large = train_df.filter(train_df.PerKMSecondsBrakeMeterPerSecSquaredGe45 > 10.69)
data_Small = train_df.filter(train_df.PerKMSecondsBrakeMeterPerSecSquaredGe45 <= 10.69)
print(data_Large.count())
print(data_Small.count())

data_Large_test = test_df.filter(test_df.PerKMSecondsBrakeMeterPerSecSquaredGe45 > 10.69)
data_Small_test = test_df.filter(test_df.PerKMSecondsBrakeMeterPerSecSquaredGe45 <= 10.69)

print(data_Large_test.count())
print(data_Small_test.count())

# COMMAND ----------

# Total Testing Size
print(test_df.count())
print(data_35_zero.count())
print(data_45_50_match.count())
print(test_df.count() + data_35_zero.count() + data_45_50_match.count())


# COMMAND ----------

data_Large = data_Large.withColumnRenamed("PerKMSecondsBrakeMeterPerSecSquaredGe472", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsBrakeMeterPerSecSquaredGe45','PerKMSecondsBrakeMeterPerSecSquaredGe50']).setOutputCol('features')
data_Large = assembler.transform(data_Large)
data_Large = data_Large.select("features","label")
lr = LinearRegression()
model_Large = lr.fit(data_Large)
print('intercept and coefficients')
print(model_Large.intercept, model_Large.coefficients)
print('pValues')
print(model_Large.summary.pValues)

data_Small = data_Small.withColumnRenamed("PerKMSecondsBrakeMeterPerSecSquaredGe472", "label")
assembler=VectorAssembler().setInputCols(['PerKMSecondsBrakeMeterPerSecSquaredGe45','PerKMSecondsBrakeMeterPerSecSquaredGe50']).setOutputCol('features')
data_Small = assembler.transform(data_Small)
data_Small = data_Small.select("features","label")
lr = LinearRegression()
model_Small = lr.fit(data_Small)
print('intercept and coefficients')
print(model_Small.intercept, model_Small.coefficients)
print('pValues')
print(model_Small.summary.pValues)

lr_path1 = "dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/" + "Brake472_Large"
model_Large.save(lr_path1)

lr_path2 = "dbfs:/mnt/analytics/connectedcar/chenha01/KIAEU/" + "Brake472_Small"
model_Small.save(lr_path2)

# COMMAND ----------

data_Large_test = data_Large_test.withColumnRenamed("PerKMSecondsBrakeMeterPerSecSquaredGe472", "label")
data_Large_test = assembler.transform(data_Large_test)
data_Large_test = data_Large_test.select('features', 'label')
data_Large_test = model_Large.transform(data_Large_test)

data_Small_test = data_Small_test.withColumnRenamed("PerKMSecondsBrakeMeterPerSecSquaredGe472", "label")
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


# COMMAND ----------

# if prediction out of range
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
first=F.udf(lambda v:float(v[0]),FloatType())
second=F.udf(lambda v:float(v[1]),FloatType())
test_df = test_df.withColumn("PerKMSecondsBrakeMeterPerSecSquaredGe45", first("features")).withColumn("PerKMSecondsBrakeMeterPerSecSquaredGe50", second("features"))

#0.03%
test_df.filter((test_df.prediction > test_df.PerKMSecondsBrakeMeterPerSecSquaredGe45) | (test_df.prediction < test_df.PerKMSecondsBrakeMeterPerSecSquaredGe50)).count() / test_df.count()

# COMMAND ----------

print(test_df.count())
data_35_zero = data_35_zero.withColumn("label", data_35_zero.PerKMSecondsBrakeMeterPerSecSquaredGe472).withColumn("prediction", data_35_zero.PerKMSecondsBrakeMeterPerSecSquaredGe472)
print(data_35_zero.count())
data_45_50_match = data_45_50_match.withColumn("label", data_45_50_match.PerKMSecondsBrakeMeterPerSecSquaredGe472).withColumn("prediction", data_45_50_match.PerKMSecondsBrakeMeterPerSecSquaredGe472)
print(data_45_50_match.count())
data_back1 = unionAll([data_35_zero, data_45_50_match])
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
data_45_zero = data_KIA.filter(data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe45==0)
print(data_KIA.filter(data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe45 == 0).count() / data_KIA.count())
data_KIA = data_KIA.filter(data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe45>0)
data_KIA.count()

#KIA
data_45_50_match = data_KIA.filter(data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe45==data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe50)
print(data_KIA.filter(data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe45 == data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe50).count() / data_KIA.count())
data_KIA = data_KIA.filter(data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe45 != data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe50)
data_KIA.count()


# COMMAND ----------

data_Large_KIA = data_KIA.filter(data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe45 > 10.69)
data_Large_KIA = data_Large_KIA.withColumnRenamed("PerKMSecondsBrakeMeterPerSecSquaredGe472", "label")
data_Large_KIA = assembler.transform(data_Large_KIA)
data_Large_KIA = data_Large_KIA.select('features', 'label')

data_Small_KIA = data_KIA.filter(data_KIA.PerKMSecondsBrakeMeterPerSecSquaredGe45 <= 10.69)
data_Small_KIA = data_Small_KIA.withColumnRenamed("PerKMSecondsBrakeMeterPerSecSquaredGe472", "label")
data_Small_KIA = assembler.transform(data_Small_KIA)

# COMMAND ----------


data_Small_KIA = data_Small_KIA.select('features', 'label')


data_Large_KIA = model_Large.transform(data_Large_KIA)
data_Small_KIA = model_Small.transform(data_Small_KIA)

data_45_zero = data_45_zero.withColumn("label", data_45_zero.PerKMSecondsBrakeMeterPerSecSquaredGe472).withColumn("prediction", data_45_zero.PerKMSecondsBrakeMeterPerSecSquaredGe472)
print(data_45_zero.count())
data_45_50_match = data_45_50_match.withColumn("label", data_45_50_match.PerKMSecondsBrakeMeterPerSecSquaredGe472).withColumn("prediction", data_45_50_match.PerKMSecondsBrakeMeterPerSecSquaredGe472)
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

# COMMAND ----------

test_df__ = test_df.withColumn("ratio", test_df.label / test_df.PerKMSecondsAccMeterPerSecSquaredGe45)
test_df__ = test_df__.filter(test_df__.ratio == 0)
test_df_pandas = test_df__.toPandas()
ax = test_df_pandas['PerKMSecondsAccMeterPerSecSquaredGe45'].plot.hist(bins=50, alpha=0.5)
ax

# COMMAND ----------

test_df__ = test_df.withColumn("ratio", test_df__.label / test_df__.PerKMSecondsAccMeterPerSecSquaredGe45)
test_df__ = test_df__.filter(test_df__.ratio == 1)
test_df_pandas = test_df__.toPandas()
ax = test_df_pandas['PerKMSecondsAccMeterPerSecSquaredGe45'].plot.hist(bins=50, alpha=0.5)
ax

# COMMAND ----------

test_df_pandas.head(10)

# COMMAND ----------

test_df_pandas = test_df.withColumn("ratio", test_df.label / test_df.PerKMSecondsAccMeterPerSecSquaredGe50).toPandas()
ax = test_df_pandas['ratio'].plot.hist(bins=50, alpha=0.5)
ax

# COMMAND ----------

test_df_pandas = test_df.toPandas()
test_df_pandas.plot.scatter(x='PerKMSecondsAccMeterPerSecSquaredGe45', y='label') 

# COMMAND ----------

test_df_pandas = test_df.toPandas()
test_df_pandas.plot.scatter(x='PerKMSecondsAccMeterPerSecSquaredGe50', y='label') 

# COMMAND ----------



# COMMAND ----------


