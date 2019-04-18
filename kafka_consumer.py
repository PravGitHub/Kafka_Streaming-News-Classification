from pyspark import SparkContext
import sys
from pyspark.sql.types import *
from pyspark.sql import SQLContext, SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.streaming import StreamingContext
from kafka import KafkaConsumer

#----------------------Processing data using KafkaConsumer-----------------------------------------

sc = SparkContext.getOrCreate()
sc.setLogLevel("ERROR")
data = sc.textFile("/home/asdf/Documents/news", 1)
data = data.map(lambda l: l.strip('"')).map(lambda l: l.split("||")).map(lambda l: (int(l[0]),l[1]))

consumer = KafkaConsumer('guardian2', bootstrap_servers=['localhost:9092'], auto_offset_reset='earliest', consumer_timeout_ms=1000,value_deserializer=lambda x: x.decode('utf-8'))

list = []
for i in consumer:
    tmp = []
    tmp.append(int(i.value.split("||")[0]))
    tmp.append(i.value.split("||")[1])
    list.append(tmp)


sqlContext = SQLContext(sc)
df = sqlContext.createDataFrame(data, schema=["label", "text"])
test_data = sqlContext.createDataFrame(list, schema=["label", "text"])

lr = Pipeline.load("/home/asdf/Documents/models/lr")
nb = Pipeline.load("/home/asdf/Documents/models/nb")

predictions = lr.fit(df).transform(test_data)
nb_predictions = nb.fit(df).transform(test_data)
accuracy = MulticlassClassificationEvaluator(predictionCol="prediction")
recall = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedPrecision")
precision = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedRecall")

print("Logistic Regression:\n")
print("Accuracy:", accuracy.evaluate(predictions) * 100)
print("\nRecall:", recall.evaluate(predictions) * 100)
print("\nPrecision:", precision.evaluate(predictions) * 100)
print('\n')

print("\nNaive Bayes:")
print("\nAccuracy:", accuracy.evaluate(nb_predictions) * 100)
print("\nRecall:", recall.evaluate(nb_predictions) * 100)
print("\nPrecision:", precision.evaluate(nb_predictions) * 100)




