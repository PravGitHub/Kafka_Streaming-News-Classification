
from pyspark import SparkContext
import sys
from pyspark.sql.types import *
from pyspark.sql import SQLContext, SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from pyspark.ml.classification import LogisticRegressionModel, NaiveBayesModel


#-----------------------Processing stream data in batches of 60 secs-------------------------------------------

def func(rd):
    rd = rd.map(lambda l: l.strip('"')).map(lambda l: l.split("||")).map(lambda l: (int(l[0]), l[1]))
    sqlContext = SQLContext(sc)
    test_data = sqlContext.createDataFrame(rd, schema=["label", "text"])

    data = sc.textFile("/home/asdf/Documents/news", 1)
    data = data.map(lambda l: l.strip('"')).map(lambda l: l.split("||")).map(lambda l: (int(l[0]), l[1]))


    df = sqlContext.createDataFrame(data, schema=["label", "text"])

    lr = Pipeline.load("/home/asdf/Documents/models/lr")
    nb = Pipeline.load("/home/asdf/Documents/models/nb")

    lr_pred = lr.fit(df).transform(test_data)
    nb_pred = nb.fit(df).transform(test_data)
    accuracy = MulticlassClassificationEvaluator(predictionCol="prediction")
    recall = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedPrecision")
    precision = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedRecall")



    with open("/home/asdf/Documents/op1", 'a') as file:
        file.write("Logistic Regression:\n")
        file.write("Accuracy:"+str(accuracy.evaluate(lr_pred) * 100))
        file.write("\nRecall:"+str(recall.evaluate(lr_pred) * 100))
        file.write("\nPrecision:"+str(precision.evaluate(lr_pred) * 100))
        file.write('\n')

        file.write("\nNaive Bayes:\n")
        file.write("Accuracy:"+str(accuracy.evaluate(nb_pred) * 100))
        file.write("\nRecall:"+str(recall.evaluate(nb_pred) * 100))
        file.write("\nPrecision:"+str(precision.evaluate(nb_pred) * 100))


#-----------------------------------------------------------------------------------

if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    ssc = StreamingContext(sc, 60)
    kstream = KafkaUtils.createDirectStream(ssc=ssc, topics=["guardian2"],kafkaParams={'bootstrap.servers': 'localhost:9092','auto.offset.reset': 'smallest'})
    kstream = kstream.map(lambda x: x[1])
    kstream.foreachRDD(lambda r: func(r))

    ssc.start()
    ssc.awaitTermination(timeout=200)