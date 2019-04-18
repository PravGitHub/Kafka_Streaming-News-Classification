from pyspark import SparkContext
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.feature import CountVectorizer, StopWordsRemover, RegexTokenizer
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from kafka import KafkaConsumer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#-------------------Building the logistic regression and naive bayes pipelines----------------------------------

if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")
    data = sc.textFile("/home/asdf/Documents/news", 1)
    data = data.map(lambda l: l.strip('"')).map(lambda l: l.split("||")).map(lambda l: (int(l[0]),l[1]))

    sqlContext = SQLContext(sc)
    df = sqlContext.createDataFrame(data, schema=["label", "text"])

    regex_tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")

    stop_words = []
    with open('/home/asdf/Documents/stopwords.txt', 'r') as contents:
        stop_words = contents.read().split()

    stop_words_remover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(stop_words)

    count_vectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)

    lr = LogisticRegression(maxIter=100, regParam=0.01)
    nb = NaiveBayes(labelCol="label", featuresCol="features", smoothing=1.0, modelType="multinomial")
    pipe1 = Pipeline(stages=[regex_tokenizer, stop_words_remover, count_vectors, lr])
    pipe2 = Pipeline(stages=[regex_tokenizer, stop_words_remover, count_vectors, nb])

    pipe1.save("models/lr")
    pipe2.save("models/nb")


