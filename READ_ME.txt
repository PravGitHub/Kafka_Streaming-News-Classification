
NOTE: The inputs such as api-keys, dates, training data etc., are hard coded. Please edit it before running.

producer.py -- Sends the messages to the kafka server.
kb.py --- Builds the pipeline for logistic regression and naive bayes, and stores it on disk.
models -- contains the pipelines for naive bayes and logistic regression.

news -- contains the training data.
categories -- contains the list of categories.
stopwords.txt -- contains the list of all stop words.

NOTE: I have implemented the processing, using 2 approaches.

1) Using KafkaConsumer:
File --- kafka_consumer.py 
output -- Screenshot from 2019-04-14 16-07-43.png

2) Using DirectStream: Processes data in 60 sec batches.
File -- kc.py
output file -- op1