from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from cleantext import sanitize
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pathlib import Path

def main(context):
    """Main function takes a Spark SQL context."""
    # YOUR CODE HERE
    # YOU MAY ADD OTHER FUNCTIONS AS NEEDED
    
    # TASK 1
    # Load data from parquets if directory exists
    # Otherwise, read from json/csv files and write to new parquet
    if Path("/home/cs143/project2/comments.pqt").is_dir():
        commentsDF = sqlContext.read.parquet('comments.pqt')
    else:
        commentsDF = context.read.json("comments-minimal.json.bz2")
        commentsDF.write.parquet("comments.pqt")
        
    if Path("/home/cs143/project2/labels.pqt").is_dir():
        labelsDF = sqlContext.read.parquet('labels.pqt')
    else:
        labelsDF = context.read.csv("labeled_data.csv", header=True)
        labelsDF.write.parquet("labels.pqt")

    # if Path("/home/cs143/project2/submissions.pqt").is_dir():
    #     submissionsDF = sqlContext.read.parquet('submissions.pqt')
    # else:
    #     submissionsDF = context.read.json("submissions.json.bz2")
    #     submissionsDF.write.parquet("submissions.pqt")

    # Skip to Task 6A if parquet for sanitized, labelled data already exists
    if Path("/home/cs143/project2/sanitized_data.pqt").is_dir():
        dataDF = sqlContext.read.parquet('sanitized_data.pqt')
    else:
        # TASK 2
        # Question 1: F = {id -> label_dem,label_gop,label_djt)
        # Question 2:
        # Yes, this table seems normalized. The collector stored it this way because it was the most straightforward way of storing the comment ID and its associated labels
        dataDF = labelsDF.join(commentsDF, labelsDF.Input_id == commentsDF.id)

        # TASKS 4, 5
        # Call sanitize as UDF on body attribute of dataframe
        sanitize_udf = udf(sanitize, ArrayType(StringType()))
        dataDF = dataDF.withColumn("sanitized_text", sanitize_udf('body'))
        dataDF.write.parquet("sanitized_data.pqt")
        

    # TASKS 6A, 6B
    cv = CountVectorizer(inputCol="sanitized_text", outputCol="features", binary=True, minDF=10)
    udf_pos_colum = udf(pos_column, IntegerType())
    udf_neg_colum = udf(neg_column, IntegerType())

    positive_df = dataDF.withColumn("poslabel", udf_pos_colum('labeldjt'))
    negative_df = dataDF.withColumn("neglabel", udf_neg_colum('labeldjt'))
    pos = cv.fit(positive_df)
    neg = cv.fit(negative_df)
    pos = pos.transform(positive_df)
    neg = neg.transform(negative_df)

    # Initialize two logistic regression models.
    # Replace labelCol with the column containing the label, and featuresCol with the column containing the features.
    poslr = LogisticRegression(labelCol="poslabel", featuresCol="features", maxIter=10)
    neglr = LogisticRegression(labelCol="neglabel", featuresCol="features", maxIter=10)
    # This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
    posEvaluator = BinaryClassificationEvaluator(labelCol="poslabel")
    negEvaluator = BinaryClassificationEvaluator(labelCol="neglabel")
    # There are a few parameters associated with logistic regression. We do not know what they are a priori.
    # We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
    # We will assume the parameter is 1.0. Grid search takes forever.
    posParamGrid = ParamGridBuilder().addGrid(poslr.regParam, [1.0]).build()
    negParamGrid = ParamGridBuilder().addGrid(neglr.regParam, [1.0]).build()
    # We initialize a 5 fold cross-validation pipeline.
    posCrossval = CrossValidator(
        estimator=poslr,
        evaluator=posEvaluator,
        estimatorParamMaps=posParamGrid,
        numFolds=5)
    negCrossval = CrossValidator(
        estimator=neglr,
        evaluator=negEvaluator,
        estimatorParamMaps=negParamGrid,
        numFolds=5)
    # Although crossvalidation creates its own train/test sets for
    # tuning, we still need a labeled test set, because it is not
    # accessible from the crossvalidator (argh!)
    # Split the data 50/50
    posTrain, posTest = pos.randomSplit([0.5, 0.5])
    negTrain, negTest = neg.randomSplit([0.5, 0.5])
    # Train the models
    print("Training positive classifier...")
    posModel = posCrossval.fit(posTrain)
    # print("Training negative classifier...")
    negModel = negCrossval.fit(negTrain)

    # # Once we train the models, we don't want to do it again. We can save the models and load them again later.
    posModel.save("project2/pos.model")
    negModel.save("project2/neg.model")

    # result = model.transform(dataDF)
    # result.show(truncate=True)

    # TASK 8
    # Remove sarcastic or quote comments
    commentsDF = commentsDF.filter((~commentsDF.body.like("%/s%")) & (~commentsDF.body.like("&gt%"))).select("*")
    
def pos_column(value):
    return 1 if int(value) == 1 else 0
def neg_column(value):
    return 1 if int(value) == -1 else 0
    
if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    sc.setLogLevel("WARN")
    main(sqlContext)
