from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from cleantext import sanitize
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel,ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pathlib import Path

def main(context):
    """Main function takes a Spark SQL context."""
    # YOUR CODE HERE
    # YOU MAY ADD OTHER FUNCTIONS AS NEEDED
    try:
        commentsDF = sqlContext.read.parquet('comments.pqt')
    except:
        commentsDF = context.read.json("comments-minimal.json.bz2")
        commentsDF.write.parquet("comments.pqt")
    try:
        labelsDF = sqlContext.read.parquet('labels.pqt')
    except:
        labelsDF = context.read.csv("labeled_data.csv", header=True)
        labelsDF.write.parquet("labels.pqt")
    try:
        submissionsDF = sqlContext.read.parquet("submissions.pqt")
    except:
        submissionsDF = context.read.json("submissions.json.bz2")
        submissionsDF.write.parquet("submissions.pqt")


    dataDF = labelsDF.join(commentsDF, labelsDF.Input_id == commentsDF.id)
    # ### TASK 4 + 5
    sanitize_udf = udf(sanitize, ArrayType(StringType()))
    dataDF = dataDF.withColumn("sanitized_text", sanitize_udf('body'))
    # dataDF.write.parquet("sanitized_data.pqt")


    # TASKS 6A, 6B
    cv = CountVectorizer(inputCol="sanitized_text", outputCol="features", binary=True, minDF=10)
    udf_pos_colum = udf(pos_column, IntegerType())
    udf_neg_colum = udf(neg_column, IntegerType())
    model = cv.fit(dataDF)
    result = model.transform(dataDF)
    # positive_df = result.withColumn("poslabel", udf_pos_colum('labeldjt'))
    # negative_df = result.withColumn("neglabel", udf_neg_colum('labeldjt'))


    # # Initialize two logistic regression models.
    # # Replace labelCol with the column containing the label, and featuresCol with the column containing the features.
    # poslr = LogisticRegression(labelCol="poslabel", featuresCol="features", maxIter=10)
    # neglr = LogisticRegression(labelCol="neglabel", featuresCol="features", maxIter=10)
    # # This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
    # posEvaluator = BinaryClassificationEvaluator(labelCol="poslabel")
    # negEvaluator = BinaryClassificationEvaluator(labelCol="neglabel")
    # # There are a few parameters associated with logistic regression. We do not know what they are a priori.
    # # We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
    # # We will assume the parameter is 1.0. Grid search takes forever.
    # posParamGrid = ParamGridBuilder().addGrid(poslr.regParam, [1.0]).build()
    # negParamGrid = ParamGridBuilder().addGrid(neglr.regParam, [1.0]).build()
    # # We initialize a 5 fold cross-validation pipeline.
    # posCrossval = CrossValidator(
    #     estimator=poslr,
    #     evaluator=posEvaluator,
    #     estimatorParamMaps=posParamGrid,
    #     numFolds=5)
    # negCrossval = CrossValidator(
    #     estimator=neglr,
    #     evaluator=negEvaluator,
    #     estimatorParamMaps=negParamGrid,
    #     numFolds=5)
    # # Although crossvalidation creates its own train/test sets for
    # # tuning, we still need a labeled test set, because it is not
    # # accessible from the crossvalidator (argh!)
    # # Split the data 50/50
    # posTrain, posTest = positive_df.randomSplit([0.5, 0.5])
    # negTrain, negTest = negative_df.randomSplit([0.5, 0.5])
    # # Train the models
    # print("Training positive classifier...")
    # posModel = posCrossval.fit(posTrain)
    # print("Training negative classifier...")
    # negModel = negCrossval.fit(negTrain)

    # # # Once we train the models, we don't want to do it again. We can save the models and load them again later.
    # posModel.save("project2/pos.model")
    # negModel.save("project2/neg.model")
    # # # TASK 8
    # # Remove sarcastic or quote comments
    posModel = CrossValidatorModel.load('project2/pos.model')
    negModel = CrossValidatorModel.load('project2/neg.model')
    commentsDF = commentsDF.filter((~commentsDF.body.like("%/s%")) &
                    (~commentsDF.body.like("&gt%"))).select("*")
    clean_udf = udf(clean_link, StringType())
    cleanedDF = commentsDF.withColumn("clean_link_id", clean_udf('link_id'))
    pre_sanitizedDF = cleanedDF.join(submissionsDF,
        cleanedDF.clean_link_id == submissionsDF.id).select(cleanedDF['created_utc'],
        cleanedDF['body'],cleanedDF['author_flair_text'], submissionsDF['score'],
        cleanedDF['clean_link_id'], submissionsDF['title'])
    sanitizedDF = pre_sanitizedDF.withColumn('sanitized_text', sanitize_udf('body'))
    result = model.transform(sanitizedDF)
    pos_training = posModel.transform(result)
    print(pos_training)
    # result = model.transform(dataDF)
    # result.show(truncate=True)

def clean_link(link):
    return link[3:]
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
