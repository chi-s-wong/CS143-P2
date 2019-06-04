from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from cleantext import sanitize
from pyspark.sql.types import *
from pyspark.sql.functions import udf, col
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel,ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pathlib import Path


def main(context):
    udf_pos_colum = udf(pos_column, IntegerType())
    udf_neg_colum = udf(neg_column, IntegerType())
    states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
    'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia',
    'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
    'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
    'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
    'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',
    'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
    'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia',
    'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
    # try:
    commentsDF = sqlContext.read.parquet('comments.pqt').sample(False, .25, None)
    # except:
    #     commentsDF = context.read.json("comments-minimal.json.bz2")
    #     commentsDF.write.parquet("comments.pqt")

    # try:
    labelsDF = sqlContext.read.parquet('labels.pqt').sample(False, .25, None)
    # # except:
    # #     labelsDF = context.read.csv("labeled_data.csv", header=True)
    # #     labelsDF.write.parquet("labels.pqt")
    # # try:
    submissionsDF = sqlContext.read.parquet("submissions.pqt").sample(False, .25, None)

    # # except:
    # #     submissionsDF = context.read.json("submissions.json.bz2")
    # #     submissionsDF.write.parquet("submissions.pqt")


    dataDF = labelsDF.join(commentsDF, labelsDF.Input_id == commentsDF.id)
    # # ### TASK 4 + 5
    sanitize_udf = udf(sanitize, ArrayType(StringType()))
    dataDF = dataDF.withColumn("sanitized_text", sanitize_udf('body'))
    # # # dataDF.write.parquet("sanitized_data.pqt")


    # # TASKS 6A, 6B
    cv = CountVectorizer(inputCol="sanitized_text", outputCol="features",
                          binary=True, minDF=10)
    model = cv.fit(dataDF)
    result = model.transform(dataDF)
    positive_df = result.withColumn("poslabel", udf_pos_colum('labeldjt'))
    negative_df = result.withColumn("neglabel", udf_neg_colum('labeldjt'))

    # # try:
    # posModel = CrossValidatorModel.load('project2/pos.model')
    # negModel = CrossValidatorModel.load('project2/neg.model')
    # except:
    posModel, negModel = train_models(positive_df, negative_df)

    task10 = get_pos_negDF(commentsDF, submissionsDF, posModel, negModel, model,
                            sanitize_udf)
    # task10.write.parquet("task10.pqt")
    task10.show(n=80)
    task10.createOrReplaceTempView("dataTable")
    # perc_across_subm = context.sql("""SELECT id, AVG(pos) AS pos_avg, AVG(neg)
    #                                  AS neg_avg, COUNT(id) FROM dataTable
    #                                  GROUP BY id""")
    times = context.sql("""SELECT from_unixtime(time,'YYYY-MM-dd') AS date,
                        AVG(pos) AS Positive, AVG(neg) AS Negative FROM
                        dataTable GROUP BY date""")


    task10 = task10.filter(col('state').isin(states))
    task10.createOrReplaceTempView('dataTable')
    states = context.sql("""SELECT state, AVG(pos) AS Positive, AVG(neg) AS
                            Negative, COUNT(state) from dataTable GROUP BY state""")
    states.show(n=50)
    # perc_across_subm.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("percents.csv")
    # times.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("times.csv")
    # states.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("states.csv")

def get_pos_negDF(commentsDF, submissionsDF, posModel, negModel, model, clean):
    udf_clean = udf(clean_link, StringType())
    udf_pos = udf(get_pos_prob, IntegerType())
    udf_neg = udf(get_neg_prob, IntegerType())
    # # TASK 8
    # # Remove sarcastic or quote comments
    commentsDF = commentsDF.filter((~commentsDF.body.like("%/s%")) &
                    (~commentsDF.body.like("&gt%"))).select("*")
    print(commentsDF)
    cleanedDF = commentsDF.withColumn("clean_link_id", udf_clean('link_id'))
    cleanedDF = cleanedDF.withColumnRenamed("score", "comment_score")
    print(cleanedDF)
    pre_sanitizedDF = cleanedDF.join(submissionsDF,
        cleanedDF.clean_link_id == submissionsDF.id).select(
        cleanedDF['created_utc'], cleanedDF['body'], cleanedDF['comment_score'],
        cleanedDF['author_flair_text'], submissionsDF['score'],
        cleanedDF['clean_link_id'], submissionsDF['title'])
    sanDF = pre_sanitizedDF.withColumn('sanitized_text', clean('body'))
    result = model.transform(sanDF)
    pos_training = posModel.transform(result).selectExpr('features',
        'comment_score as comScore', 'score as subcore',
        'clean_link_id as id', 'created_utc as time', 'body',
        'author_flair_text as state', 'prediction as p', 'rawPrediction as rP',
        'title','probability as pos_probability',
        'sanitized_text')
    neg_training = negModel.transform(pos_training)
    neg_training = neg_training.withColumn('pos', udf_pos('pos_probability'))
    neg_training = neg_training.withColumn('neg', udf_neg('probability'))
    return neg_training


def train_models(pos, neg):
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
    print("Training negative classifier...")
    negModel = negCrossval.fit(negTrain)

    # Once we train the models, we don't want to do it again.
    # We can save the models and load them again later.
    posModel.save("project2/pos.model")
    negModel.save("project2/neg.model")
    return posModel, negModel

def get_pos_prob(probability):
    return 1 if float(probability[1]) > .2 else 0
def get_neg_prob(probability):
    return 1 if float(probability[1]) > .25 else 0
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
