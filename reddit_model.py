from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from cleantext import sanitize
from pyspark.sql.types import *
from pyspark.sql.functions import udf, col
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
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

    # TASK 1
    # Read the files in from pqt, if they are not there, create them
    try:
        commentsDF = sqlContext.read.parquet('comments.pqt').sample(False, .2, None)
    except:
        commentsDF = context.read.json("comments-minimal.json.bz2")
        commentsDF.write.parquet("comments.pqt")

    try:
        labelsDF = sqlContext.read.parquet('labels.pqt')
    except:
        labelsDF = context.read.csv("labeled_data.csv", header=True)
        labelsDF.write.parquet("labels.pqt")
    try:
        submissionsDF = sqlContext.read.parquet("submissions.pqt").sample(False, .2, None)
    except:
        submissionsDF = context.read.json("submissions.json.bz2")
        submissionsDF.write.parquet("submissions.pqt")

    # TASKS 4,5
    dataDF = labelsDF.join(commentsDF, labelsDF.Input_id == commentsDF.id)
    sanitize_udf = udf(sanitize, ArrayType(StringType()))
    dataDF = dataDF.withColumn("sanitized_text", sanitize_udf('body'))
    # dataDF.write.parquet("sanitized_data.pqt")


    # TASKS 6A, 6B
    try:
        model = CountVectorizerModel.load("project2/model")
    except:
        cv = CountVectorizer(inputCol="sanitized_text", outputCol="features",
                             binary=True, minDF=10)
        model = cv.fit(dataDF)
        model.save("project2/model")
    result = model.transform(dataDF)
    # Add a new column to each based on the original label {1, 0, -1, -99}
    positive_df = result.withColumn("poslabel", udf_pos_colum('labeldjt'))
    negative_df = result.withColumn("neglabel", udf_neg_colum('labeldjt'))

    #Task 7
    try:
        posModel = CrossValidatorModel.load('project2/pos.model')
        negModel = CrossValidatorModel.load('project2/neg.model')
    except:
        posModel, negModel = train_models(positive_df, negative_df)
    # TASK 10
    task10 = get_pos_negDF(commentsDF, submissionsDF, posModel, negModel, model,
                            sanitize_udf)
    # Create a temporary view to select from
    task10.createOrReplaceTempView("dataTable")
    top10q = context.sql("""SELECT id,title, AVG(pos) AS pos_avg, AVG(neg)
                            AS neg_avg, COUNT(id) as count FROM dataTable
                            GROUP BY id,title""")
    top10q.createOrReplaceTempView("top10")
    top10_pos = context.sql("""SELECT id,title,pos_avg from top10
                             ORDER BY pos_avg DESC LIMIT 10""")
    top10_neg = context.sql("""SELECT id,title,neg_avg from top10
                             ORDER BY neg_avg DESC LIMIT 10""")
    # top10_pos.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("10_pos.csv")
    # top10_neg.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("10_neg.csv")

    times = context.sql("""SELECT from_unixtime(time,'YYYY-MM-dd') AS date,
                        AVG(pos) AS Positive, AVG(neg) AS Negative FROM
                        dataTable GROUP BY date""")
    # Filter only where the commentor has a US State flair
    task10 = task10.filter(col('state').isin(states))
    task10.createOrReplaceTempView('dataTable')
    states = context.sql("""SELECT state, AVG(pos) AS Positive, AVG(neg) AS
                          Negative, COUNT(state) from dataTable GROUP BY state""")
    # times.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("times.csv")
    # states.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("states.csv")


'''
Uses the training models on the incoming datasets
@param commentsDF - comments dataset
@param submissionsDF - submissions dataset
@param posModel - positive training model
@param negModel - negative training model
@param model - CountVectorizerModel
@param clean - sanitize function
@return - returns the trained dataset according to tasks 8 and 9
in this order (posModel, negModel)
'''
def get_pos_negDF(commentsDF, submissionsDF, posModel, negModel, model, clean):
    udf_clean = udf(clean_link, StringType())
    udf_pos = udf(get_pos_prob, IntegerType())
    udf_neg = udf(get_neg_prob, IntegerType())
    # TASKS 8,9
    # Remove sarcastic or quote comments
    commentsDF = commentsDF.filter((~commentsDF.body.like("%/s%")) &
                    (~commentsDF.body.like("&gt%"))).select("*")
    # Remove the 't3_' in the begininning of link_ids
    cleanedDF = commentsDF.withColumn("clean_link_id", udf_clean('link_id'))
    # Rename before join
    cleanedDF = cleanedDF.withColumnRenamed("score", "comment_score")
    pre_sanitizedDF = cleanedDF.join(submissionsDF,
        cleanedDF.clean_link_id == submissionsDF.id).select(
        cleanedDF['created_utc'], cleanedDF['body'], cleanedDF['comment_score'],
        cleanedDF['author_flair_text'], submissionsDF['score'],
        cleanedDF['clean_link_id'], submissionsDF['title'])
    # Sanitize the body of the comments
    sanDF = pre_sanitizedDF.withColumn('sanitized_text', clean('body'))
    result = model.transform(sanDF)
    # Avoid a join by renaming some columns here
    pos_training = posModel.transform(result).selectExpr('features',
        'comment_score as comScore', 'score as subcore',
        'clean_link_id as id', 'created_utc as time', 'body',
        'author_flair_text as state', 'prediction as p', 'rawPrediction as rP',
        'title','probability as pos_probability',
        'sanitized_text')

    neg_training = negModel.transform(pos_training)
    # Construct new columns based on ROC threshold
    neg_training = neg_training.withColumn('pos', udf_pos('pos_probability'))
    neg_training = neg_training.withColumn('neg', udf_neg('probability'))
    return neg_training

'''
Trains the two models if they are not already generated
@param pos - dataframe with the 'poslabel' column
@param neg - dataframe with the 'neglabel' column
@return - returns the positive and negative training models
in this order (posModel, negModel)
'''
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
    # Don't split the data since this dataset is much smaller than unseen data
    posTrain,posTest = pos.randomSplit([.5, .5])
    negTrain,negTest = neg.randomSplit([.5, .5])
    # Train the models
    print("Training positive classifier...")
    posModel = posCrossval.fit(posTrain)
    print("Training negative classifier..dd.")
    negModel = negCrossval.fit(negTrain)

    # Once we train the models, we don't want to do it again.
    # We can save the models and load them again later.
    posModel.save("project2/pos.model")
    negModel.save("project2/neg.model")
    return posModel, negModel


# HELPER UDF FUNCTIONSS
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
