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

states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia',
'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',
'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia',
'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

def main(context):
    udf_pos_colum = udf(pos_column, IntegerType())
    udf_neg_colum = udf(neg_column, IntegerType())

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
    cv = CountVectorizer(inputCol="sanitized_text", outputCol="features",
                          binary=True, minDF=10)
    model = cv.fit(dataDF)
    result = model.transform(dataDF)
    positive_df = result.withColumn("poslabel", udf_pos_colum('labeldjt'))
    negative_df = result.withColumn("neglabel", udf_neg_colum('labeldjt'))

    try:
        posModel = CrossValidatorModel.load('project2/pos.model')
        negModel = CrossValidatorModel.load('project2/neg.model')
    except:
        posModel, negModel = train_models(positive_df, negative_df)


    try:
        task10 = sqlContext.read.parquet("task10.pqt")
    except:
        task10 = get_pos_negDF(dataDF, submissionsDF, posModel, negModel, model,
                                sanitize_udf)
        task10.write.parquet("task10.pqt")
    task10.show(n=50)


def get_pos_negDF(dataDF, submissionsDF, posModel, negModel, model, sanitize):
    udf_clean = udf(clean_link, StringType())
    udf_pos = udf(get_pos_prob, IntegerType())
    udf_neg = udf(get_neg_prob, IntegerType())
    # # TASK 8
    # # Remove sarcastic or quote comments
    commentsDF = dataDF.filter((~dataDF.body.like("%/s%")) &
                    (~dataDF.body.like("&gt%"))).select("*")
    cleanedDF = commentsDF.withColumn("clean_link_id", udf_clean('link_id'))
    pre_sanitizedDF = cleanedDF.join(submissionsDF,
        cleanedDF.clean_link_id == submissionsDF.id).select(
        cleanedDF['created_utc'], cleanedDF['body'],
        cleanedDF['author_flair_text'], submissionsDF['score'],
        cleanedDF['clean_link_id'], submissionsDF['title'])
    sanDF = pre_sanitizedDF.withColumn('sanitized_text',sanitize('body'))
    result = model.transform(sanDF)
    pos_training = posModel.transform(result).selectExpr('features',
        'clean_link_id as id', 'created_utc as time', 'body',
        'author_flair_text as state', 'title','probability as pos_probability',
        'sanitized_text')
    both_training = negModel.transform(pos_training)
    both_with_pos = both_training.withColumn('pos', udf_pos('pos_probability'))
    return both_with_pos.withColumn('neg', udf_neg('probability'))

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
