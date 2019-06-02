from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from cleantext import sanitize
from pyspark.sql.types import ArrayType
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
from pyspark.ml.feature import CountVectorizer



def main(context):
    """Main function takes a Spark SQL context."""
    # YOUR CODE HERE
    # YOU MAY ADD OTHER FUNCTIONS AS NEEDED

    # commentsDF = context.read.json("comments-minimal.json.bz2")
    # submissionsDF = context.read.json("submissions.json.bz2")
    # labelsDF = context.read.csv("labeled_data.csv", header=True)

    ### TASK 1
    ## Run the following if you already have the parquet files
    commentsDF = sqlContext.read.parquet('comments.pqt')
    labelsDF = sqlContext.read.parquet('labels.pqt')

    # Load data from files


    # Write data into parquet files for faster loading in the future
    # commentsDF.write.parquet("comments.pqt")
    # submissionsDF.write.parquet("submissions.pqt")
    # labelsDF.write.parquet("labels.pqt")


    ### TASK 2
    ## Join labelsDF and commentsDF
    ## Question 1: F = {id -> label_dem,label_gop,label_djt)
    ## Question 2:
    # Yes, this table seems normalized. The collector stored it this way because it was the most straightforward way of storing the comment ID and its associated labels

    dataDF = labelsDF.join(commentsDF, labelsDF.Input_id == commentsDF.id)


    # ### TASK 4 + 5
    sanitize_udf = udf(sanitize, ArrayType(StringType()))
    dataDF = dataDF.withColumn("sanitized_text", sanitize_udf('body'))
    # dataDF.write.parquet("sanitized_data.pqt")

    ### TASK 6A
    cv = CountVectorizer(inputCol="sanitized_text", outputCol="features", binary=True, minTF=10)
    model = cv.fit(dataDF)
    print(model)
    result = model.transform(dataDF)
    result.show(truncate=True)
    # This throws the following exception
    # IllegalArgumentException: 'requirement failed: Column sanitized_text must
    # be of type equal to one of the following types: [array<string>, array<string>]
    # but was actually of type string.'



if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)
