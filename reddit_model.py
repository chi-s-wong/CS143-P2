from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext


def main(context):
    """Main function takes a Spark SQL context."""
    # YOUR CODE HERE
    # YOU MAY ADD OTHER FUNCTIONS AS NEEDED

    ### TASK 1
    ## Load data
    # commentsDF = context.read.json("comments-minimal.json.bz2")
    # submissionsDF = context.read.json("submissions.json.bz2")
    # labelsDF = context.read.csv("labeled_data.csv")

    ## Write data into parquet files for faster loading in the future
    # commentsDF.write.parquet("comments.pqt")
    # submissionsDF.write.parquet("submissions.pqt")
    # labelsDF.write.parquet("labels.pqt")

    ### TASK 2
    ## Join labelsDF and commentsDF
    ## Question 1: F = {id -> label_dem,label_gop,label_djt)
    ## Question 2:
    
    # Yes, this table seems normalized. The collector stored it this way because it was the most straightforward way of storing the comment ID and its associated labels
    # dataDF = labelsDF.join(commentsDF, labelsDF._c0 == commentsDF.id)

    
    ### TASK 4 + 5
    from cleantext import sanitize
    sanitize_udf = spark.udf.register("sanitize", sanitize)
    dataDF = dataDF.withColumn("sanitized_text", sanitize_udf('body'))
    dataDF.write.parquet("sanitized_data.pqt")

    ### TASK 6A
    


if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)
