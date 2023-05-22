#optimize_version
#Cache frequently used DataFrames: If you are going to use the same DataFrame multiple times, cache it to avoid recomputing it each time.
#Use broadcast joins when joining small DataFrames with large DataFrames. This helps in reducing network shuffling.
#Use the same indexer for both train_data and validation_data to make sure they share the same indexed values for recording_msid.
#single_model_train
import numpy
import sys
import pyspark
import os
import time

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import collect_list, col, udf
from pyspark.sql.functions import collect_list, lit, array
from pyspark.sql.types import ArrayType, IntegerType

def main(spark, userID):

    # load data	
    train_mat = spark.read.parquet("/user/yc6285_nyu_edu/train_mat.parquet")
    val_truth = spark.read.parquet("/user/yc6285_nyu_edu/val_truth.parquet")

    # filter train_mat
    train_mat.createOrReplaceTempView("train_mat")
    spark.sql("""
              SELECT user_id, recording_msid, rating
              FROM TRAIN_MAT
              WHERE RATING>4
              """).createOrReplaceTempView("step1")

    train_mat = spark.sql("""
        SELECT t.user_id, t.recording_msid, t.rating
        FROM step1 t
        WHERE t.recording_msid IN (
            SELECT recording_msid
            FROM step1
            GROUP BY recording_msid
            HAVING COUNT(DISTINCT user_id) >= 5)
        """)
		
    print('filtering done')

    regParam=0.0001
    rank=50
    alpha=0.5 #use best alpha
    fractions=[0.25, 0.5, 0.75, 1.0]
    for frac in fractions:
        # Randomly sample 25% of rows
        sample = train_mat.sample(fraction=frac, seed=42)	
        # Create the ALS model object with specified parameters
        start_time = time.time()
        print('create als model')
        als = ALS(maxIter=5,regParam=regParam,rank=rank,alpha=alpha,
                  userCol="user_id", itemCol="recording_msid", ratingCol="rating",
                  coldStartStrategy="drop", nonnegative=True)

        model = als.fit(train_mat)
        predictions=model.transform(train_mat)
        predictions.createOrReplaceTempView("step0")
        spark.sql("""
            SELECT user_id, recording_msid, 
                   DENSE_RANK() OVER (PARTITION BY user_id ORDER BY prediction DESC) AS count_rank
            FROM step0
        """).createOrReplaceTempView("middle")
        als_result_df=spark.sql("""
            SELECT user_id, recording_msid
            FROM middle
            WHERE count_rank <=100
            ORDER BY user_id ASC, count_rank ASC
        """)
        als_result = (
            als_result_df.groupBy("user_id")
            .agg(collect_list("recording_msid").alias("prediction"))
        )
        elapsed_time = time.time() - start_time
        print(f'Time taken for fraction={frac}: {elapsed_time:.2f} seconds')

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('view data').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)










