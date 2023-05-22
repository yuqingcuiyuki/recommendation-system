import os
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, row_number
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.sql.functions import collect_list, lit, array
from pyspark.sql.window import Window
from pyspark.sql.functions import monotonically_increasing_id


def main(spark, userID):
    
	# load data: train
    train = spark.read.parquet("/user/yc6285_nyu_edu/interactions_train.parquet") #switch to validation and test to get "train_mat"
    print('load train')
	# Create a temporary view of the DataFrame
    train.createOrReplaceTempView("train_view")
	
	# load data: val
    val = spark.read.parquet("/user/yc6285_nyu_edu/interactions_validation.parquet")
	# Create a temporary view of the DataFrame
    val.createOrReplaceTempView("val_view")

	# load data: val
    test = spark.read.parquet("/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet")
	# Create a temporary view of the DataFrame
    test.createOrReplaceTempView("test_view")
	
	# create msid map
    spark.sql("""
        SELECT recording_msid, 
               CAST(ROW_NUMBER() OVER (ORDER BY recording_msid) AS DOUBLE) AS rank
        FROM (
            SELECT DISTINCT recording_msid 
            FROM train_view
            UNION
            SELECT DISTINCT recording_msid
            FROM val_view
            UNION
            SELECT DISTINCT recording_msid 
            FROM test_view
        ) combined_views
        """).createOrReplaceTempView("msid_mapping")	
	
	################################################################
    # create "rating" matrix
    print('create rating matrix for train')
    train_mat=spark.sql("""
    SELECT t.user_id, rank as recording_msid, rating
    FROM
        (SELECT user_id, recording_msid, count(*) AS rating
        FROM train_view
        GROUP BY 1,2) AS t
    LEFT JOIN msid_mapping as m
    on t.recording_msid=m.recording_msid
    """)
    train_mat.show(5)
    
	# export train matrix as parquet
    train_mat.write.format("parquet").save("train_mat.parquet")
    print("train mat exported")  #val_mat, test_mat
	#################################################################
	
	
	#################################################################	
	# calculate truth from validation set
    print('get truth from val')
    print('step1')
    spark.sql("""
        SELECT user_id, recording_msid, COUNT(*) AS count
        FROM val_view
        GROUP BY user_id, recording_msid
	""").createOrReplaceTempView("step1")
	
    print('step2')
    spark.sql("""
        SELECT user_id, recording_msid, 
               DENSE_RANK() OVER (PARTITION BY user_id ORDER BY count DESC) AS count_rank
        FROM step1
    """).createOrReplaceTempView("step2")

    print('step3')
    spark.sql("""
        SELECT user_id, recording_msid
        FROM step2
        WHERE count_rank <=100
        ORDER BY user_id ASC, count_rank ASC
    """).createOrReplaceTempView("step3")
	
	
    top_100_truth = spark.sql("""
        SELECT user_id, rank as recording_msid
        FROM step3 as s
        LEFT JOIN msid_mapping as m
        ON s.recording_msid=m.recording_msid
    """)

    print('done')
	
	# convert truth from val to format ready to use metric
    # Group the top_100_truth DataFrame by user_id and collect the recording_msids as lists
    print('convert truth')
    truth_df = (
        top_100_truth.groupBy("user_id")
        .agg(collect_list("recording_msid").alias("label"))
    )
	
    truth_df.show(5)
    # export truth df from validation set as parquet
    truth_df.write.format("parquet").save("val_truth.parquet")
    print("val truth exported")
	#################################################################

	#################################################################	
	# calculate truth from test set
    print('get truth from test')
    print('step1')
    spark.sql("""
        SELECT user_id, recording_msid, COUNT(*) AS count
        FROM test_view
        GROUP BY user_id, recording_msid
	""").createOrReplaceTempView("step1")
	
    print('step2')
    spark.sql("""
        SELECT user_id, recording_msid, 
               DENSE_RANK() OVER (PARTITION BY user_id ORDER BY count DESC) AS count_rank
        FROM step1
    """).createOrReplaceTempView("step2")

    print('step3')
    spark.sql("""
        SELECT user_id, recording_msid
        FROM step2
        WHERE count_rank <=100
        ORDER BY user_id ASC, count_rank ASC
    """).createOrReplaceTempView("step3")
	
	
    top_100_truth = spark.sql("""
        SELECT user_id, rank as recording_msid
        FROM step3 as s
        LEFT JOIN msid_mapping as m
        ON s.recording_msid=m.recording_msid
    """)

    print('done')
	
	# convert truth from test to format ready to use metric
    # Group the top_100_truth DataFrame by user_id and collect the recording_msids as lists
    print('convert truth')
    truth_df = (
        top_100_truth.groupBy("user_id")
        .agg(collect_list("recording_msid").alias("label"))
    )
	
    truth_df.show(5)
    # export truth df from test set as parquet
    truth_df.write.format("parquet").save("test_truth.parquet")
    print("test truth exported")
	#################################################################
	
	
	
# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    config = SparkConf().setAll([('spark.executor.memory', '8g'), \
                                         ('spark.driver.memory', '8g'), \
                                         ('spark.blacklist.enabled', False), \
                                         ('spark.serializer', 'org.apache.spark.serializer.KryoSerializer'), \
                                         ('spark.sql.autoBroadcastJoinThreshold', 100 * 1024 * 1024)])
    
    # Create the spark session object
    spark = SparkSession.builder.appName('preprocess').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']
    # create spark context
    sc = spark.sparkContext
    # Call our main routine
    main(spark, userID)

