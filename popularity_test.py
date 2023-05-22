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
    # load data
    train = spark.read.parquet("/user/yc6285_nyu_edu/interactions_train.parquet")
    val = spark.read.parquet("/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet")



	# Create a temporary view of the DataFrame
    train.createOrReplaceTempView("train_view")
    val.createOrReplaceTempView("val_view")

	
	# select top 100 msid based on appearences
	# they will be recommended to all users
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
        ) combined_views
        """).createOrReplaceTempView("msid_mapping")	
	
	
    print('get recommendation')
    print('Find the top 100 popular music msid from the training set')
    spark.sql("""
        SELECT recording_msid, num_occ, num_user, (1-0.75)*num_occ+0.75*num_user as score 
        FROM (
            SELECT recording_msid, count(*) as num_occ, count(distinct user_id) as num_user 
            FROM train_view 
            GROUP BY recording_msid 
        ) as t
        ORDER BY score DESC 
        LIMIT 100;
    """).createOrReplaceTempView('recommend_msids')
	
    top_100_recommend=spark.sql("""
        SELECT rank as recording_msid
        FROM recommend_msids as r
        LEFT JOIN msid_mapping as m
        on r.recording_msid=m.recording_msid
	""")
    print('done')
	
	
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
	
	# convert recommendation and truth from val to format ready to use metric

    # Convert the top_100_recommend table into a dictionary
    print('convert recommend')
    top_100_recommend=top_100_recommend.repartition(100)
    recommendations = (
        top_100_recommend.select("recording_msid")
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    # Convert the Python list to a list of literals
    print('to literals')
    recommendation_literals = [lit(r) for r in recommendations]

    # Create a DataFrame with user_id and their recommendations
    print('to df')
    recommendation_df = (
        top_100_truth.select("user_id")
        .distinct()
        .withColumn("prediction", array(recommendation_literals))
    )

    # Group the top_100_truth DataFrame by user_id and collect the recording_msids as lists
    print('convert truth')
    truth_df = (
        top_100_truth.groupBy("user_id")
        .agg(collect_list("recording_msid").alias("label"))
    )

    # Join recommendation_df and truth_df on user_id
    print('join')
    joined_df = recommendation_df.join(truth_df, on="user_id")
    total_rows=joined_df.count()
    print(total_rows, len(joined_df.columns))
	
    print(joined_df.columns)
	
    evaluator = RankingEvaluator()
    evaluator.setPredictionCol("prediction")
    print('calculate metric')
    joined_df = joined_df.repartition(100) 
    metric = evaluator.evaluate(joined_df)
	# Convert the joined DataFrame into an RDD of (user_id, (recommendations, truth))
    # print('convert to rdd')
    # prediction_and_labels = joined_df.rdd.map(
    #     lambda row: (row["user_id"], (row["recommendations"], row["truth"]))
    # )

    print("Mean Average Precision:", metric)
	
	
# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    config = SparkConf().setAll([('spark.executor.memory', '8g'), \
                                         ('spark.driver.memory', '8g'), \
                                         ('spark.blacklist.enabled', False), \
                                         ('spark.serializer', 'org.apache.spark.serializer.KryoSerializer'), \
                                         ('spark.sql.autoBroadcastJoinThreshold', 100 * 1024 * 1024)])
    
    # Create the spark session object
    spark = SparkSession.builder.appName('popularity').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']
    # create spark context
    sc = spark.sparkContext
    # Call our main routine
    main(spark, userID)
