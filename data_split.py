import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, row_number

def main(spark, userID):
    # load data
    interactions = spark.read.parquet("/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet")
    
	# Create a temporary view of the DataFrame
    interactions.createOrReplaceTempView("interactions_view")
	
	# count total number of distinct user
    print('total number of distinct user_id')
    spark.sql("SELECT COUNT(DISTINCT USER_ID) AS TOTAL FROM INTERACTIONS_VIEW").show()
	
	# count num of user_id with only one appearence
    print('num of user_id with one appearence')
    spark.sql("SELECT COUNT(*) FROM (SELECT user_id, COUNT(*) AS num_interactions FROM interactions_view GROUP BY user_id HAVING num_interactions = 1) AS temp").show()

    # Filter out users with only one appearance
    spark.sql("""
        SELECT *
        FROM interactions_view
        WHERE user_id IN (
            SELECT user_id
            FROM interactions_view
            GROUP BY user_id
            HAVING COUNT(*) > 1
        )
    """).createOrReplaceTempView("interactions_filtered_view")
    print('num of distinct user_id after filtering')
    spark.sql("SELECT COUNT(DISTINCT USER_ID) AS TOTAL FROM interactions_filtered_view").show()

    # Assign a rank to each row within each group using the RANK() function
    ranked_interactions = spark.sql("""
        SELECT *, RANK() OVER (PARTITION BY user_id ORDER BY timestamp) AS rank
        FROM interactions_filtered_view
    """)

    # Calculate the criteria value for each group
    criteria = spark.sql("""
        SELECT user_id, COUNT(*) * 0.7 AS criteria
        FROM interactions_filtered_view
        GROUP BY user_id
    """)

    # Join the ranked interactions with the criteria values
    interactions_processed = ranked_interactions.join(criteria, on="user_id")

    # Filter out rows where the rank is greater than the criteria value as train
    interactions_train = interactions_processed.filter(col("rank") <= col("criteria"))
    # the rest is validation
    interactions_validation = interactions_processed.filter(col("rank") > col("criteria"))
	
    interactions_train.createOrReplaceTempView("interactions_train_view")
    interactions_validation.createOrReplaceTempView("interactions_validation_view")
	# sanity check
	# 1. train and validation should have same num of distinct user_id
	# Count the number of distinct user IDs
    num_users_train = spark.sql("""
        SELECT COUNT(DISTINCT user_id) AS num_users
        FROM interactions_train_view
    """).collect()[0]["num_users"]
    print("Number of distinct user_id in train:", num_users_train)

    num_users_val = spark.sql("""
        SELECT COUNT(DISTINCT user_id) AS num_users
        FROM interactions_validation_view
    """).collect()[0]["num_users"]
    print("Number of distinct user_id in validation:", num_users_val)

	# 2. num of rows in train and validation should add up to the value after filtering
	# and be around 70/30
	# Count the total number of rows
    num_rows_train = interactions_train.count()
    num_rows_val = interactions_validation.count()
    print("Total number of rows in train:", num_rows_train)
    print("Total number of rows in validation:", num_rows_val)

    # Show the selected interactions
    interactions_train.limit(10).show()

    # Write train and validation set to parquet files
    print('Save to parquet')
    interactions_train.write.format("parquet").save("/user/yc6285_nyu_edu/interactions_train.parquet")
    interactions_validation.write.format("parquet").save("/user/yc6285_nyu_edu/interactions_validation.parquet")

    print('Done')
	
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('view data').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
