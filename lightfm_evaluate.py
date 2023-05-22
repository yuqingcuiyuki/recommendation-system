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
    
    no_components=30 #10, 20, 30
    item_alpha=[1e-4,1e-2,1e-1]
    user_alpha=[1e-4,1e-2,1e-1]
	
    truth_df = spark.read.parquet("/user/yc6285_nyu_edu/val_truth.parquet")
	
	
    for i_alpha in item_alpha:
        for u_alpha in user_alpha:
            file_name=(f'train_pred_{no_components}_{i_alpha}_{u_alpha}.parquet')
            file_path = "/user/yc6285_nyu_edu/"+ file_name

            # load data
            train_result = spark.read.parquet(file_path)

            # Join recommendation_df and truth_df on user_id
            print('join')
            joined_df = train_result.join(truth_df, on="user_id")
            print(joined_df.columns)
	
            evaluator = RankingEvaluator()
            evaluator.setPredictionCol("predictions")
            print('calculate metric')

            metric = evaluator.evaluate(joined_df)
            print(f"no_components: {no_components}, item_alpha: {i_alpha}, user_alpha: {u_alpha}, MAP: {metric}")
	
	
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('view data').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
