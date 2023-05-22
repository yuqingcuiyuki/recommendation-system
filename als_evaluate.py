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
    
    list_regParam=[0.0001, 0.01]
    list_rank=[10, 50, 100]
    list_alpha=[0.1,0.5,1]	

    truth_df = spark.read.parquet("/user/yc6285_nyu_edu/val_truth.parquet")
	
    # Create the ALS model object with specified parameters
    for rank in list_rank:
        for regParam in list_regParam:
            for alpha in list_alpha:
                file_name=(f'train_pred_{rank}_{regParam}_{alpha}.parquet')
                file_path = "/user/yc6285_nyu_edu/"+ file_name

                # load data
                train_result = spark.read.parquet(file_path)

                # Join recommendation_df and truth_df on user_id
                print('join')
                joined_df = train_result.join(truth_df, on="user_id")
                print(joined_df.columns)
	
                evaluator = RankingEvaluator()
                evaluator.setPredictionCol("prediction")
                print('calculate metric')

                metric = evaluator.evaluate(joined_df)
                print(f"rank: {rank}, regPara: {regParam}, alpha: {alpha}, MAP: {metric}")
	
	
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('view data').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
