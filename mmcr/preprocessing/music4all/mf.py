import sys
import os
import json
import pandas as pd
import argparse
import pyspark
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, FloatType, IntegerType, LongType
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.utils.notebook_utils import is_jupyter
from recommenders.datasets.spark_splitters import spark_random_split
from recommenders.evaluation.spark_evaluation import SparkRatingEvaluation, SparkRankingEvaluation
from recommenders.utils.spark_utils import start_or_get_spark

print("System version: {}".format(sys.version))
print("Spark version: {}".format(pyspark.__version__))
spark = start_or_get_spark("ALS PySpark", memory="64g")
spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")

TOP_K = 10
COL_USER = "user_id"
COL_ITEM = "track_id"
COL_RATING = "count"


def als(df_user):
    data = spark.createDataFrame(df_user)
    header = {
        "userCol": COL_USER,
        "itemCol": COL_ITEM,
        "ratingCol": COL_RATING,
    }
    # 
    als = ALS(
        rank=128,
        maxIter=15,
        implicitPrefs=False,
        regParam=0.05,
        coldStartStrategy='drop',
        nonnegative=False,
        seed=42,
        **header
    )
    model = als.fit(data)
    model.save(os.path.join(args.save_path, "ALS"))
    
def main(args):
    df_user = pd.read_csv(os.path.join(args.save_path, "music4all-onion/userid_trackid_count.tsv.bz2"), compression="bz2", sep="\t")
    track_encoding = json.load(open(os.path.join(args.save_path, "music4all-cold/track_encoding.json"),'r'))
    # log scale 적용해보자!
    df_user['track_id'] = df_user['track_id'].map(track_encoding)
    als(df_user)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default="../../../dataset")
    args = parser.parse_args()
    main(args)

spark.stop()