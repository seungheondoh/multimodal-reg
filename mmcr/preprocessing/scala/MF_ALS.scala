package com.naver.now.batch

import com.naver.now.common.utils.BatchProcess
import org.apache.spark.sql.{SparkSession, DataFrame, Row}
import org.apache.spark.sql.functions._
import com.naver.now.data.{QoeLog, RecMetaAll}
import com.naver.now.common.utils.RefineUtils.{refineQuery, clusteringEntity, zipEntityVideoScore, rerankByPopScore, getEntityVideos}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.recommendation.ALS
import com.naver.now.data.{Channel, Entity, EntityWeight, NETEntity, Person, PersonCastType, Video, VideoWeight, VideoScore}

object Filtering extends BatchProcess {
  private val NPARAMS: Int = 2
  private def printUsage(): Unit = {
    val usage: String = {
      "smsg_emp_list\n" +
      "dt\n"
    }
    println(usage)
  }
	
  private def parseArgs(args: Array[String]): Array[String] = {
    if (args.length != NPARAMS) {
      printUsage()
      System.exit(1)
    }
    args
  }

  def custom_als(spark: SparkSession, outputPath: String, split: String, iter_filtered_qoe: DataFrame): Unit ={
    import spark.implicits._
    val userIndexDF = iter_filtered_qoe.select("id_no").distinct.rdd.zipWithIndex.map(x => (x._1.getString(0), x._2)).toDF("id_no", "userId").repartition(100)
    val itemIndexDF = iter_filtered_qoe.select("media_id").distinct.rdd.zipWithIndex.map(x => (x._1.getString(0), x._2)).toDF("media_id", "mediaId").repartition(100)
    itemIndexDF.write.option("compression", "none").save(s"${outputPath}/${split}/itemIndex")
    userIndexDF.write.option("compression", "none").save(s"${outputPath}/${split}/userIndex")
    val userIndex = spark.read.parquet(s"${outputPath}/${split}/userIndex")
    val itemIndex = spark.read.parquet(s"${outputPath}/${split}/itemIndex")
    val userItemsIndex = iter_filtered_qoe.join(userIndex, Seq("id_no")).join(itemIndex, Seq("media_id"))
    val u_max_ds = userItemsIndex.groupBy("id_no").agg(max("ds").as("last_ds"))
    val u_min_ds = userItemsIndex.groupBy("id_no").agg(min("ds").as("start_ds"))
    val als = new ALS().setMaxIter(15).setRegParam(0.1).setRank(128).setNumBlocks(128).setImplicitPrefs(true).setUserCol("userId").setItemCol("mediaId").setRatingCol("score")
    val model = als.fit(userItemsIndex)
    model.write.save(s"${outputPath}/${split}/models")
  }


  override def run(spark: SparkSession, args: Array[String]): Unit = {
    import spark.implicits._
    val parsedArgs = parseArgs(args)
    val dt = parsedArgs(1)
    /* load data */
    val outputPath = "hdfs://pct/user/nowrec/tmp/seungheon/small/WMF"
    val iter_filtered_qoe = spark.read.parquet("hdfs://pct/user/nowrec/tmp/seungheon/small/iter_filtered_qoe")
    val train_data = iter_filtered_qoe.where($"split"==="train").groupBy($"id_no", $"media_id", $"ds").agg(sum($"score").as("score")).where($"score".isNotNull) //log를 추가해야함
    val valid_data = iter_filtered_qoe.where($"split"==="valid").groupBy($"id_no", $"media_id", $"ds").agg(sum($"score").as("score")).where($"score".isNotNull) //log를 추가해야함
    val test_data = iter_filtered_qoe.where($"split"==="test").groupBy($"id_no", $"media_id", $"ds").agg(sum($"score").as("score")).where($"score".isNotNull) //log를 추가해야함
    custom_als(spark, outputPath, "train", train_data)
    custom_als(spark, outputPath, "valid", valid_data)
    custom_als(spark, outputPath, "test", test_data)
  }
}
