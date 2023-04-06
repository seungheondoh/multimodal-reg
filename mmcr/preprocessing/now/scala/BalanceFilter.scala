import com.naver.now.common.utils.BatchProcess
import org.apache.spark.sql.{SparkSession, DataFrame, Row}
import org.apache.spark.sql.functions._
import com.naver.now.data.{QoeLog, RecMetaAll}
import com.naver.now.common.utils.RefineUtils.{refineQuery, clusteringEntity, zipEntityVideoScore, rerankByPopScore, getEntityVideos}
import org.apache.spark.sql.expressions.Window
import com.naver.now.data.{Channel, Entity, EntityWeight, NETEntity, Person, PersonCastType, Video, VideoWeight, VideoScore}

object Preprocessing extends BatchProcess {
  private val NPARAMS: Int = 3
  private def printUsage(): Unit = {
    val usage: String = {
      "smsg_emp_list\n" +
      "dt\n" +
      "threshold\n"
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

  override def run(spark: SparkSession, args: Array[String]): Unit = {
    import spark.implicits._
    val parsedArgs = parseArgs(args)
    val dt = parsedArgs(1)
    val threshold = parsedArgs(2)
    /* load data */
    val qoe_log = spark.read.parquet("hdfs://pct/user/now/warehouse/videoinfra/qoe") // 3월부터의 로그, 
    // val video_meta = spark.read.parquet("hdfs://pct/user/nowrec/rec_meta_all.real/refine/daily_batch/dt=latest/RecMetaAll") // 최근 60일 비디오 데이터임 => trend 가 반영된 video quality
    val video_meta = spark.read.parquet("hdfs://pct/user/nowrec/tmp/seungheon/video_meta")
    
    /* filtering params */
    val l_target_columns = List("log_time", "watch_time", "id_no", "id_country", "age", "gender", "fraud_level", "media_id", "live", "duration", "ds")
    val minItemNum = 250 // For user filter
    val minUserNum = 20 // For item filter
    val minScore = 0.3
    val minWatchTime = 30000 // 30sec
    val maxWatchTime = 7200000 // 2hour
    println("start filtering")
    val static_filtered_qoe = spark.read.parquet("hdfs://pct/user/nowrec/tmp/seungheon/static_filtered_qoe")
    = spark.read.parquet("hdfs://pct/user/nowrec/tmp/seungheon/balance/v_meta")
    // fail to build recursive function
    val iter_filtered_qoe_1st = static_filtered_qoe.where($"itemCnt" >= minItemNum && $"userCnt" >= minUserNum).drop("itemCnt", "userCnt") /* log 610358647 */
    val iter_filtered_qoe_2nd = iter_filtered_qoe_1st.withColumn("itemCnt", count($"media_id").over(Window.partitionBy($"id_no"))).withColumn("userCnt", count($"id_no").over(Window.partitionBy($"media_id"))).where($"itemCnt" >= minItemNum && $"userCnt" >= minUserNum).drop("itemCnt", "userCnt") /* log 599138959 */
    val iter_filtered_qoe_3rd = iter_filtered_qoe_2nd.withColumn("itemCnt", count($"media_id").over(Window.partitionBy($"id_no"))).withColumn("userCnt", count($"id_no").over(Window.partitionBy($"media_id"))).where($"itemCnt" >= minItemNum && $"userCnt" >= minUserNum).drop("itemCnt", "userCnt") 
    val iter_filtered_qoe_4th = iter_filtered_qoe_3rd.withColumn("itemCnt", count($"media_id").over(Window.partitionBy($"id_no"))).withColumn("userCnt", count($"id_no").over(Window.partitionBy($"media_id"))).where($"itemCnt" >= minItemNum && $"userCnt" >= minUserNum).drop("itemCnt", "userCnt") 
    val iter_filtered_qoe = iter_filtered_qoe_4th.withColumn("itemCnt", count($"media_id").over(Window.partitionBy($"id_no"))).withColumn("userCnt", count($"id_no").over(Window.partitionBy($"media_id"))).where($"itemCnt" >= minItemNum && $"userCnt" >= minUserNum) /* log 595244386 */ 

    val u_media_id = iter_filtered_qoe.select("media_id").distinct()
    val filterd_video_meta = video_meta.join(u_media_id, Seq("media_id"), "inner")
    val channel_summary = filterd_video_meta.groupBy("channel_category").count().sort($"count".desc)
    val df_news = filterd_video_meta.filter($"channel_category" === "NEWS").limit(10000)
    val df_enter = filterd_video_meta.filter($"channel_category" === "ENTER").limit(10000)
    val df_drama = filterd_video_meta.filter($"channel_category" === "DRAMA").limit(10000)
    val df_sport = filterd_video_meta.filter($"channel_category" === "SPORT").limit(10000)
    val df_artis = filterd_video_meta.filter($"channel_category" === "ARTIS").limit(10000)
    val df_balance_vmeta = df_news.union(df_enter).union(df_drama).union(df_sport).union(df_artis)
    val balance_media_id = df_balance_vmeta.select("media_id").distinct()
    val balaneced_filtered_qoe = iter_filtered_qoe.join(balance_media_id, Seq("media_id"), "inner")
    // datasplit
    val item_to_ds = balaneced_filtered_qoe.groupBy("media_id").agg(min("ds").as("upload_ds"))
    // df_item_count.agg(sum("count")) == 39245
    val df_item_count = item_to_ds.groupBy("upload_ds").count().sort($"upload_ds".asc).withColumn("total", lit(50000))
    val window = Window.partitionBy($"total").orderBy($"upload_ds".asc).rowsBetween(Window.unboundedPreceding, Window.currentRow)
    val df_cumsum = df_item_count.withColumn("cumsum", sum($"count").over(window)).withColumn("cum_ratio", $"cumsum"/$"total")
    val tr_ratio = 0.9
    val va_ratio = 0.95
    val test_ds = df_cumsum.where(($"cum_ratio" > va_ratio)).select("upload_ds").withColumn("wc_split", lit("c_test")).withColumn("split", lit("test"))
    val val_ds = df_cumsum.where(($"cum_ratio" >= tr_ratio) && ($"cum_ratio" <= va_ratio)).select("upload_ds").withColumn("wc_split", lit("c_valid")).withColumn("split", lit("valid"))
    val train_ds = df_cumsum.where(($"cum_ratio" < tr_ratio)).select("upload_ds").withColumn("wc_split", lit("c_train"))
    val Array(warm_train_valid, warm_test) = train_ds.randomSplit(Array(0.90, 0.1))
    val Array(warm_train, warm_valid) = warm_train_valid.randomSplit(Array(0.90, 0.1))
    val df_warm_train = warm_train.withColumn("wc_split", lit("w_train")).withColumn("split", lit("train"))
    val df_warm_valid = warm_valid.withColumn("wc_split", lit("w_valid")).withColumn("split", lit("valid"))
    val df_warm_test = warm_test.withColumn("wc_split", lit("w_test")).withColumn("split", lit("test"))
    val df_warm = df_warm_train.union(df_warm_valid).union(df_warm_test)

    val df_split = df_warm.union(val_ds).union(test_ds)

    val split_media_id = item_to_ds.join(df_split,Seq("upload_ds"), "inner") // 153811
    val qoe_with_split = balaneced_filtered_qoe.join(split_media_id, Seq("media_id"), "inner")

    val u_viewCnt = qoe_with_split.groupBy("id_no").count()
    val u_max_ds = qoe_with_split.groupBy("id_no").agg(max("ds").as("last_ds"))
    val u_min_ds = qoe_with_split.groupBy("id_no").agg(min("ds").as("start_ds"))
    val avg_watchTime = qoe_with_split.groupBy("id_no").mean("watch_time")
    val u_age = qoe_with_split.groupBy("id_no").agg(max("age").as("age"))
    val u_gender = qoe_with_split.groupBy("id_no").agg(max("gender").as("gender"))
    val user_meta = u_viewCnt.join(u_max_ds, Seq("id_no"), "inner").join(u_min_ds, Seq("id_no"), "inner").join(avg_watchTime, Seq("id_no"), "inner").join(u_age, Seq("id_no"), "inner").join(u_gender, Seq("id_no"), "inner").withColumn("dateDiff", datediff($"last_ds", $"start_ds"))
    qoe_with_split.write.parquet("hdfs://pct/user/nowrec/tmp/seungheon/balance/iter_filtered_qoe")
    df_balance_vmeta.write.parquet("hdfs://pct/user/nowrec/tmp/seungheon/balance/v_meta")
    user_meta.write.parquet("hdfs://pct/user/nowrec/tmp/seungheon/balance/u_meta") // small file
    println(qoe_with_split.count(), df_balance_vmeta.count(), user_meta.count(), qoe_with_split.select("media_id").distinct().count(), qoe_with_split.select("id_no").distinct().count())
  }
}