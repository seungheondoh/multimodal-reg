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
    val video_meta = spark.read.parquet("hdfs://pct/user/nowrec/rec_meta_all.real/refine/daily_batch/dt=latest/RecMetaAll") // 최근 60일 비디오 데이터임 => trend 가 반영된 video quality
    // val video_meta = spark.read.parquet("hdfs://pct/user/nowrec/tmp/seungheon/video_meta")
    /* filtering params */
    val l_target_columns = List("log_time", "watch_time", "id_no", "id_country", "age", "gender", "fraud_level", "media_id", "live", "duration", "ds")
    val minItemNum = 20 // For user filter
    val minUserNum = 250 // For item filter
    val minScore = 0.3
    val minWatchTime = 30000 // 30sec
    val maxWatchTime = 7200000 // 2hour

    /* code */
    val df = qoe_log.select(l_target_columns.head, l_target_columns.tail: _*) /* 3348150210 */
    val df_nofraud = df.where($"fraud_level" === 0)  /* 3111891146 */
    val df_nolive = df_nofraud.where($"live" === false)  /* 2467120191 */
    val df_dropnull = df_nolive.where($"id_no".isNotNull && $"id_no" =!= "") /* 1701135586 */
    val df_score = df_dropnull.withColumn("score", $"watch_time" / $"duration")
    val df_filtered = df_score.where(($"watch_time" >= minWatchTime) && ($"watch_time" <= maxWatchTime)) /* 1055806903 */
    val df_score_filtered = df_filtered.where($"score".isNotNull && $"score" >= minScore) /* 1055806903 */
    val save_media_id = df_score_filtered.select($"media_id").intersect(video_meta.select($"media_id")) /* unique_item 3305223 */
    val static_filtered_qoe = df_score_filtered.join(save_media_id, Seq("media_id"), "inner").withColumn("itemCnt", count($"media_id").over(Window.partitionBy($"id_no"))).withColumn("userCnt", count($"id_no").over(Window.partitionBy($"media_id"))) /* log: 957160437, unique_item: 3305223 */
    static_filtered_qoe.write.parquet("hdfs://pct/user/nowrec/tmp/seungheon/static_filtered_qoe")

    println("start filtering")
    val static_filtered_qoe = spark.read.parquet("hdfs://pct/user/nowrec/tmp/seungheon/static_filtered_qoe")
    // fail to build recursive function
    val iter_filtered_qoe_1st = static_filtered_qoe.where($"itemCnt" >= minItemNum && $"userCnt" >= minUserNum).drop("itemCnt", "userCnt") /* log 610358647 */
    val iter_filtered_qoe_2nd = iter_filtered_qoe_1st.withColumn("itemCnt", count($"media_id").over(Window.partitionBy($"id_no"))).withColumn("userCnt", count($"id_no").over(Window.partitionBy($"media_id"))).where($"itemCnt" >= minItemNum && $"userCnt" >= minUserNum).drop("itemCnt", "userCnt") /* log 599138959 */
    val iter_filtered_qoe_3rd = iter_filtered_qoe_2nd.withColumn("itemCnt", count($"media_id").over(Window.partitionBy($"id_no"))).withColumn("userCnt", count($"id_no").over(Window.partitionBy($"media_id"))).where($"itemCnt" >= minItemNum && $"userCnt" >= minUserNum).drop("itemCnt", "userCnt") 
    val iter_filtered_qoe_4th = iter_filtered_qoe_3rd.withColumn("itemCnt", count($"media_id").over(Window.partitionBy($"id_no"))).withColumn("userCnt", count($"id_no").over(Window.partitionBy($"media_id"))).where($"itemCnt" >= minItemNum && $"userCnt" >= minUserNum).drop("itemCnt", "userCnt") 
    val iter_filtered_qoe = iter_filtered_qoe_4th.withColumn("itemCnt", count($"media_id").over(Window.partitionBy($"id_no"))).withColumn("userCnt", count($"id_no").over(Window.partitionBy($"media_id"))).where($"itemCnt" >= minItemNum && $"userCnt" >= minUserNum) /* log 595244386 */ 
    iter_filtered_qoe.write.parquet("hdfs://pct/user/nowrec/tmp/seungheon/small/iter_filtered_qoe")

    // filter out media
    val iter_filtered_qoe = spark.read.parquet("hdfs://pct/user/nowrec/tmp/seungheon/small/iter_filtered_qoe")
    val u_media_id = iter_filtered_qoe.select("media_id").distinct()
    val filterd_video_meta = video_meta.join(u_media_id, Seq("media_id"), "inner")
    val channel_summary = filterd_video_meta.groupBy("channel_category").count().sort($"count".desc)
    val v_viewCnt = iter_filtered_qoe.groupBy("media_id").count()
    val avg_watchTime = iter_filtered_qoe.groupBy("media_id").mean("watch_time")
    val v_meta = filterd_video_meta.join(v_viewCnt, Seq("media_id"), "inner").join(avg_watchTime, Seq("media_id"), "inner")
    v_meta.write.parquet("hdfs://pct/user/nowrec/tmp/seungheon/small/v_meta") // small file

    val u_viewCnt = iter_filtered_qoe.groupBy("id_no").count()
    val u_max_ds = iter_filtered_qoe.groupBy("id_no").agg(max("ds").as("last_ds"))
    val u_min_ds = iter_filtered_qoe.groupBy("id_no").agg(min("ds").as("start_ds"))
    val avg_watchTime = iter_filtered_qoe.groupBy("id_no").mean("watch_time")
    val u_age = iter_filtered_qoe.groupBy("id_no").agg(max("age").as("age"))
    val u_gender = iter_filtered_qoe.groupBy("id_no").agg(max("gender").as("gender"))
    val user_meta = u_viewCnt.join(u_max_ds, Seq("id_no"), "inner").join(u_min_ds, Seq("id_no"), "inner").join(avg_watchTime, Seq("id_no"), "inner").join(u_age, Seq("id_no"), "inner").join(u_gender, Seq("id_no"), "inner").withColumn("dateDiff", datediff($"last_ds", $"start_ds"))
    user_meta.write.parquet("hdfs://pct/user/nowrec/tmp/seungheon/small/u_meta") // small file

    // media = {
    //   "media_id"
    //   "avg_watch_time",
    //   "duration",
    //   "view_count:"
    //   "start_broadcast"
    //   "end_broadcast"
    // }
  }
}