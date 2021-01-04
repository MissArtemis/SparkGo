import com.zjh.sf.function.Discrete
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types._
import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.ml.classification.LogisticRegression


class DiscreteTest extends AnyFunSuite{
  test("Discrete"){
    val sparkSession = SparkSession.builder().master("local").appName("testDiscrete").getOrCreate()
    val schema = StructType(List(
      StructField("A",IntegerType,nullable = false),
      StructField("B",StringType,nullable = false),
      StructField("C",StringType,nullable = false),
      StructField("D",StringType,nullable = false),
      StructField("E",IntegerType,nullable = false),
      StructField("F",DoubleType,nullable = false),
      StructField("G",StringType,nullable = false),
      StructField("H",StringType,nullable = false),
      StructField("I",DoubleType, nullable=false)
    )
      )
    val rdd = sparkSession.sparkContext.parallelize(Seq(
      Row(1,"b1","c1","d1",3,2.3,"2021-01-04 10:01:12.000","2008-08-08 10:59:59.000",1.3),
      Row(0,"b2","c2","d2",3,4.5,"2021-01-01 09:01:31.001","2012-12-12 12:12:12.001",1.7),
      Row(1,"b3","c3","d3",2,8.7,"2021-01-02 23:21:22.002","2014-11-13 15:56:58.002",2.9),
      Row(1,"b1","c2","d2",2,1.1,"2020-12-30 23:00:00.003","2015-05-06 17:36:38.003",6.1),
      Row(0,"b2","c1","d1",1,5.4,"2020-12-29 21:09:09.004","2019-09-10 16:46:48.004",7.5),
      Row(0,"b2","c5","d9",10,21.7,"2021-01-03 19:07:07.005","2018-08-13 18:41:42.005",0.9),
      Row(0,"b1","c1","d3",9,7.5,"2020-12-27 13:00:03.006","2017-09-10 03:31:14.006",0.7),
      Row(0,"b5","c2","d1",6,6.6,"2020-12-31 15:30:33.007","2017-07-07 07:17:37.007",7.1),
      Row(1,"b2","c4","d4",5,6.4,"2021-01-01 22:22:22.008","2016-02-05 05:31:49.008",7.0),
      Row(1,"b5","c4","d6",2,2.8,"2021-01-02 02:03:09.009","2001-01-01 12:59:59.009",5.0)
    ))

    val df = sparkSession.createDataFrame(rdd,schema)
    df.show(false)
    val discrete = new Discrete(data = df, categoricalCols = Array("B","C","D"),logCols=Array("E","F"),labelCol = "A",dateDiffCols = Array("G"),logDateCols=Array("H"))
    val Transformer = discrete.transform
    val TransformedDF = Transformer._1
    val reflectMap = Transformer._2
    TransformedDF.show(false)
    println(reflectMap)
    val logReg = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")
    val model = logReg.fit(TransformedDF)
    val result = model.transform(TransformedDF)
    result.show(false)

  }

}
