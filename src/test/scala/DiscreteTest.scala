
import com.zjh.sf.function.Discrete
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types._
import org.scalatest.funsuite.AnyFunSuite


class DiscreteTest extends AnyFunSuite{
  test("Discrete"){
    val sparkSession = SparkSession.builder().master("local").appName("testDiscrete").getOrCreate()
    val schema = StructType(List(
      StructField("A",IntegerType,nullable = false),
      StructField("B",StringType,nullable = false),
      StructField("C",StringType,nullable = false),
      StructField("D",StringType,nullable = false),
      StructField("E",IntegerType,nullable = false),
      StructField("F",DoubleType,nullable = false)
    )
      )
    val rdd = sparkSession.sparkContext.parallelize(Seq(
      Row(1,"b1","c1","d1",3,2.3),
      Row(0,"b2","c2","d2",3,4.5),
      Row(1,"b3","c3","d3",2,8.7),
      Row(1,"b1","c2","d2",2,1.1),
      Row(0,"b2","c1","d1",1,5.4),
      Row(0,"b2","c5","d9",10,21.7)
    ))

    val df = sparkSession.createDataFrame(rdd,schema)
    val discrete = new Discrete(data = df, categoricalCols = Array("B","C","D"),logCols=Array("E","F"),labelCol = "A")
    val TransformedDF = discrete.transform
    TransformedDF.show()


  }

}
