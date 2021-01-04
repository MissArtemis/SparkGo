package com.zjh.sf.function

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import scala.collection.immutable.HashMap

class Discrete(
                var data:DataFrame = null,
                var labelCol:String = null,
                var logCols:Array[String] = null,
                var categoricalCols:Array[String] = null,
                var dateDiffCols:Array[String] = null,
                var logDateCols:Array[String] = null ) {

  def categoricalDiscrete(data: DataFrame, categoricalCols:Array[String]):DataFrame = {
    import data.sparkSession.implicits._
    var datatmp = data
    val outputCols = categoricalCols.map(c=>c+"_discreted")
    val indexer = new StringIndexer().setInputCols(categoricalCols).setOutputCols(outputCols).fit(datatmp)
    datatmp = indexer.transform(datatmp)
    datatmp = datatmp.drop(categoricalCols:_*)
    datatmp
  }

  def logDiscrete(data:DataFrame,logCols:Array[String]):DataFrame = {
    import data.sparkSession.implicits._
    var datatmp = data
    for(c <- logCols){
      datatmp = datatmp.withColumn(c+"_discreted",log(datatmp(c)).cast(IntegerType))
    }
    datatmp = datatmp.drop(logCols:_*)
    datatmp
  }

  def transform():DataFrame={
    var discrete = categoricalDiscrete(data,categoricalCols)
    discrete = logDiscrete(discrete,logCols)
    discrete

  }

}
