package com.zjh.sf.function

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.catalyst.util.DateTimeUtils.currentTimestamp
import scala.collection.immutable.HashMap
import org.apache.spark.ml.feature.VectorAssembler

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

  def dateDiffDiscrete(data:DataFrame,dateDiffCols:Array[String]):DataFrame = {
    import data.sparkSession.implicits._
    var datatmp = data
    for(c <- dateDiffCols){
      datatmp = datatmp.withColumn(c,col(c).cast(TimestampType))
      datatmp = datatmp.withColumn(c+"_discreted",datediff(current_timestamp(),col(c)))
    }
    datatmp = datatmp.drop(dateDiffCols:_*)
    datatmp
  }

  def logDateDiscrete(data:DataFrame,logDateCols:Array[String]):DataFrame={
    import data.sparkSession.implicits._
    var datatmp  =data
    for(c <- logDateCols){
      datatmp = datatmp.withColumn(c,col(c).cast(TimestampType))
      datatmp = datatmp.withColumn(c+"_discreted",log(datediff(current_timestamp(),col(c))).cast(IntegerType))
    }
    datatmp = datatmp.drop(logDateCols:_*)
    datatmp
  }

  def labelDiscrete(data:DataFrame,labelCol:String):DataFrame={
    var datatmp = data
    val outputCol = "label"
    val indexer = new StringIndexer().setInputCol(labelCol).setOutputCol(outputCol).fit(datatmp)
    datatmp = indexer.transform(datatmp)
    datatmp = datatmp.drop(labelCol)
    datatmp

  }

  def transform():DataFrame={
    var discrete = categoricalDiscrete(data,categoricalCols)
    discrete = logDiscrete(discrete,logCols)
    discrete = dateDiffDiscrete(discrete,dateDiffCols)
    discrete = logDateDiscrete(discrete,logDateCols)
    discrete = labelDiscrete(discrete,labelCol)
    val featureCols = discrete.drop("label").columns
    val vectorAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    discrete = vectorAssembler.transform(discrete)
    discrete = discrete.drop(featureCols:_*)
    discrete

  }

}
