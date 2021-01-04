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

  def categoricalDiscrete(data: DataFrame, categoricalCols:Array[String]):(DataFrame,Map[String,DataFrame]) = {
    import data.sparkSession.implicits._
    var datatmp = data
    var discreteMap = new HashMap[String,DataFrame]
    val outputCols = categoricalCols.map(c=>c+"_discreted")
    val indexer = new StringIndexer().setInputCols(categoricalCols).setOutputCols(outputCols).fit(datatmp)
    datatmp = indexer.transform(datatmp)
    for(c <- categoricalCols){
      val columnMap = datatmp.select(c,c+"_discreted").groupBy(c).agg(max(col(c+"_discreted")).alias(c+"_discreted"))
      discreteMap += (c -> columnMap)
    }
    datatmp = datatmp.drop(categoricalCols:_*)
    (datatmp,discreteMap)
  }

  def logDiscrete(data:DataFrame,logCols:Array[String]):(DataFrame,Map[String,DataFrame]) = {
    import data.sparkSession.implicits._
    var datatmp = data
    var discreteMap = new HashMap[String,DataFrame]
    for(c <- logCols){
      datatmp = datatmp.withColumn(c+"_discreted",log(datatmp(c)).cast(IntegerType))
      val columnMap = datatmp.select(c,c+"_discreted").groupBy(c).agg(max(col(c+"_discreted")).alias(c+"_discreted"))
      discreteMap += (c -> columnMap)
    }
    datatmp = datatmp.drop(logCols:_*)
    (datatmp,discreteMap)
  }

  def dateDiffDiscrete(data:DataFrame,dateDiffCols:Array[String]):(DataFrame,Map[String,DataFrame]) = {
    import data.sparkSession.implicits._
    var datatmp = data
    var discreteMap = new HashMap[String,DataFrame]
    for(c <- dateDiffCols){
      datatmp = datatmp.withColumn(c,col(c).cast(TimestampType))
      datatmp = datatmp.withColumn(c+"_discreted",datediff(current_timestamp(),col(c)))
      val columnMap = datatmp.select(c,c+"_discreted").groupBy(c).agg(max(col(c+"_discreted")).alias(c+"_discreted"))
      discreteMap += (c -> columnMap)
    }
    datatmp = datatmp.drop(dateDiffCols:_*)
    (datatmp,discreteMap)
  }

  def logDateDiscrete(data:DataFrame,logDateCols:Array[String]):(DataFrame,Map[String,DataFrame])={
    import data.sparkSession.implicits._
    var datatmp  =data
    var discreteMap = new HashMap[String,DataFrame]
    for(c <- logDateCols){
      datatmp = datatmp.withColumn(c,col(c).cast(TimestampType))
      datatmp = datatmp.withColumn(c+"_discreted",log(datediff(current_timestamp(),col(c))).cast(IntegerType))
      val columnMap = datatmp.select(c,c+"_discreted").groupBy(c).agg(max(col(c+"_discreted")).alias(c+"_discreted"))
      discreteMap += (c -> columnMap)
    }
    datatmp = datatmp.drop(logDateCols:_*)
    (datatmp,discreteMap)
  }

  def labelDiscrete(data:DataFrame,labelCol:String):(DataFrame,Map[String,DataFrame])={
    var datatmp = data
    val outputCol = "label"
    var discreteMap = new HashMap[String,DataFrame]
    val indexer = new StringIndexer().setInputCol(labelCol).setOutputCol(outputCol).fit(datatmp)
    datatmp = indexer.transform(datatmp)
    val columnMap = datatmp.select(labelCol,outputCol).groupBy(labelCol).agg(max(outputCol).alias(outputCol))
    discreteMap += (labelCol -> columnMap)
    datatmp = datatmp.drop(labelCol)
    (datatmp,discreteMap)
  }

  def transform():(DataFrame,Map[String,DataFrame])={
    var reflectMap = new HashMap[String,DataFrame]
    var transformer = categoricalDiscrete(data,categoricalCols)
    var discrete = transformer._1
    var columnMap = transformer._2
    reflectMap = reflectMap.++(columnMap)
    transformer = logDiscrete(discrete,logCols)
    discrete = transformer._1
    columnMap = transformer._2
    reflectMap = reflectMap ++ columnMap
    transformer = dateDiffDiscrete(discrete,dateDiffCols)
    discrete = transformer._1
    columnMap = transformer._2
    reflectMap = reflectMap ++ columnMap
    transformer = logDateDiscrete(discrete,logDateCols)
    discrete = transformer._1
    columnMap = transformer._2
    reflectMap = reflectMap ++ columnMap
    transformer = labelDiscrete(discrete,labelCol)
    discrete = transformer._1
    columnMap = transformer._2
    reflectMap = reflectMap ++ columnMap
    val featureCols = discrete.drop("label").columns
    val vectorAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    discrete = vectorAssembler.transform(discrete)
    discrete = discrete.drop(featureCols:_*)
    (discrete,reflectMap)

  }

}
