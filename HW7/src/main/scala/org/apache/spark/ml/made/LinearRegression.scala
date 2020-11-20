package org.apache.spark.ml.made

import breeze.stats.distributions.Gaussian
import org.apache.log4j.Logger
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.types.StructType

import scala.util.control.Breaks.{break, breakable}

/**
 * Параметры для линейной регрессии
 */
trait LinearRegressionParams extends HasInputCol with HasOutputCol with HasLabelCol {
  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

/**
 * Эстиматор линейной регрессии
 *
 * @param uid     строка с уникальным идентификатором
 * @param epsilon точность для остановки обучения
 * @param maxIter максимальное количество итераций
 * @param lr      регулятор шага градиентного спуска. потом нормируется на длину вектора градиента.
 *
 */
class LinearRegression(override val uid: String,
                       val epsilon: Double,
                       val maxIter: Int,
                       val lr: Double) extends Estimator[LinearRegressionModel]
  with LinearRegressionParams with MLWritable {
  def this(epsilon: Double, maxIter: Int, lr: Double) = this(Identifiable.randomUID("linearRegression"), epsilon, maxIter, lr)

  /** обучение модели */
  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()
    val logger = Logger.getLogger("LR.fit")

    val normal01: Gaussian = breeze.stats.distributions.Gaussian(0, 1)
    val row = dataset.select($(inputCol)).head(1)(0)(0).asInstanceOf[DenseVector]
    val size: Int = row.size

    var w = Vectors.fromBreeze(breeze.linalg.DenseVector.rand(size, normal01)).toDense
    var b: Double = normal01.get()
    val vectors: DataFrame = dataset.toDF()
    breakable {
      for (i <- 0 to maxIter) {
        val model = new LinearRegressionModel(w, b).setInputCol($(inputCol)).setOutputCol("calc")
        val summary = vectors.rdd.mapPartitions((data: Iterator[Row]) => {
          val summarizer1 = new MultivariateOnlineSummarizer()
          val summarizer2 = new MultivariateOnlineSummarizer()
          data.foreach(v => {
            val vB: breeze.linalg.Vector[Double] = v.getAs[DenseVector](0).asBreeze
            val y: Double = v.getAs[Double](1)
            val calc = model.calcOne(vB)
            val eps = calc - y
            val grad = vB * eps
            summarizer1.add(mllib.linalg.Vectors.dense(eps))
            summarizer2.add(mllib.linalg.Vectors.fromBreeze(grad))
          })
          Iterator(Tuple2(summarizer1, summarizer2))
        }).reduce((t1, t2) => (t1._1 merge t2._1, t1._2 merge t2._2))
        val eps = summary._1.mean.asML(0)
        val rGradMean = summary._2.mean.asML

        val grad = rGradMean.asInstanceOf[Vector].toDense
        val gradBreeze = grad.asBreeze
        var gradBreeze2: Double = gradBreeze.t * gradBreeze
        if (math.abs(gradBreeze2) <= epsilon)
          break
        if (gradBreeze2 <= 1)
          gradBreeze2 = 1.0
        val l = lr / math.sqrt(gradBreeze2)
        w = Vectors.fromBreeze(w.asBreeze - gradBreeze * l).toDense
        b = b - l * eps
        if (i % 10 == 0) {
          logger.info(f"step=$i%3d eps=$eps%.10f a=$w b=$b%.5f")
        }
      }
    }
    logger.info(f"step=last a=$w b=$b%.5f")
    copyValues(new LinearRegressionModel(w, b)).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      val vectors: (Vector, Vector, Vector) = (Vectors.dense(epsilon), Vectors.dense(maxIter), Vectors.dense(lr))
      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/params")
    }
  }

}

/** Компаньон для эстиматора */
object LinearRegression extends MLReadable[LinearRegression] {
  override def read: MLReader[LinearRegression] = new MLReader[LinearRegression] {
    override def load(path: String): LinearRegression = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)
      val vectors = sqlContext.read.parquet(path + "/params")
      implicit val encoder: Encoder[Vector] = ExpressionEncoder()
      val (eps: Vector, maxIter: Vector, lr: Vector) = vectors.select(vectors("_1").as[Vector], vectors("_2").as[Vector], vectors("_3").as[Vector]).first()
      val estim = new LinearRegression(eps(0), maxIter(0).asInstanceOf[Int], lr(0))
      metadata.getAndSetParams(estim)
      estim
    }
  }
}

/**
 * Модель линейной регрессии.
 *
 * @param uid строка с уникальным идентификатором.
 * @param a   вектор коэффициентов при Х.
 * @param b   свободный член.
 */
class LinearRegressionModel private[made](override val uid: String,
                                          val a: DenseVector,
                                          val b: Double
                                         ) extends Model[LinearRegressionModel]
  with LinearRegressionParams with MLWritable {

  private[made] def this(a: Vector, b: Double) = this(Identifiable.randomUID("linearRegressionModel"), a.toDense, b)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(a, b))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val multUdf = dataset.sqlContext.udf.register(uid + "_transform", (x: Vector) => (x.asBreeze dot a.asBreeze) + b)
    dataset.withColumn($(outputCol), multUdf(dataset($(inputCol))))
  }

  def calcOne(v: breeze.linalg.Vector[Double]): Double = {
    (v dot a.asBreeze) + b
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      val vectors: (Vector, Vector) = a.asInstanceOf[Vector] -> Vectors.dense(b)
      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

/** компаньон для чтения модели */
object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val (a: Vector, b: Vector) = vectors.select(vectors("_1").as[Vector], vectors("_2").as[Vector]).first()
      val model = new LinearRegressionModel(a, b(0))
      metadata.getAndSetParams(model)
      model
    }
  }
}