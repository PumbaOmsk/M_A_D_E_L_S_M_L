package org.apache.spark.ml.made

import breeze.linalg.*
import breeze.stats.distributions.Gaussian
import org.apache.log4j.Logger
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.avg
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, SparkSession}
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
    //    SchemaUtils.checkColumnType(schema, getLabelCol, DoubleType)

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
 * @param uid строка с уникальным идентификатором
 */
class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel]
  with LinearRegressionParams with DefaultParamsWritable {
  /** Точность для остановки обучения */
  val epsilon = 1e-9
  /** Максимальное количество итераций */
  val MAX_ITER = 10000
  /** регулятор шага градиентного спуска. потом нормируется на длину вектора градиента. */
  val lambda = 0.5

  def this() = this(Identifiable.randomUID("linearRegression"))

  /** обучение модели */
  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    val logger = Logger.getLogger("LR.fit")

    val normal01: Gaussian = breeze.stats.distributions.Gaussian(0, 1)
    val row = dataset.select($(inputCol)).head(1)(0)(0).asInstanceOf[DenseVector]
    val size: Int = row.size

    var w = Vectors.fromBreeze(breeze.linalg.DenseVector.rand(size, normal01)).toDense
    var b: Double = normal01.get()
    import dataset.sqlContext.implicits._
    breakable {
      for (i <- 0 to MAX_ITER) {
        val model = new LinearRegressionModel(w, b).setInputCol($(inputCol)).setOutputCol("calc")
        val y = model.transform(dataset)
        val epsDf = y.withColumn("eps", $"calc" - $"y").cache()

        val multUdf = epsDf.sqlContext.udf.register(uid + "_grad", (x: Vector, eps: Double) => Vectors.fromBreeze(x.asBreeze * eps))
        val gradDf = epsDf.withColumn("grad", multUdf(epsDf("features"), epsDf("eps"))).cache()

        implicit val encoder: Encoder[Vector] = ExpressionEncoder()
        val vectors: Dataset[Vector] = gradDf.select(gradDf("grad").as[Vector])
        val summary = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
          val summarizer = new MultivariateOnlineSummarizer()
          data.foreach(v => summarizer.add(mllib.linalg.Vectors.fromBreeze(v.asBreeze)))
          Iterator(summarizer)
        }).reduce(_ merge _)
        val rGradMean: Vector = summary.mean.asML
        //        val Row(Row(rGradMean)) = gradDf
        //          .select(Summarizer.metrics("mean").summary(gradDf("grad")))
        //          .first()

        val eps: Double = gradDf.agg(avg($"eps")).first()(0).asInstanceOf[Double]
        if (math.abs(eps) <= epsilon)
          break
        val grad = rGradMean.asInstanceOf[Vector].toDense
        val gradBreeze = grad.asBreeze
        var gradBreeze2 = gradBreeze.t * gradBreeze
        if (gradBreeze2 <= 1)
          gradBreeze2 = 1.0
        val l = lambda / math.sqrt(gradBreeze2)
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
}

/** Компаньон для эстиматора */
object LinearRegression extends DefaultParamsReadable[LinearRegression] {
  /** создание данных для тестирования модели */
  def createTestData(spark: SparkSession, n: Int, a: breeze.linalg.DenseVector[Double], b: Double): DataFrame = {
    import spark.implicits._

    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    val X = breeze.linalg.DenseMatrix.rand(n, a.length, normal01)

    val y = X * a + b
    val data = breeze.linalg.DenseMatrix.horzcat(X, y.asDenseMatrix.t)
    val df: DataFrame = data(*, ::).iterator.map(x => Tuple2(Vectors.dense(x(0), x(1), x(2)), x(3))).toSeq.toDF("features", "y")
    df
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

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      val vectors = a.asInstanceOf[Vector] -> Vectors.dense(b)
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