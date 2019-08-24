import org.nd4j.linalg.factory.Nd4j

import org.nd4s.Implicits._

object Nd4sExample {

  def main(args: Array[String]): Unit = {

    val a1 = Nd4j.createFromArray(1, 2, 3)
    val a2 = Nd4j.createFromArray(10, 20, 30)

    val result = a1 + a2

    print(result)
  }
}
