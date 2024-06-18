package com.peyo.tflex

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.SystemClock
import android.view.View
import kotlinx.android.synthetic.main.main.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.util.*
import java.util.AbstractMap.SimpleEntry
import kotlin.collections.ArrayList
import kotlin.concurrent.thread

class MainActivity: Activity() {
    companion object {
        private const val TAG = "TFLEx01"
        private val images = arrayOf("test_image.jpg", "test_image1.jpg",
                "test_image2.jpg", "test_image3.jpg", "test_image4.jpg",
                "test_image5.jpg", "test_image.jpg", "test_image1.jpg",
                "test_image2.jpg", "test_image3.jpg", "test_image4.jpg",
                "test_image5.jpg", "test_image.jpg", "test_image1.jpg",
                "test_image2.jpg", "test_image3.jpg", "test_image4.jpg",
                "test_image5.jpg", "test_image.jpg", "test_image1.jpg",
                "test_image2.jpg", "test_image3.jpg", "test_image4.jpg",
                "test_image5.jpg")

    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main)
        loadLabels()
        tfliteModel1 = FileUtil.loadMappedFile(this, "mobilenet_v1_1_0_224_float.tflite")
        tfliteModel2 = FileUtil.loadMappedFile(this, "mobilenet_v1_1_0_224_quantized.tflite")
    }

    override fun onPause() {
        super.onPause()
        stop = true
    }

    override fun onDestroy() {
        super.onDestroy()
        tflite1.close()
    }

    fun onComputeClick(v: View) {
        if (running1 || running2) {
            stop = true
        } else {
            stop = false
            if (!running1) {
                thread {
                    running1 = true
                    val options = Interpreter.Options()
                    options.setNumThreads(4)
                    tflite1 = Interpreter(tfliteModel1, options)

                    inferenceTime1 = 0
                    firstFrame1 = true

                    while (!stop) {
                        for (image in images) {
                            convertBitmapToByteBuffer1(getBitmap1(image))

                            startTime1 = SystemClock.uptimeMillis()
                            tflite1.run(imgData1, outputs1)
                            printLabels1()

                            if (stop) {
                                break
                            }
                        }
                    }
                    tflite1.close()
                    running1 = false
                }
            }
            if (!running2) {
                thread {
                    running2 = true
                    val options = Interpreter.Options()
                    options.addDelegate(NnApiDelegate())
                    tflite2 = Interpreter(tfliteModel2, options)

                    inferenceTime2 = 0
                    firstFrame2 = true

                    while (!stop) {
                        for (image in images) {
                            convertBitmapToByteBuffer2(getBitmap2(image))

                            startTime2 = SystemClock.uptimeMillis()
                            tflite2.run(imgData2, outputs2)
                            printLabels2()

                            if (stop) {
                                break
                            }
                        }
                    }
                    tflite2.close()
                    running2 = false
                }
            }

        }
     }

    private fun printLabels1() {
        val runtime = SystemClock.uptimeMillis() - startTime1

        for (i in 0 until getNumLabels()) {
            sortedLabels1.add(SimpleEntry(labelList[i], outputs1[0][i]))
            if (sortedLabels1.size > RESULTS_TO_SHOW) {
                sortedLabels1.poll()
            }
        }

        var text = ""
        for (i in 0 until sortedLabels1.size) {
            val label = sortedLabels1.poll()
            text = String.format("\n  %s: %f", label.key, label.value) + text
        }
        text = "Result:" + text

        runOnUiThread {
            time1.text = "Inference time (ms): " + runtime
            if (firstFrame1) {
                firstFrame1 = false
            } else {
                inferenceTime1 += runtime
            }

            result1.text = text
        }
    }

    private fun printLabels2() {
        if ((count2 % 8) != 0L) return
        val runtime = SystemClock.uptimeMillis() - startTime2

        for (i in 0 until getNumLabels()) {
            sortedLabels2.add(SimpleEntry(labelList[i], outputs2[0][i]))
            if (sortedLabels2.size > RESULTS_TO_SHOW) {
                sortedLabels2.poll()
            }
        }

        var text = ""
        for (i in 0 until sortedLabels2.size) {
            val label = sortedLabels2.poll()
            text = String.format("\n  %s: %d", label.key, label.value) + text
        }
        text = "Result:" + text

        runOnUiThread {
            time2.text = "Inference time (ms): " + runtime
            if (firstFrame2) {
                firstFrame2 = false
            } else {
                inferenceTime2 += runtime
            }

            result2.text = text
        }
    }

    private var running1 : Boolean = false
    private var running2 : Boolean = false
    private var stop : Boolean = false
    private var startTime1: Long = 0
    private var inferenceTime1 : Long = 0
    private var firstFrame1 : Boolean = true
    private var count2: Long = 0
    private var startTime2: Long = 0
    private var inferenceTime2 : Long = 0
    private var firstFrame2 : Boolean = true

    private lateinit var tfliteModel1: MappedByteBuffer
    private lateinit var tfliteModel2: MappedByteBuffer
    private var imgData1: ByteBuffer? = null
    private var imgData2: ByteBuffer? = null
    private val intValues = IntArray(224 * 224)
    private val IMAGE_MEAN1 = 128.0f
    private val IMAGE_STD1 = 128.0f

    private fun convertBitmapToByteBuffer1(bitmap: Bitmap) {
        if (imgData1 == null) {
            imgData1 = ByteBuffer.allocateDirect(
                    1 * 224 * 224 * 3 * 4)
            imgData1!!.order(ByteOrder.nativeOrder())
        }
        imgData1!!.rewind()
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        var pixel = 0
        for (i in 0 until 224) {
            for (j in 0 until 224) {
                val v: Int = intValues.get(pixel++)
                imgData1!!.putFloat(((v shr 16 and 0xFF) - IMAGE_MEAN1) / IMAGE_STD1)
                imgData1!!.putFloat(((v shr 8 and 0xFF) - IMAGE_MEAN1) / IMAGE_STD1)
                imgData1!!.putFloat(((v and 0xFF) - IMAGE_MEAN1) / IMAGE_STD1)
            }
        }
    }

    private fun convertBitmapToByteBuffer2(bitmap: Bitmap) {
        if (imgData2 == null) {
            imgData2 = ByteBuffer.allocateDirect(
                1 * 224 * 224 * 3)
            imgData2!!.order(ByteOrder.nativeOrder())
        }
        imgData2!!.rewind()
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        var pixel = 0
        for (i in 0 until 224) {
            for (j in 0 until 224) {
                val v: Int = intValues.get(pixel++)
                imgData2!!.put((v shr 16 and 0xFF).toByte())
                imgData2!!.put((v shr 8 and 0xFF).toByte())
                imgData2!!.put((v and 0xFF).toByte())
            }
        }
    }

    private val RESULTS_TO_SHOW = 3

    private val sortedLabels1 = PriorityQueue<Map.Entry<String, Float>>(RESULTS_TO_SHOW)
        { o1, o2 -> o1.value.compareTo(o2.value) }
    private val sortedLabels2 = PriorityQueue<Map.Entry<String, Byte>>(RESULTS_TO_SHOW)
    { o1, o2 -> o1.value.compareTo(o2.value) }

    private lateinit var tflite1: Interpreter
    private lateinit var tflite2: Interpreter
    private var labelList = ArrayList<String>()
    private lateinit var outputs1: Array<FloatArray>
    private lateinit var outputs2: Array<ByteArray>

    private fun loadLabels() {
        val reader = BufferedReader(InputStreamReader(assets.open("labels.txt")))
        var line = reader.readLine()
        while(line != null) {
            labelList.add(line)
            line = reader.readLine()
        }
        outputs1 = Array(1) { FloatArray(labelList.size) }
        outputs2 = Array(1) { ByteArray(labelList.size) }
    }

    private fun getNumLabels(): Int {
        return labelList.size
    }

    private fun getBitmap1(imageName: String): Bitmap {
        val stream = BitmapFactory.decodeStream(assets.open(imageName))
        runOnUiThread {
            imageView1.setImageBitmap(Bitmap.createScaledBitmap(stream, 480, 480, true))
        }
        return Bitmap.createScaledBitmap(stream, 224, 224, true)
    }

    private fun getBitmap2(imageName: String): Bitmap {
        val stream = BitmapFactory.decodeStream(assets.open(imageName))
        if ((count2 % 8) == 0L) {
        runOnUiThread {
            imageView2.setImageBitmap(Bitmap.createScaledBitmap(stream, 480, 480, true))
        }}
        return Bitmap.createScaledBitmap(stream, 224, 224, true)
    }
}
