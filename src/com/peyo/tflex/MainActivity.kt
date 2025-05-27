package com.peyo.tflex

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
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
        private val images = arrayOf("1.jpg","2.jpg","3.jpg","4.jpg",
            "5.jpg","6.jpg","7.jpg","8.jpg",
            "9.jpg","10.jpg","11.jpg","12.jpg")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main)
        loadLabels()
        tfliteModel = FileUtil.loadMappedFile(this, "logistic.tflite")
    }

    override fun onDestroy() {
        super.onDestroy()
        tflite.close()
    }

    fun onComputeClick(v: View) {
        thread {
            val options = Interpreter.Options()
            if (nnapiToggle.isChecked) {
                options.addDelegate(NnApiDelegate())
            } else {
                options.setNumThreads(1)
            }
            tflite = Interpreter(tfliteModel, options)

            inferenceTime = 0
            firstFrame = true
            for(image in images) {
                convertBitmapToByteBuffer(getBitmap(image))

                startTime = SystemClock.uptimeMillis()
                tflite.run(imgData, outputs)
                printLabels(image)

                Thread.sleep(500)
            }

            runOnUiThread {
                textView1.text = "Summary: \n\t Average Inference time (ms): " +
                        "${inferenceTime / (images.size - 1)}"
                textView2.text = ""
            }
            tflite.close()
        }
     }

    private fun printLabels(str: String) {
        val runtime = SystemClock.uptimeMillis() - startTime
        var text = ""
        text = if (outputs[0][0] > 0.5f) "강아지" else "고양이"
        text = str + ": Result:" + text

        runOnUiThread {
            textView1.text = "Inference time (ms): " + runtime
            if (firstFrame) {
                firstFrame = false
            } else {
                inferenceTime += runtime
            }

            textView2.text = text
        }
    }

    private var startTime: Long = 0
    private var inferenceTime : Long = 0
    private var firstFrame : Boolean = true

    private lateinit var tfliteModel: MappedByteBuffer
    private var imgData: ByteBuffer? = null
    private val intValues = IntArray(256 * 256) // 그림 256 * 256
    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        if (imgData == null) {
            imgData = ByteBuffer.allocateDirect(
                    1 * 256 * 256 * 3 * 4) // int = byte * 4
            imgData!!.order(ByteOrder.nativeOrder())
        }
        imgData!!.rewind()
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        var pixel = 0
        for (i in 0 until 256) {            // 256 * 256 * 3 => [[[1,2,3], [1,2,3], [1,2,3]] ... [[1,2,3], [1,2,3], [1,2,3]], ... [[1,2,3], [1,2,3], [1,2,3]]]
            for (j in 0 until 256) {        // https://forums.oracle.com/ords/apexds/post/what-does-0xff-and-0x0f-mean-6851
                val v: Int = intValues.get(pixel++)
//                Log.i( "test", v.toString())
//                imgData!!.putFloat(v / 255f);
                imgData!!.putFloat(((v shr 16 and 0xFF) /255f)) // 16bit~23 R
                imgData!!.putFloat(((v shr 8 and 0xFF) /255f))  // 8 ~ 15 bit G
                imgData!!.putFloat(((v and 0xFF) /255f))        // 0 ~ 7bit B
            }
        }
    }

    private lateinit var tflite: Interpreter
    private lateinit var outputs: Array<FloatArray>

    private fun loadLabels() {
        outputs = Array(1) { FloatArray(1) }
    }

    private fun getBitmap(imageName: String): Bitmap {
        val stream = BitmapFactory.decodeStream(assets.open(imageName))
        runOnUiThread {
            imageView.setImageBitmap(Bitmap.createScaledBitmap(stream, 480, 480, true))
        }
        return Bitmap.createScaledBitmap(stream, 256, 256, true)
    }
}
