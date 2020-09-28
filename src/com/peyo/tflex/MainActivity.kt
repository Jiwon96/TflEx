package com.peyo.tflex

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.View
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread

class MainActivity: Activity() {
    private lateinit var labelList: ArrayList<String>

    companion object {
        private const val TAG = "TFLEx01"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main)
    }

    fun onComputeClick(v: View) {
        thread {
            val tfliteModel = FileUtil.loadMappedFile(this, "mobilenet_v1_1_0_224_float.tflite")
            val tflite = Interpreter(tfliteModel, Interpreter.Options().addDelegate(NnApiDelegate()))

            convertBitmapToByteBuffer(getBitmap())
            var outputs = Array(1) { FloatArray(getNumLabels()) }

            tflite.run(imgData, outputs)

            outputs[0].sort()
            for (pr in outputs[0].takeLast(3).reversed()) {
                Log.d(TAG, "prob " + pr)
            }
            tflite.close()
        }
     }

    private var imgData: ByteBuffer? = null
    private val intValues = IntArray(224 * 224)
    private val IMAGE_MEAN = 128.0f
    private val IMAGE_STD = 128.0f

    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        if (imgData == null) {
            imgData = ByteBuffer.allocateDirect(
                    1 * 224 * 224 * 3 * 4)
            imgData!!.order(ByteOrder.nativeOrder())
        }
        imgData!!.rewind()
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        var pixel = 0
        for (i in 0 until 224) {
            for (j in 0 until 224) {
                val v: Int = intValues.get(pixel++)
                imgData!!.putFloat(((v shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                imgData!!.putFloat(((v shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                imgData!!.putFloat(((v and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
            }
        }
    }

    private fun getNumLabels(): Int {
        labelList = ArrayList<String>()
        val reader = BufferedReader(InputStreamReader(assets.open("labels.txt")))
        var line = reader.readLine()
        while(line != null) {
            labelList.add(line)
            line = reader.readLine()
        }
        return labelList.size
    }

    private fun getBitmap(): Bitmap {
        val stream = BitmapFactory.decodeStream(assets.open("test_image2.jpg"))
        return Bitmap.createScaledBitmap(stream, 224, 224, true)
    }

    fun onNNApiClick(v: View) {
    }
}
