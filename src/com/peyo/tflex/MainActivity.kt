package com.peyo.tflex

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.View
import kotlinx.android.synthetic.main.main.*
import androidx.annotation.NonNull
import com.peyo.tflex.ml.Mobilenet
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.model.Model


import kotlin.concurrent.thread

class MainActivity: Activity() {
    companion object {
        private const val TAG = "TFLEx01"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main)


    }

    fun onComputeClick(v: View) {
        thread {
            val options = Model.Options.Builder().setDevice(Model.Device.GPU).build()
            val model = Mobilenet.newInstance(this, options)

            val tfImage = TensorImage.fromBitmap(getBitmap())

            val outputs = model.process(tfImage)
                    .probabilityAsCategoryList.apply {
                        sortByDescending { it.score }
                    }.take(3)
            for (output in outputs) {
                Log.d(TAG, output.label)
            }
        }
     }

    private fun getBitmap(): Bitmap {
        val stream = BitmapFactory.decodeStream(assets.open("test_image1.jpg"))
        return Bitmap.createScaledBitmap(stream,224, 224, true)
    }

    fun onNNApiClick(v: View) {
    }

    private fun enableCaptureButton(delay: Long) {
        thread {
            Thread.sleep(delay)
            this@MainActivity.runOnUiThread {
            }
        }
    }

}
