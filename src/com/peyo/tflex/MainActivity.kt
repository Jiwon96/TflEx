package com.peyo.tflex

import android.app.Activity
import android.os.Bundle
import android.view.View

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
    }

    fun onNNApiClick(v: View) {
    }

    private fun enableCaptureButton(delay:Long) {
        thread {
            Thread.sleep(delay)
            this@MainActivity.runOnUiThread {
            }
        }
    }

}
