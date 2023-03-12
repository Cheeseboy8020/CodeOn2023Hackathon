package com.aneeshjosyula.codeon

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.speech.RecognitionListener
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import org.tensorflow.lite.task.gms.vision.TfLiteVision

private const val TAG = "CodeonMain"
class MainActivity : AppCompatActivity() {
    lateinit var bButton: Button
    lateinit var ivImage: ImageView
    lateinit var tvText: TextView
    lateinit var helper: ImageClassifierHelper
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        bButton = findViewById(R.id.bButton)
        ivImage = findViewById(R.id.ivImage)
        tvText = findViewById(R.id.tvText)
        helper = ImageClassifierHelper(context = this)
        var data: Intent?
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        val takePicture = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                // There are no request codes
                data = result.data
                Log.d(TAG, "Took picture")
                val classifications = helper.classify(data?.extras?.get("data") as Bitmap, 1)
                Log.d(TAG, classifications!!.size.toString())
                ivImage.setImageBitmap(data?.extras?.get("data") as Bitmap)
                Log.d(TAG, classifications[0].categories[0].score.toString())
                Log.d(TAG, classifications[0].categories.size.toString())
                if (classifications[0].categories[0].score > 0.5) {
                    tvText.text = "No Glaucoma Detected"
                    tvText.setTextColor(getColor(R.color.green))
                }
                else{
                    tvText.text = "Glaucoma Detected"
                    tvText.setTextColor(getColor(R.color.red))
                }
            }
        }
        bButton.setOnClickListener {
            Log.d(TAG, "Button Clicked")
            try {
                takePicture.launch(takePictureIntent)
            } catch (e: ActivityNotFoundException) {
                // display error state to the user
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        helper.clearImageClassifier()
    }
}