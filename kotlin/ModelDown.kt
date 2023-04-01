package com.example.cycleye

import com.google.firebase.ml.custom.FirebaseCustomRemoteModel
import com.google.firebase.ml.modeldownloader.CustomModel
import com.google.firebase.ml.modeldownloader.CustomModelDownloadConditions
import com.google.firebase.ml.modeldownloader.DownloadType
import com.google.firebase.ml.modeldownloader.FirebaseModelDownloader
import org.tensorflow.lite.Interpreter

fun main(){
    val conditions = CustomModelDownloadConditions.Builder()
        .requireWifi()  // Also possible: .requireCharging() and .requireDeviceIdle()
        .build()

    FirebaseModelDownloader.getInstance()
        .getModel("classification_model", DownloadType.LOCAL_MODEL_UPDATE_IN_BACKGROUND,
            conditions)
        .addOnSuccessListener { model: CustomModel? ->
            print("성공?")
            // Download complete. Depending on your app, you could enable the ML
            // feature, or switch from the local model to the remote model, etc.

            // The CustomModel object contains the local path of the model file,
            // which you can use to instantiate a TensorFlow Lite interpreter.
            val modelFile = model?.file
            if (modelFile != null) {
                var interpreter = Interpreter(modelFile)
            }
        }
}
