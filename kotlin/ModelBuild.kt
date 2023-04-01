package com.example.cycleye
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import com.google.firebase.FirebaseApp
import com.google.firebase.ml.custom.*


fun main() {
    val localModel = FirebaseCustomLocalModel.Builder()
        .setAssetFilePath("models/classification_model.tflite")
        .build()

        val options = FirebaseModelInterpreterOptions.Builder(localModel).build()
        val interpreter = FirebaseModelInterpreter.getInstance(options)

        val inputOutputOptions = FirebaseModelInputOutputOptions.Builder()
            .setInputFormat(0, FirebaseModelDataType.FLOAT32, intArrayOf(1, 224, 224, 3))
            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, intArrayOf(1, 3))
            .build()
        var imageString = "images/21_X395_C015_0131_3.jpg"
        var myInputImage = BitmapFactory.decodeFile(imageString)
        val bitmap = Bitmap.createScaledBitmap(myInputImage, 224, 224, true)

        val batchNum = 0
        val input = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }
        for (x in 0..223) {
            for (y in 0..223) {
                val pixel = bitmap.getPixel(x, y)
                // Normalize channel values to [-1.0, 1.0]. This requirement varies by
                // model. For example, some models might require values to be normalized
                // to the range [0.0, 1.0] instead.
                input[batchNum][x][y][0] = (Color.red(pixel) - 127) / 255.0f
                input[batchNum][x][y][1] = (Color.green(pixel) - 127) / 255.0f
                input[batchNum][x][y][2] = (Color.blue(pixel) - 127) / 255.0f
            }
        }
        val inputs = FirebaseModelInputs.Builder()
            .add(input) // add() as many input arrays as your model requires
            .build()

        interpreter?.run(inputs, inputOutputOptions)
            ?.addOnSuccessListener { result ->
                // ...
                val output = result.getOutput<Array<FloatArray>>(0)
                val probabilities = output[0]
//                val reader = BufferedReader(
//                    InputStreamReader(assets.open("retrained_labels.txt"))
//                )
//                for (i in probabilities.indices) {
//                    val label = reader.readLine()
//                    Log.i("MLKit", String.format("%s: %1.4f", label, probabilities[i]))
//                }
                print(probabilities)
            }
            ?.addOnFailureListener { e ->
                // Task failed with an exception
                // ...
            }
}
