import android.content.res.AssetFileDescriptor
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MyModel {
    private var interpreter: Interpreter

    init {
        // 모델 파일 로드
        val tfliteModel = loadModelFile()
        val options = Interpreter.Options()
        interpreter = Interpreter(tfliteModel, options)
    }

    // 모델 파일 로드
    private fun loadModelFile(): ByteBuffer {
        val assetFileDescriptor = assets.open("models/classification_model.tflite")
        val inputStream = assetFileDescriptor.createInputStream()
        val fileChannel = inputStream.channel
        val startOffset = fileChannel.position()
        val declaredLength = assetFileDescriptor.declaredLength
        val buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        return buffer.order(ByteOrder.nativeOrder())
    }

    // 이미지를 입력으로 넣어 모델 실행
    fun runModel(bitmap: Bitmap): FloatArray {
        val inputShape = interpreter.getInputTensor(0).shape()
        val inputSize = inputShape[1] * inputShape[2] * inputShape[3]
        val inputBuffer = ByteBuffer.allocateDirect(4 * inputSize)
        inputBuffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputShape[1] * inputShape[2])
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputShape[2], inputShape[1], true)
        scaledBitmap.getPixels(pixels, 0, scaledBitmap.width, 0, 0, scaledBitmap.width, scaledBitmap.height)
        for (pixel in pixels) {
            val r = (pixel and 0xff0000) shr 16
            val g = (pixel and 0xff00) shr 8
            val b = pixel and 0xff
            inputBuffer.putFloat(r / 255.0f)
            inputBuffer.putFloat(g / 255.0f)
            inputBuffer.putFloat(b / 255.0f)
        }

        // 모델 실행
        val outputShape = interpreter.getOutputTensor(0).shape()
        val outputSize = outputShape[1]
        val outputBuffer = ByteBuffer.allocateDirect(4 * outputSize)
        outputBuffer.order(ByteOrder.nativeOrder())
        interpreter.run(inputBuffer, outputBuffer)

        // 결과 처리
        val outputArray = FloatArray(outputSize)
        outputBuffer.rewind()
        outputBuffer.asFloatBuffer().get(outputArray)
        return outputArray
    }
}
