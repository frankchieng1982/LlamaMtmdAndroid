package com.example.llamamtmdapp

import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import java.io.File
import java.io.FileOutputStream

class MainActivity : ComponentActivity() {
    private lateinit var promptEditText: EditText
    private lateinit var selectImageButton: Button
    private lateinit var imagePreview: ImageView
    private lateinit var imagePathTextView: TextView
    private lateinit var runInferenceButton: Button
    private lateinit var outputTextView: TextView
    private var selectedImageUri: Uri? = null

    private val getImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        selectedImageUri = uri
        if (uri != null) {
            // Show preview
            contentResolver.openInputStream(uri)?.use { input ->
                val bitmap = BitmapFactory.decodeStream(input)
                imagePreview.setImageBitmap(bitmap)
            }
            imagePreview.visibility = ImageView.VISIBLE
            imagePathTextView.visibility = TextView.GONE
        } else {
            imagePreview.visibility = ImageView.GONE
            imagePathTextView.visibility = TextView.VISIBLE
            imagePathTextView.text = "No image selected"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (checkSelfPermission(android.Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(arrayOf(android.Manifest.permission.READ_MEDIA_IMAGES), 100)
            }
        }
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        promptEditText = findViewById(R.id.promptEditText)
//        promptEditText.setText("Describe this image in detail.")   // This appears as text
        selectImageButton = findViewById(R.id.selectImageButton)
        imagePreview = findViewById(R.id.imagePreview)
        imagePathTextView = findViewById(R.id.imagePathTextView)
        runInferenceButton = findViewById(R.id.runInferenceButton)
        outputTextView = findViewById(R.id.outputTextView)

        selectImageButton.setOnClickListener {
            getImageLauncher.launch("image/*")
        }

        runInferenceButton.setOnClickListener {
            val uri = selectedImageUri
            val prompt = promptEditText.text.toString().trim()
            when {
                uri == null -> Toast.makeText(this, "Select an image first", Toast.LENGTH_SHORT).show()
                prompt.isEmpty() -> Toast.makeText(this, "Enter a prompt", Toast.LENGTH_SHORT).show()
                else -> runMtmdInference(prompt, uri)
            }
        }

        copyAssetsToInternalStorage()
    }

    private fun copyAssetsToInternalStorage() {
        val internalDir = "${filesDir.absolutePath}/models/"
        val modelDir = File(internalDir)
        if (!modelDir.exists()) modelDir.mkdirs()
        listOf(
            "SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
            "mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
            "SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
            "mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
        ).forEach { assetFile ->
            val outFile = File(internalDir + assetFile)
            if (!outFile.exists()) {
                assets.open("models/$assetFile").use { input ->
                    FileOutputStream(outFile).use { output ->
                        input.copyTo(output)
                    }
                }
            }
        }
    }

    private fun runMtmdInference(prompt: String, imageUri: Uri) {
        val imageFile = File(filesDir, "temp_image.jpg")
        contentResolver.openInputStream(imageUri)?.use { input ->
            imageFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }

        outputTextView.text = ""  // Clear output before starting
        // Keep preview visible
        imagePreview.visibility = ImageView.VISIBLE

//        val modelPath = "${filesDir.absolutePath}/models/SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
        val modelPath = "${filesDir.absolutePath}/models/SmolVLM2-256M-Video-Instruct-Q8_0.gguf"
//        val mmprojPath = "${filesDir.absolutePath}/models/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
        val mmprojPath = "${filesDir.absolutePath}/models/mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf"
        val imagePath = imageFile.absolutePath

        Thread {
            try {
                val result = runInference(modelPath, mmprojPath, imagePath, prompt, 4096)
                if (result != 0) {
                    runOnUiThread {
                        Toast.makeText(this, "Inference failed", Toast.LENGTH_SHORT).show()
                    }
                }
            } finally {
                imageFile.delete() // Always delete temp
            }
        }.start()
    }

    // New method called from JNI for each token
    fun updateOutput(token: String) {
        runOnUiThread {
            outputTextView.append(token)
        }
    }

    private external fun runInference(
        modelPath: String,
        mmprojPath: String,
        imagePath: String,
        prompt: String,
        ctxSize: Int
    ): Int  // Removed StringBuilder param

    companion object {
        init {
            System.loadLibrary("native-lib")
        }
    }
}