package com.app.opencvapplication

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.SeekBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.isVisible
import com.app.opencvapplication.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var blurValue: Int = 0
    private var originalBitmap: Bitmap? = null

    private val pickImage =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                result.data?.data?.let { uri ->
                    originalBitmap =
                        BitmapFactory.decodeStream(contentResolver.openInputStream(uri))
                    binding.originalImg.setImageBitmap(originalBitmap)
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        checkOpenCVInit()
        setupBlurSeekBar()
        initViews()
    }

    private fun checkOpenCVInit() {
        if (OpenCVLoader.initLocal()) {
            showToast(R.string.opencv_init_success)
            showProgressBar(false)
        } else {
            showToast(R.string.opencv_init_failed)
        }
    }

    private fun initViews() {
        binding.selectImageButton.setOnClickListener {
            openGallery()
        }

        binding.processButton.setOnClickListener {
            originalBitmap?.let { bitmap ->
                applyBlackAndWhiteFilter(bitmap)
            } ?: showToast(R.string.image_isnt_selected)
        }
    }

    private fun setupBlurSeekBar() {
        binding.blurSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                blurValue = progress
                binding.blurLabel.text = getString(R.string.blur_label, blurValue)
                originalBitmap?.let { bitmap ->
                    applyAndDisplayBlur(bitmap)
                } ?: showToast(R.string.image_isnt_selected)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}

            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK)
        intent.type = getString(R.string.image_type)
        pickImage.launch(intent)
    }

    private fun applyBlackAndWhiteFilter(originalBitmap: Bitmap) {
        showProgressBar(false)
        val originalMat = Mat(originalBitmap.height, originalBitmap.width, CvType.CV_8UC4)
        Utils.bitmapToMat(originalBitmap, originalMat)

        val grayMat = Mat(originalMat.rows(), originalMat.cols(), CvType.CV_8UC1)
        Imgproc.cvtColor(originalMat, grayMat, Imgproc.COLOR_BGR2GRAY)

        val grayBitmap = Bitmap.createBitmap(
            originalBitmap.width,
            originalBitmap.height,
            Bitmap.Config.ARGB_8888,
        )
        Utils.matToBitmap(grayMat, grayBitmap)

        binding.originalImg.setImageBitmap(originalBitmap)
        binding.bawImg.setImageBitmap(grayBitmap)
    }

    private fun applyBlur(originalBitmap: Bitmap, blurValue: Int): Bitmap {
        val originalMat = Mat(originalBitmap.height, originalBitmap.width, CvType.CV_8UC4)
        Utils.bitmapToMat(originalBitmap, originalMat)

        val kernelSize = if (blurValue % 2 == 0) blurValue + 1 else blurValue

        val blurredMat = Mat()
        val blurSize = Size(kernelSize.toDouble(), kernelSize.toDouble())
        Imgproc.GaussianBlur(originalMat, blurredMat, blurSize, 0.0)

        val blurredBitmap = Bitmap.createBitmap(
            originalBitmap.width,
            originalBitmap.height,
            Bitmap.Config.ARGB_8888,
        )
        Utils.matToBitmap(blurredMat, blurredBitmap)

        return blurredBitmap
    }

    private fun applyAndDisplayBlur(originalBitmap: Bitmap) {
        CoroutineScope(Dispatchers.Main).launch {
            showProgressBar(true)
            val blurredBitmap = withContext(Dispatchers.IO) {
                applyBlur(originalBitmap, blurValue)
            }
            binding.bawImg.setImageBitmap(blurredBitmap)
            showProgressBar(false)
        }
    }

    private fun showProgressBar(show: Boolean) {
        binding.progressBar.isVisible = show
    }

    private fun showToast(message: Int) {
        Toast.makeText(this, getString(message), Toast.LENGTH_SHORT).show()
    }
}
