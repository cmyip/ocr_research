package com.axios.lpr.engine

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max

fun Bitmap.toRgbImage(): RgbImage {
    val b = if (config == Bitmap.Config.ARGB_8888) this else copy(Bitmap.Config.ARGB_8888, false)
    val px = IntArray(b.width * b.height)
    b.getPixels(px, 0, b.width, 0, 0, b.width, b.height)
    return RgbImage.fromArgb(px, b.width, b.height)
}

fun RgbImage.toBitmap(): Bitmap = Bitmap.createBitmap(toArgb(), width, height, Bitmap.Config.ARGB_8888)

fun RgbImage.saveJpeg(f: File, quality: Int = 92) {
    f.parentFile?.mkdirs()
    FileOutputStream(f).use { toBitmap().compress(Bitmap.CompressFormat.JPEG, quality, it) }
}

fun RgbImage.savePng(f: File) {
    f.parentFile?.mkdirs()
    FileOutputStream(f).use { toBitmap().compress(Bitmap.CompressFormat.PNG, 100, it) }
}

object ImageLoader {
    const val MAX_SIDE = 4096

    /**
     * Decodes an image from a content URI (Photo Picker, SAF, share intent): applies EXIF
     * orientation, converts to ARGB_8888, and caps the long side at [maxSide].
     */
    fun decode(cr: ContentResolver, uri: Uri, maxSide: Int = MAX_SIDE): Bitmap {
        if (Build.VERSION.SDK_INT >= 28) {
            val src = ImageDecoder.createSource(cr, uri)
            return ImageDecoder.decodeBitmap(src) { dec, info, _ ->
                dec.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                dec.isMutableRequired = false
                val s = info.size
                val long = max(s.width, s.height)
                if (long > maxSide) {
                    val r = maxSide / long.toFloat()
                    dec.setTargetSize((s.width * r).toInt().coerceAtLeast(1), (s.height * r).toInt().coerceAtLeast(1))
                }
            }.let { if (it.config != Bitmap.Config.ARGB_8888) it.copy(Bitmap.Config.ARGB_8888, false) else it }
        }
        // API 26–27: BitmapFactory + manual EXIF rotation
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        cr.openInputStream(uri).use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / sample > maxSide) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample; inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val bmp = cr.openInputStream(uri).use { BitmapFactory.decodeStream(it, null, opts) } ?: error("cannot decode $uri")
        val orientation = cr.openInputStream(uri).use { ExifInterface(it!!).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }
        return applyExif(bmp, orientation)
    }

    fun applyExif(bmp: Bitmap, orientation: Int): Bitmap {
        val m = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
            else -> return bmp
        }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }

    fun displayName(cr: ContentResolver, uri: Uri): String? = runCatching {
        cr.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        }
    }.getOrNull()
}
