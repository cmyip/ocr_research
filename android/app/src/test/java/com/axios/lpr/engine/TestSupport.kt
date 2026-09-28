package com.axios.lpr.engine

import java.io.File

/** Reads SDK records straight from ../weights for JVM tests. */
class FileModelFiles(private val dir: File = locate("weights")) : ModelFiles {
    override val catalog: ModelCatalog = ModelCatalog.parse(locate("android/app/src/main/assets/models.json").readText())
    override fun bytes(id: Int) = File(dir, catalog[id].file).readBytes()
    override fun lines(id: Int) = AndroidModelStore.decodeLines(bytes(id))

    companion object {
        /** Walks up from the Gradle test working dir (android/app) to the repo root. */
        fun locate(rel: String): File {
            var d: File? = File("").absoluteFile
            while (d != null) {
                val f = File(d, rel)
                if (f.exists()) return f
                d = d.parentFile
            }
            error("cannot find $rel")
        }
    }
}

/** Decodes a sample JPEG with the JDK's ImageIO (via reflection: unit tests compile against android.jar). */
fun loadSample(name: String): RgbImage {
    val file = FileModelFiles.locate("sample_data/$name.jpeg")
    val imageIO = Class.forName("javax.imageio.ImageIO")
    val img = imageIO.getMethod("read", File::class.java).invoke(null, file)
    val cls = img.javaClass
    val w = cls.getMethod("getWidth").invoke(img) as Int
    val h = cls.getMethod("getHeight").invoke(img) as Int
    val getRgb = cls.getMethod("getRGB", Int::class.java, Int::class.java, Int::class.java, Int::class.java, IntArray::class.java, Int::class.java, Int::class.java)
    val px = getRgb.invoke(img, 0, 0, w, h, null, 0, w) as IntArray
    return RgbImage.fromArgb(px, w, h)
}
