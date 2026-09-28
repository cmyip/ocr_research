package com.axios.lpr.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(
    entities = [SessionEntity::class, CaptureEntity::class, VehicleEntity::class, PlateEntity::class, PlateReadEntity::class],
    version = 1,
    exportSchema = true,
)
abstract class LprDatabase : RoomDatabase() {
    abstract fun dao(): LprDao

    companion object {
        const val NAME = "axios_lpr.db"
        fun create(context: Context): LprDatabase =
            Room.databaseBuilder(context, LprDatabase::class.java, NAME).build()
    }
}
