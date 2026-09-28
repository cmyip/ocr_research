package com.axios.lpr.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Folder
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.vector.ImageVector
import com.axios.lpr.AppContainer

/**
 * "Import" entry point: system Photo Picker (no storage permission) or the Storage Access
 * Framework for files (Downloads, Drive, SD, USB). Multiple selections become a batch.
 */
@Composable
fun ImportMenu(c: AppContainer, compact: Boolean = false, icon: ImageVector = Icons.Outlined.PhotoLibrary) {
    var open by remember { mutableStateOf(false) }
    val photos = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia(50)) { uris ->
        c.importer.import(uris, "photo library")
    }
    val files = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        c.importer.import(uris, "files")
    }
    Box {
        if (compact) FilledTonalIconButton(onClick = { open = true }) { Icon(icon, "Import images") }
        else FilledTonalButton(onClick = { open = true }) { Icon(icon, null); Text("  Import images") }
        DropdownMenu(open, onDismissRequest = { open = false }) {
            DropdownMenuItem(text = { Text("Photo library") }, leadingIcon = { Icon(Icons.Outlined.Image, null) }, onClick = {
                open = false
                photos.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            })
            DropdownMenuItem(text = { Text("Files…") }, leadingIcon = { Icon(Icons.Outlined.Folder, null) }, onClick = {
                open = false
                files.launch(arrayOf("image/*"))
            })
        }
    }
}
