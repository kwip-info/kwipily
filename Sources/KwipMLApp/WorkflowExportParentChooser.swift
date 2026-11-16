import AppKit

/// Package publication needs authority over the parent (staging + atomic rename),
/// even when the final product is a single checkpoint file. Single-file document
/// exports that write directly to the selected URL continue to use NSSavePanel.
@MainActor enum WorkflowExportParentChooser {
    static func panel(title: String, message: String, prompt: String = "Choose folder") -> NSOpenPanel {
        let panel = NSOpenPanel()
        panel.title = title
        panel.message = message
        panel.prompt = prompt
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        return panel
    }
}
