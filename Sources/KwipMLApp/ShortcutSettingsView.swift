import SwiftUI
import KwipMLCore

extension ShortcutChord {
    var keyboardShortcut: KeyboardShortcut {
        var modifiers: EventModifiers = []
        if command { modifiers.insert(.command) }
        if option { modifiers.insert(.option) }
        if shift { modifiers.insert(.shift) }
        if control { modifiers.insert(.control) }
        return KeyboardShortcut(KeyEquivalent(key.first ?? "?"), modifiers: modifiers)
    }
}

struct ShortcutSettingsView: View {
    @ObservedObject var store: WorkspaceStore
    @State private var selected: AppCommandID = .learningLibrary
    @State private var key = "l"
    @State private var shift = true
    @State private var option = false
    @State private var control = false
    @State private var message: String?
    private var proposed: ShortcutChord { ShortcutChord(key: key.lowercased(), option: option, shift: shift, control: control) }
    private var validationError: String? {
        do {
            var overrides = store.appPreferences.shortcutOverrides ?? [:]
            overrides[selected.rawValue] = proposed
            try AppShortcutRegistry.validateOverrides(overrides)
            return nil
        } catch { return error.localizedDescription }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionLabel(title: "Keyboard shortcuts")
            Text("Choose a command, then set a letter or digit with modifiers. Command is always included. Text editing, window and canvas shortcuts remain reserved.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            FlatChoiceControl(title: "Shortcut command", selection: selected.rawValue,
                options: AppShortcutRegistry.commands.map { .init(id: $0.id.rawValue, title: $0.title + " · " + store.shortcut(for: $0.id).displayLabel) }) { id in
                    guard let command = AppCommandID(rawValue: id) else { return }
                    selected = command; loadSelection()
                }
            HStack(spacing: 12) {
                Text("⌘ Command").font(.caption)
                Toggle("Shift", isOn: $shift)
                Toggle("Option", isOn: $option)
                Toggle("Control", isOn: $control)
                TextField("Key", text: $key).flatField().frame(width: 60)
                    .accessibilityLabel("Shortcut key").onSubmit { apply() }
            }.font(.caption)
            Text("Saved: \(store.shortcut(for: selected).displayLabel) · Proposed: \(proposed.displayLabel)")
                .font(.system(size: 12, design: .monospaced))
            if let error = validationError { Text(error).font(.caption).foregroundStyle(MLTheme.rose) }
            HStack {
                Button("Apply shortcut") { apply() }.disabled(validationError != nil)
                Button("Reset all shortcuts") {
                    if store.resetShortcuts() { loadSelection(); message = "Default shortcuts restored. Canvas defaults and saved projects are unchanged." }
                }
            }
            Text("Reset restores the ten commands listed above to their original shortcuts. Canvas defaults, layouts and learning progress are unchanged.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            if let message { Text(message).font(.caption).foregroundStyle(MLTheme.teal) }
        }.onAppear { loadSelection() }
    }
    private func loadSelection() {
        let chord = store.shortcut(for: selected)
        key = chord.key; shift = chord.shift; option = chord.option; control = chord.control; message = nil
    }
    private func apply() {
        guard validationError == nil else { return }
        if store.setShortcut(proposed, for: selected) { message = "Shortcut saved. The menu bar now shows the active binding." }
    }
}
