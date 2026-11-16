import Foundation
import KwipMLCore

extension WorkspaceStore {
    var appPreferenceStore: AppPreferencesStore {
        AppPreferencesStore(fileURL: applicationDirectory.appendingPathComponent("app-preferences.json"))
    }
    @discardableResult func setRestoreCanvasSessionAtLaunch(_ enabled: Bool) -> Bool {
        changeCanvasPreferences {$0.restoreAtLaunch=enabled}
    }
    @discardableResult func setAuthoringRetention(_ policy: FlowAuthoringRetentionPreferences) -> Bool {
        do {
            var next = try appPreferenceStore.load(); next.authoringRetention = policy
            try appPreferenceStore.save(next); appPreferences = next; appPreferencesError = nil; return true
        } catch { appPreferencesError = "Authoring evidence allowance was not saved. " + error.localizedDescription; return false }
    }
    func loadAppPreferences() {
        do { appPreferences = try appPreferenceStore.load(); appPreferencesError = nil }
        catch { appPreferencesError = "App preferences could not be loaded. Factory defaults are in use. \(error.localizedDescription)" }
    }
    @discardableResult func setCanvasZoomSensitivity(_ value: Double) -> Bool {
        changeCanvasPreferences {$0.zoomSensitivity=value}
    }
    @discardableResult func setCanvasDefaults(_ settings: CanvasPresentationSettings) -> Bool {
        do {
            var next = try appPreferenceStore.load()
            next.canvasDefaults = settings
            try appPreferenceStore.save(next)
            appPreferences = next; appPreferencesError = nil
            return true
        } catch { appPreferencesError = "Canvas defaults were not saved. \(error.localizedDescription)"; return false }
    }
    @discardableResult func resetCanvasDefaults() -> Bool {
        do {
            appPreferences = try appPreferenceStore.resetCanvasDefaults()
            appPreferencesError = nil; return true
        } catch { appPreferencesError = "Canvas defaults were not reset. \(error.localizedDescription)"; return false }
    }
    @discardableResult func setAppearance(_ appearance: AppearancePreferences) -> Bool {
        changeApplicationSettings(.appearance(appearance))
    }
    @discardableResult func resetAppearance() -> Bool {
        changeApplicationSettings(.reset(.appearance))
    }
    func shortcut(for command: AppCommandID) -> ShortcutChord {
        (try? AppShortcutRegistry.effectiveBindings(overrides: appPreferences.shortcutOverrides)[command]) ?? AppShortcutRegistry.definition(for: command).defaultShortcut
    }
    @discardableResult func setShortcut(_ chord: ShortcutChord, for command: AppCommandID) -> Bool {
        do {
            var next = try appPreferenceStore.load()
            var overrides = next.shortcutOverrides ?? [:]
            if chord == AppShortcutRegistry.definition(for: command).defaultShortcut { overrides.removeValue(forKey: command.rawValue) }
            else { overrides[command.rawValue] = chord }
            next.shortcutOverrides = overrides.isEmpty ? nil : overrides
            try AppShortcutRegistry.validateOverrides(next.shortcutOverrides)
            try appPreferenceStore.save(next)
            appPreferences = next; appPreferencesError = nil; return true
        } catch { appPreferencesError = "Shortcut was not saved. \(error.localizedDescription)"; return false }
    }
    @discardableResult func resetShortcuts() -> Bool {
        do {
            appPreferences = try appPreferenceStore.resetShortcuts()
            appPreferencesError = nil; return true
        } catch { appPreferencesError = "Shortcuts were not reset. \(error.localizedDescription)"; return false }
    }
    func initialWorkspaceView() -> WorkspaceViewState {
        WorkspaceViewState(modelCanvas: CanvasViewState(presentation: appPreferences.canvasDefaults),
                           dataCanvas: CanvasViewState(presentation: appPreferences.canvasDefaults))
    }
}
