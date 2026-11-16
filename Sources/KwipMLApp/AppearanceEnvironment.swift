import SwiftUI
import KwipMLCore

/// Shared by window roots, without retaining their workspace or editor sessions.
@MainActor final class AppAppearanceState: ObservableObject {
    @Published var preferences: AppearancePreferences = .factory
    @Published var canvasZoomSensitivity:Double = 1
}
private struct AppearanceThemeKey: EnvironmentKey {
    static let defaultValue = AppearanceThemePolicy(preferences: .factory, colorScheme: .dark,
        increasedContrast: false, reduceTransparency: false)
}
private struct CanvasZoomSensitivityKey:EnvironmentKey {static let defaultValue:Double = 1}
extension EnvironmentValues {
    var canvasZoomSensitivity:Double {get {self[CanvasZoomSensitivityKey.self]} set {self[CanvasZoomSensitivityKey.self]=newValue}}
    var appearanceTheme: AppearanceThemePolicy {
        get { self[AppearanceThemeKey.self] }
        set { self[AppearanceThemeKey.self] = newValue }
    }
}
struct AppAppearanceModifier: ViewModifier {
    @ObservedObject var state: AppAppearanceState
    @Environment(\.colorScheme) private var scheme
    @Environment(\.colorSchemeContrast) private var contrast
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    private var preferred: ColorScheme? {
        switch state.preferences.mode { case .dark: return .dark; case .light: return .light; case .system: return nil }
    }
    func body(content: Content) -> some View {
        let theme = AppearanceThemePolicy(preferences: state.preferences, colorScheme: scheme,
            increasedContrast: contrast == .increased, reduceTransparency: reduceTransparency)
        content.environment(\.canvasZoomSensitivity,state.canvasZoomSensitivity).environment(\.appearanceTheme, theme)
            .preferredColorScheme(preferred)
            .tint(theme.accentColor).accentColor(theme.accentColor)
            .transaction { if reduceMotion { $0.animation = nil; $0.disablesAnimations = true } }
    }
}
struct AppearanceChrome: ViewModifier {
    @Environment(\.appearanceTheme) private var theme
    func body(content: Content) -> some View {
        content.background {
            if theme.chromeOpaque { theme.panelColor }
            else { Rectangle().fill(.ultraThinMaterial).overlay(theme.panelColor.opacity(0.82)) }
        }
    }
}
extension View {
    func appAppearance(_ state: AppAppearanceState) -> some View { modifier(AppAppearanceModifier(state: state)) }
    func appearanceChrome() -> some View { modifier(AppearanceChrome()) }
}
