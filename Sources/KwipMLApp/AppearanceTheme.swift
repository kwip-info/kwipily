import SwiftUI
import AppKit
import KwipMLCore

/// A value derived from saved choices and this presentation's effective environment.
/// Accessibility overrides presentation only; they never rewrite the saved preference.
struct AppearanceThemePolicy {
    let preferences: AppearancePreferences
    let colorScheme: ColorScheme
    let increasedContrast: Bool
    let reduceTransparency: Bool

    init(preferences: AppearancePreferences, colorScheme: ColorScheme,
         increasedContrast: Bool, reduceTransparency: Bool) {
        self.preferences = preferences
        switch preferences.mode {
        case .dark: self.colorScheme = .dark
        case .light: self.colorScheme = .light
        case .system: self.colorScheme = colorScheme
        }
        self.increasedContrast = increasedContrast
        self.reduceTransparency = reduceTransparency
    }
    var chromeOpaque: Bool {
        preferences.transparency == .opaque || increasedContrast || reduceTransparency
    }
    var syntaxPaletteFingerprint: String {
        "syntax-v1/\(colorScheme == .dark ? "dark" : "light")/\(preferences.palette.rawValue)/\(increasedContrast ? "increased" : "normal")"
    }
    var accentColor: Color { Color(nsColor: native.accent) }
    var accentForeground: Color { Color(nsColor: native.accentForeground) }
    var backgroundColor: Color { Color(nsColor: native.background) }
    var panelColor: Color { Color(nsColor: native.panel) }

    var native: AppearanceNativePalette {
        let dark = colorScheme == .dark
        let background = dark ? rgb(0.094, 0.094, 0.110) : rgb(0.975, 0.975, 0.983)
        let panel = dark ? rgb(0.12, 0.12, 0.14) : rgb(0.94, 0.945, 0.958)
        let text = dark ? rgb(0.92, 0.925, 0.95) : rgb(0.13, 0.14, 0.17)
        let muted: NSColor
        if dark { muted = increasedContrast ? rgb(0.82, 0.84, 0.88) : rgb(0.64, 0.66, 0.71) }
        else { muted = increasedContrast ? rgb(0.22, 0.24, 0.28) : rgb(0.36, 0.38, 0.42) }
        let accent: NSColor
        switch (dark, preferences.palette) {
        case (true, .rose): accent = rgb(0.79, 0.55, 0.62)
        case (true, .teal): accent = rgb(0.43, 0.77, 0.72)
        case (false, .rose): accent = rgb(0.52, 0.22, 0.33)
        case (false, .teal): accent = rgb(0.08, 0.37, 0.34)
        }
        let rule: NSColor
        if dark { rule = increasedContrast ? rgb(0.65, 0.67, 0.72) : rgb(0.24, 0.24, 0.26) }
        else { rule = increasedContrast ? rgb(0.35, 0.37, 0.41) : rgb(0.78, 0.79, 0.82) }
        return AppearanceNativePalette(background: background, panel: panel, text: text, muted: muted,
            rule: rule, accent: accent, accentForeground: dark ? background : rgb(1, 1, 1),
            keyword: accent,
            string: dark ? rgb(0.52, 0.80, 0.70) : rgb(0.10, 0.38, 0.24),
            comment: muted,
            number: dark ? rgb(0.85, 0.73, 0.51) : rgb(0.48, 0.30, 0.06),
            name: dark ? rgb(0.57, 0.73, 0.90) : rgb(0.12, 0.32, 0.58))
    }
    private func rgb(_ red: CGFloat, _ green: CGFloat, _ blue: CGFloat) -> NSColor {
        NSColor(srgbRed: red, green: green, blue: blue, alpha: 1)
    }
}

/// Resolved opaque colors shared by SwiftUI and native code editors. Component
/// category hues remain separate from the user-selected interactive accent.
struct AppearanceNativePalette {
    let background: NSColor
    let panel: NSColor
    let text: NSColor
    let muted: NSColor
    let rule: NSColor
    let accent: NSColor
    let accentForeground: NSColor
    let keyword: NSColor
    let string: NSColor
    let comment: NSColor
    let number: NSColor
    let name: NSColor
}
