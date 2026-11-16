import SwiftUI
import AppKit
import KwipMLCore

enum MLTheme {
    private static func adaptive(_ role: KeyPath<AppearanceNativePalette, NSColor>) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            let matched = appearance.bestMatch(from: [.accessibilityHighContrastDarkAqua, .accessibilityHighContrastAqua, .darkAqua, .aqua])
            let dark = matched == .darkAqua || matched == .accessibilityHighContrastDarkAqua
            let contrast = matched == .accessibilityHighContrastDarkAqua || matched == .accessibilityHighContrastAqua
            return AppearanceThemePolicy(preferences: .init(mode: .system), colorScheme: dark ? .dark : .light,
                increasedContrast: contrast, reduceTransparency: true).native[keyPath: role]
        })
    }
    static let background = adaptive(\.background)
    static let panel = adaptive(\.panel)
    static let rule = adaptive(\.rule)
    static let rose = adaptive(\.accent)
    static let teal = Color(nsColor: NSColor(name: nil) { appearance in
        let dark = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        return AppearanceThemePolicy(preferences: .init(mode: .system, palette: .teal), colorScheme: dark ? .dark : .light,
            increasedContrast: false, reduceTransparency: true).native.accent
    })
    static let muted = adaptive(\.muted)
}
struct SectionLabel: View {
    let title: String
    var body: some View { Text(title.uppercased()).font(.system(size: 10, weight: .semibold)).tracking(1.3).foregroundStyle(MLTheme.muted) }
}

struct FlatButtonStyle: ButtonStyle {
    var prominent = false
    @Environment(\.appearanceTheme) private var theme
    @Environment(\.isEnabled) private var enabled
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.font(.system(size: 12, weight: .medium))
            .padding(.horizontal, 11).padding(.vertical, 7)
            .foregroundStyle(prominent ? theme.accentForeground : Color.primary)
            .background(prominent ? theme.accentColor.opacity(configuration.isPressed ? 0.7 : 1) : Color.clear)
            .overlay(Rectangle().stroke(prominent ? Color.clear : MLTheme.rule, lineWidth: 1))
            .opacity(enabled ? 1 : 0.35)
    }
}
struct FlatField: ViewModifier {
    func body(content: Content) -> some View {
        content.textFieldStyle(.plain).padding(.horizontal, 8).padding(.vertical, 7)
            .background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
    }
}
extension View { func flatField() -> some View { modifier(FlatField()) } }
