import SwiftUI
import KwipMLCore

struct AppearanceSettingsView: View {
    @ObservedObject var store: WorkspaceStore
    @Environment(\.appearanceTheme) private var theme
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionLabel(title: "Color mode")
            HStack {
                Text("Window appearance").font(.system(size: 13))
                Spacer()
                Picker("Color mode", selection: binding(\.mode)) {
                    Text("System").tag(AppearancePreferences.Mode.system)
                    Text("Light").tag(AppearancePreferences.Mode.light)
                    Text("Dark").tag(AppearancePreferences.Mode.dark)
                }.pickerStyle(.segmented).frame(width: 300).labelsHidden()
                    .accessibilityIdentifier("settings.appearance.mode")
            }
            Text("Applies to all \(ProductBrand.name) windows.").font(.caption).foregroundStyle(MLTheme.muted)
            HStack(spacing: 20) {
                Spacer()
                ForEach([AppearancePreferences.Mode.system, .light, .dark], id: \.self) { mode in
                    Button { store.changeAppearance {$0.mode=mode} } label: {
                        VStack(spacing: 8) {
                            ZStack(alignment: .topLeading) {
                                HStack(spacing: 0) {
                                    Rectangle().fill(mode == .dark ? Color(white: 0.15) : Color(white: 0.94))
                                    Rectangle().fill(mode == .light ? Color(white: 0.97) : Color(white: 0.19))
                                }
                                HStack(spacing: 3) {
                                    Circle().fill(Color.red.opacity(0.7)); Circle().fill(Color.yellow.opacity(0.7)); Circle().fill(Color.green.opacity(0.7))
                                }.frame(width: 24, height: 5).padding(6)
                            }.frame(width: 80, height: 58).clipShape(Rectangle())
                                .overlay(Rectangle().stroke(store.appPreferences.appearance.mode == mode ? theme.accentColor : MLTheme.rule, lineWidth: store.appPreferences.appearance.mode == mode ? 2 : 1))
                            Text(mode == .system ? "System" : mode == .light ? "Light" : "Dark").font(.caption)
                        }
                    }.buttonStyle(.plain).accessibilityLabel("Use \(mode.rawValue) appearance")
                }
            }
            Divider()
            SectionLabel(title: "Accent")
            HStack {
                Text("Interaction color").font(.system(size: 13))
                Spacer()
                Picker("Interaction accent", selection: binding(\.palette)) {
                    Text("Rose").tag(AppearancePreferences.Palette.rose)
                    Text("Teal").tag(AppearancePreferences.Palette.teal)
                }.pickerStyle(.segmented).frame(width: 230).labelsHidden()
                    .accessibilityIdentifier("settings.appearance.palette")
            }
            Text("Component colors keep their meaning.").font(.caption).foregroundStyle(MLTheme.muted)
            Divider()
            SectionLabel(title: "Panels")
            HStack {
                Text("Panel treatment").font(.system(size: 13))
                Spacer()
                FlatChoiceControl(title: "Panel treatment", selection: store.appPreferences.appearance.transparency.rawValue,
                    options: [.init(id: "opaque", title: "Opaque"), .init(id: "subtle", title: "Subtle transparency")]) { raw in
                        guard let value = AppearancePreferences.Transparency(rawValue: raw) else { return }
                        store.changeAppearance {$0.transparency=value}
                    }.frame(width: 300).accessibilityIdentifier("settings.appearance.transparency")
            }
            Text("Canvas content and text remain opaque.").font(.caption).foregroundStyle(MLTheme.muted)
            Label(theme.chromeOpaque && store.appPreferences.appearance.transparency == .subtle
                  ? "macOS accessibility preferences currently require opaque panels." : "macOS accessibility preferences take priority.", systemImage: "info.circle")
                .font(.caption).foregroundStyle(MLTheme.muted)
            Spacer(minLength: 16)
            HStack {
                Text("Only appearance preferences will be reset.").font(.caption).foregroundStyle(MLTheme.muted)
                Spacer()
                Button("Reset appearance") { store.resetAppearance() }.accessibilityIdentifier("settings.appearance.reset")
            }
        }
    }
    private func binding<Value>(_ key: WritableKeyPath<AppearancePreferences, Value>) -> Binding<Value> {
        Binding(get: { store.appPreferences.appearance[keyPath: key] }, set: { value in
            store.changeAppearance {$0[keyPath:key]=value}
        })
    }
}
