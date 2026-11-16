import SwiftUI

/// Shared canvas controls keep a stable hit region even when narrow layouts hide text.
struct CanvasToolbarButton: View {
    let title: String
    let symbol: String
    var compact = false
    var help: String? = nil
    let action: () -> Void
    @Environment(\.isEnabled) private var enabled
    @State private var hovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: symbol).font(.system(size: 11, weight: .medium))
                if !compact { Text(title).font(.system(size: 11, weight: .medium)).fixedSize() }
            }
            .frame(minWidth: 28, minHeight: 28)
            .padding(.horizontal, compact ? 0 : 6)
            .background(hovered && enabled ? MLTheme.rose.opacity(0.12) : Color.clear)
            .contentShape(Rectangle())
            .opacity(enabled ? 1 : 0.35)
        }.buttonStyle(.plain)
            .onHover { hovered = $0 }
            .help(help ?? title)
            .accessibilityLabel(title)
    }
}
