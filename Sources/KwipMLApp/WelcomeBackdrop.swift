import AppKit
import SwiftUI

/// AppKit renders the bundled vector directly; the decorative layer receives no input.
struct WelcomeBackdrop: View {
    private static let artwork: NSImage? = {
        guard let url = Bundle.module.url(forResource: "WelcomeBackdrop", withExtension: "svg") else { return nil }
        return NSImage(contentsOf: url)
    }()
    var body: some View {
        GeometryReader { geometry in
            if let artwork = Self.artwork {
                Image(nsImage: artwork).resizable().scaledToFill()
                    .frame(width: geometry.size.width, height: geometry.size.height).clipped()
            }
        }
    }
}
