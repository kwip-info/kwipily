import AppKit

/// Presentation metadata only. Never rewrites document paths or changes file ownership.
struct DocumentWindowIdentity: Equatable {
    enum Kind: String { case project = "Project", block = "Block / subflow", lesson = "Lesson" }
    enum Location: String { case managed = "Managed project", external = "External folder", localFile = "Local file", unsaved = "Unsaved" }
    let kind: Kind
    let location: Location
    let representedURL: URL?
    var lessonKind:String? = nil
    var symbol: String {
        switch kind { case .project: return "square.grid.2x2.fill"; case .block: return "shippingbox.fill"; case .lesson: return lessonKind.map {LessonWindowPresentation.symbol(kind:$0)} ?? "book.closed.fill" }
    }
    static func project(url: URL?, isLesson: Bool, lessonKind:String? = nil, isExternal: Bool, isManaged: Bool) -> Self {
        .init(kind: isLesson ? .lesson : .project,
              location: url == nil ? .unsaved : isExternal ? .external : isManaged ? .managed : .localFile,
              representedURL: url,lessonKind:lessonKind)
    }
    static func block(url: URL, restorationTarget: CanvasTabSession.Target?, hasExternalAccess: Bool) -> Self {
        let location: Location
        if hasExternalAccess { location = .external }
        else {
            switch restorationTarget {
            case .subflow(root: .managed, path: _, documentID: _): location = .managed
            case .subflow(root: .external, path: _, documentID: _), .standalone: location = .external
            default: location = .localFile
            }
        }
        return .init(kind: .block, location: location, representedURL: url)
    }
    @MainActor func apply(to window: NSWindow) {
        // Assigning representedURL retains AppKit's document path menu and drag proxy.
        window.representedURL = representedURL
        let description = lessonKind.map {$0+" lesson"} ?? kind.rawValue
        let tooltip = description + " · " + location.rawValue + (representedURL.map { "\n" + $0.path } ?? "")
        let image = icon()
        if let proxy = window.standardWindowButton(.documentIconButton) {
            proxy.image = image; proxy.toolTip = tooltip
            proxy.setAccessibilityLabel(description + " · " + location.rawValue)
        }
        let accessory = NSImageView(frame: .init(x: 0, y: 0, width: 18, height: 18))
        accessory.image = image; accessory.imageScaling = .scaleProportionallyDown
        accessory.toolTip = tooltip
        accessory.setAccessibilityLabel(description + " · " + location.rawValue)
        accessory.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([accessory.widthAnchor.constraint(equalToConstant: 18), accessory.heightAnchor.constraint(equalToConstant: 18)])
        window.tab.accessoryView = accessory
        window.tab.toolTip = tooltip
    }
    @MainActor private func icon() -> NSImage {
        let glyph = NSImage(systemSymbolName: symbol, accessibilityDescription: kind.rawValue)
        let external = location == .external
        let color: NSColor
        switch kind { case .project: color = .systemPink; case .block: color = .systemTeal; case .lesson: color = .systemPurple }
        return NSImage(size: .init(width: 18, height: 18), flipped: false) { bounds in
            color.withAlphaComponent(0.18).setFill(); NSBezierPath(rect: bounds.insetBy(dx: 1, dy: 1)).fill()
            glyph?.withSymbolConfiguration(.init(paletteColors: [color]))?.draw(in: bounds.insetBy(dx: 3, dy: 3))
            if external {
                NSColor.windowBackgroundColor.setFill(); NSBezierPath(rect: .init(x: 10, y: 10, width: 8, height: 8)).fill()
                NSImage(systemSymbolName: "arrow.up.right", accessibilityDescription: "External folder")?
                    .withSymbolConfiguration(.init(paletteColors: [NSColor.labelColor]))?.draw(in: .init(x: 11, y: 11, width: 6, height: 6))
            }
            return true
        }
    }
}
