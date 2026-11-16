import AppKit
import SwiftUI

/// Coordinates are screen coordinates (AppKit's upward-positive Y axis).
enum AttachedDropdownPlacement {
    static func frame(anchor: CGRect, visibleScreen: CGRect, preferredHeight: CGFloat, width: CGFloat? = nil) -> CGRect {
        let menuWidth = min(max(1, width ?? anchor.width), visibleScreen.width)
        let below = max(0, anchor.minY - visibleScreen.minY)
        let above = max(0, visibleScreen.maxY - anchor.maxY)
        let useBelow = below >= preferredHeight || below >= above
        let height = min(max(1, preferredHeight), useBelow ? below : above)
        let x = min(max(anchor.minX, visibleScreen.minX), visibleScreen.maxX - menuWidth)
        let y = useBelow ? anchor.minY - height : anchor.maxY
        return CGRect(x: x, y: y, width: menuWidth, height: height)
    }
}

enum AttachedDropdownFocus {
    @MainActor static func commandOwner(for window: NSWindow?) -> NSWindow? {
        guard let window else { return nil }
        // Only our transient dropdown is part of its parent document. App settings,
        // Block windows and arbitrary child windows keep their independent scope.
        return window is AttachedDropdownPanel ? window.parent : window
    }
    static func shouldRestoreOwner(panelWasKey: Bool, ownerVisible: Bool, applicationActive: Bool, keyWindowIsUnassigned: Bool = false) -> Bool {
        (panelWasKey || keyWindowIsUnassigned) && ownerVisible && applicationActive
    }
}

enum AttachedDropdownDismissal: Equatable { case outsideInteraction, focusLost }

extension View {
    /// A square, attached menu surface. Default width follows the trigger.
    func attachedDropdown<Content: View>(
        isPresented: Binding<Bool>, preferredHeight: CGFloat, width: CGFloat? = nil,
        onDismiss: @escaping (AttachedDropdownDismissal) -> Void = { _ in },
        @ViewBuilder content: @escaping () -> Content
    ) -> some View {
        modifier(AttachedDropdownModifier(isPresented: isPresented, preferredHeight: preferredHeight,
                                         width: width, onDismiss: onDismiss, dropdown: content))
    }
}

private struct AttachedDropdownModifier<Dropdown: View>: ViewModifier {
    @Binding var isPresented: Bool
    let preferredHeight: CGFloat
    let width: CGFloat?
    let onDismiss: (AttachedDropdownDismissal) -> Void
    let dropdown: () -> Dropdown
    @Environment(\.appearanceTheme) private var theme
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.isEnabled) private var isEnabled

    func body(content: Content) -> some View {
        content.background(AttachedDropdownAnchor(
            isPresented: $isPresented, preferredHeight: preferredHeight, width: width,
            onDismiss: onDismiss,
            content: AnyView(dropdown().environment(\.appearanceTheme, theme)
                .environment(\.colorScheme, colorScheme).environment(\.isEnabled, isEnabled))
        ).allowsHitTesting(false))
    }
}

final class AttachedDropdownPanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}

private final class DropdownAnchorView: NSView {
    var geometryChanged: (() -> Void)?
    override func viewDidMoveToWindow() { super.viewDidMoveToWindow(); geometryChanged?() }
    override func setFrameSize(_ newSize: NSSize) { super.setFrameSize(newSize); geometryChanged?() }
    override func setFrameOrigin(_ newOrigin: NSPoint) { super.setFrameOrigin(newOrigin); geometryChanged?() }
}

private struct AttachedDropdownAnchor: NSViewRepresentable {
    @Binding var isPresented: Bool
    let preferredHeight: CGFloat
    let width: CGFloat?
    let onDismiss: (AttachedDropdownDismissal) -> Void
    let content: AnyView

    func makeCoordinator() -> Coordinator { Coordinator() }
    func makeNSView(context: Context) -> DropdownAnchorView {
        let view = DropdownAnchorView()
        view.geometryChanged = { [weak coordinator = context.coordinator] in
            // SwiftUI can attach the anchor after updateNSView has already run.
            DispatchQueue.main.async { coordinator?.refresh() }
        }
        context.coordinator.anchor = view
        return view
    }
    func updateNSView(_ view: DropdownAnchorView, context: Context) {
        context.coordinator.configuration = self
        context.coordinator.refresh()
    }
    static func dismantleNSView(_ view: DropdownAnchorView, coordinator: Coordinator) {
        view.geometryChanged = nil
        coordinator.close()
    }

    final class Coordinator: NSObject, NSWindowDelegate {
        weak var anchor: DropdownAnchorView?
        var configuration: AttachedDropdownAnchor?
        private var panel: AttachedDropdownPanel?
        private var hosting: NSHostingView<AnyView>?
        private var localMonitor: Any?
        private var globalMonitor: Any?
        private var observers: [NSObjectProtocol] = []

        func refresh() {
            guard let configuration, configuration.isPresented else { close(); return }
            guard let anchor, let owner = anchor.window, owner.isVisible,
                  let screen = owner.screen ?? NSScreen.main, anchor.bounds.width > 0 else { close(); return }
            let anchorFrame = owner.convertToScreen(anchor.convert(anchor.bounds, to: nil))
            let frame = AttachedDropdownPlacement.frame(anchor: anchorFrame, visibleScreen: screen.visibleFrame,
                                                       preferredHeight: configuration.preferredHeight, width: configuration.width)
            if let panel {
                // Updating choices must retain list focus and scroll state.
                hosting?.rootView = configuration.content
                panel.setFrame(frame, display: true)
                return
            }
            let panel = AttachedDropdownPanel(contentRect: frame, styleMask: [.borderless], backing: .buffered, defer: false)
            panel.isFloatingPanel = true
            panel.level = .popUpMenu
            panel.hasShadow = true
            panel.isOpaque = false
            panel.backgroundColor = .clear
            panel.isReleasedWhenClosed = false
            panel.delegate = self
            let hosting = NSHostingView(rootView: configuration.content)
            panel.contentView = hosting
            self.hosting = hosting
            self.panel = panel
            owner.addChildWindow(panel, ordered: .above)
            panel.makeKeyAndOrderFront(nil)
            installDismissalObservers(owner: owner)
        }

        private func installDismissalObservers(owner: NSWindow) {
            localMonitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown, .otherMouseDown, .keyDown]) { [weak self] event in
                guard let self, let panel = self.panel else { return event }
                if event.type == .keyDown, event.keyCode == 53 {
                    self.dismiss(.focusLost)
                    return nil
                }
                if event.type != .keyDown, !panel.frame.contains(NSEvent.mouseLocation) {
                    // Consume a trigger click after closing. Otherwise AppKit
                    // resigns the panel before the button action and the action
                    // can immediately reopen the menu it meant to close.
                    let trigger = self.anchor.flatMap { view in
                        view.window?.convertToScreen(view.convert(view.bounds, to: nil))
                    }
                    if trigger?.contains(NSEvent.mouseLocation) == true {
                        self.dismiss(.focusLost)
                        return nil
                    }
                    self.dismiss(.outsideInteraction)
                }
                return event
            }
            globalMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown, .otherMouseDown]) { [weak self] _ in
                self?.dismiss(.outsideInteraction)
            }
            let center = NotificationCenter.default
            observers.append(center.addObserver(forName: NSApplication.didResignActiveNotification, object: nil, queue: .main) { [weak self] _ in self?.dismiss(.outsideInteraction) })
            observers.append(center.addObserver(forName: NSWindow.willCloseNotification, object: owner, queue: .main) { [weak self] _ in self?.dismiss(.outsideInteraction) })
            for name in [NSWindow.didMoveNotification, NSWindow.didResizeNotification] {
                observers.append(center.addObserver(forName: name, object: owner, queue: .main) { [weak self] _ in self?.refresh() })
            }
        }

        func windowDidResignKey(_ notification: Notification) { dismiss(.outsideInteraction) }

        private func dismiss(_ reason: AttachedDropdownDismissal) {
            guard panel != nil, let configuration else { return }
            close()
            configuration.onDismiss(reason)
            configuration.isPresented = false
        }

        func close() {
            if let localMonitor { NSEvent.removeMonitor(localMonitor); self.localMonitor = nil }
            if let globalMonitor { NSEvent.removeMonitor(globalMonitor); self.globalMonitor = nil }
            observers.forEach(NotificationCenter.default.removeObserver)
            observers.removeAll()
            panel?.delegate = nil
            if let panel {
                let owner = panel.parent
                let restore = AttachedDropdownFocus.shouldRestoreOwner(panelWasKey: panel.isKeyWindow,
                    ownerVisible: owner?.isVisible == true, applicationActive: NSApp.isActive,
                    keyWindowIsUnassigned: NSApp.keyWindow == nil)
                owner?.removeChildWindow(panel); panel.orderOut(nil)
                // Closing a key child panel does not reliably restore AppKit's
                // document key window. Do so only while this app/owner still owns focus.
                if restore { owner?.makeKeyAndOrderFront(nil) }
            }
            panel = nil
            hosting = nil
        }
    }
}
