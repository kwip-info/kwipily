import AppKit
import Combine

/// Window layout is local UI state. Opening a saved target never starts execution.
@MainActor final class CanvasTabSessionCoordinator: ObservableObject {
    struct Binding { let target: CanvasTabSession.Target; let window: NSWindow }
    struct Failure: Identifiable {
        let target: CanvasTabSession.Target
        let reason: String
        var id: CanvasTabSession.Target { target }
    }
    @Published private(set) var failures: [Failure] = []
    @Published private(set) var status: String?
    private let store: CanvasTabSessionStore
    private let bindings: () throws -> [Binding]
    private let resolve: (CanvasTabSession.Target) throws -> NSWindow
    private let screens: () -> [NSRect]
    private let presentsWindows: Bool
    private let beginRestorationBatch: () -> Void
    private let endRestorationBatch: () -> Void
    private let includesLibrary: Bool
    private var attemptedLaunch = false
    private var attemptedRestore = false
    private var pending: CanvasTabSession?
    init(store: CanvasTabSessionStore, bindings: @escaping () throws -> [Binding],
         resolve: @escaping (CanvasTabSession.Target) throws -> NSWindow,
         screens: @escaping () -> [NSRect] = { NSScreen.screens.map(\.visibleFrame) }, presentsWindows: Bool = true, includesLibrary: Bool = true,
         beginRestorationBatch: @escaping () -> Void = {}, endRestorationBatch: @escaping () -> Void = {}) {
        self.beginRestorationBatch = beginRestorationBatch; self.endRestorationBatch = endRestorationBatch
        self.store = store; self.bindings = bindings; self.resolve = resolve; self.screens = screens; self.presentsWindows = presentsWindows; self.includesLibrary = includesLibrary
    }
    func restoreOnce(enabled: Bool) {
        guard !attemptedLaunch else { return }; attemptedLaunch = true
        if enabled { retry() }
    }
    func retry() {
        do {
            guard let snapshot = try store.load().map(canvasSnapshot) else { status = nil; return }
            beginRestorationBatch()
            defer { endRestorationBatch() }
            let retryTargets = Set(failures.map(\.target))
            attemptedRestore = true; failures = []; pending = snapshot
            if !retryTargets.isEmpty {
                // Explicit retry repairs only unavailable targets. Existing owners
                // and the user's later tab arrangement are not reconstructed.
                var current = try bindings()
                for group in snapshot.groups {
                    for target in group.tabs where retryTargets.contains(target) {
                        do {
                            let window = try resolve(target)
                            if let host = group.tabs.compactMap({ sibling in current.first { $0.target == sibling }?.window }).first, host !== window {
                                host.addTabbedWindow(window, ordered: .above)
                            } else if let frame = group.frame { window.setFrame(Self.safeFrame(frame, screens: screens()), display: false) }
                            current.append(.init(target: target, window: window))
                            if presentsWindows { window.makeKeyAndOrderFront(nil) }
                        } catch { failures.append(.init(target: target, reason: error.localizedDescription)) }
                    }
                }
                status = failures.isEmpty ? nil : "Some canvases could not reopen. Their saved targets are retained for retry."
                if failures.isEmpty { pending = nil }
                return
            }
            var selected: [Int: NSWindow] = [:]
            for (index, group) in snapshot.groups.enumerated() {
                var restored: [(Int, NSWindow)] = []
                for (tabIndex, target) in group.tabs.enumerated() {
                    do { restored.append((tabIndex, try resolve(target))) }
                    catch { failures.append(.init(target: target, reason: error.localizedDescription)) }
                }
                guard let first = restored.first?.1 else { continue }
                // Detach only the restored tabs; unrelated currently-open windows remain open.
                for (_, window) in restored { window.tabGroup?.removeWindow(window) }
                var previous = first
                for (_, window) in restored.dropFirst() { previous.addTabbedWindow(window, ordered: .above); previous = window }
                let active = restored.first(where: { $0.0 == group.selected })?.1 ?? first
                first.tabGroup?.selectedWindow = active
                if let frame = group.frame { active.setFrame(Self.safeFrame(frame, screens: screens()), display: false) }
                if presentsWindows { active.orderFront(nil) }
                selected[index] = active
            }
            if let index = snapshot.activeGroup, let window = selected[index], presentsWindows { window.makeKeyAndOrderFront(nil) }
            status = failures.isEmpty ? nil : "Some canvases could not reopen. Their saved targets are retained for retry."
            if failures.isEmpty { pending = nil }
        } catch { status = "The saved canvas session was retained: " + error.localizedDescription }
    }
    /// Invoke only around the entire application quit preparation, never per-window close.
    @discardableResult func captureAfterPreparation(_ prepare: () -> Bool) throws -> Bool {
        guard prepare() else { return false }
        let original = try store.load().map(canvasSnapshot)
        // The registries retain only live owners. Hidden and minimized windows
        // still represent open canvases and must not disappear from a session.
        let items = try bindings()
        var byWindow: [ObjectIdentifier: CanvasTabSession.Target] = [:]
        for item in items { byWindow[ObjectIdentifier(item.window)] = item.target }
        var visited = Set<ObjectIdentifier>(), groups: [CanvasTabSession.Group] = [], active: Int?
        let ordered = NSApp.orderedWindows.filter { byWindow[ObjectIdentifier($0)] != nil } + items.map(\.window)
        for window in ordered where !visited.contains(ObjectIdentifier(window)) {
            let siblings = (window.tabbedWindows ?? [window]).filter { byWindow[ObjectIdentifier($0)] != nil }
            guard !siblings.isEmpty else { continue }
            siblings.forEach { visited.insert(ObjectIdentifier($0)) }
            let selected = window.tabGroup?.selectedWindow ?? window
            let index = siblings.firstIndex(where: { $0 === selected }) ?? 0
            let frame = selected.frame
            if siblings.contains(where: { $0 === NSApp.keyWindow }) { active = groups.count }
            groups.append(.init(tabs: siblings.compactMap { byWindow[ObjectIdentifier($0)] }, selected: index,
                                frame: .init(x: frame.minX, y: frame.minY, width: frame.width, height: frame.height)))
        }
        // Retain failed or not-yet-attempted targets across another quit. Never turn
        // denied access into a destructive overwrite of the saved retry record.
        let retained = attemptedRestore ? pending : original
        var seen = Set(groups.flatMap(\.tabs))
        for group in retained?.groups ?? [] {
            let targets = group.tabs.filter { !seen.contains($0) }
            // pending is the original snapshot; only failed targets remain pending.
            let missing = attemptedRestore ? targets.filter { target in failures.contains { $0.target == target } } : targets
            guard !missing.isEmpty else { continue }
            missing.forEach { seen.insert($0) }
            let selected = missing.firstIndex(of: group.tabs[group.selected]) ?? 0
            groups.append(.init(tabs: missing, selected: selected, frame: group.frame))
        }
        try store.save(.init(groups: groups, activeGroup: groups.isEmpty ? nil : active ?? 0))
        return true
    }
    /// Old sessions can contain the application launcher. Remove only that UI target;
    /// keep canvas order, selected file, frames and unavailable file retry identities.
    private func canvasSnapshot(_ snapshot: CanvasTabSession) -> CanvasTabSession {
        guard !includesLibrary else { return snapshot }
        var groups: [CanvasTabSession.Group] = [], active: Int?
        for (index, group) in snapshot.groups.enumerated() {
            let tabs = group.tabs.filter { $0 != .library }
            guard !tabs.isEmpty else { continue }
            if snapshot.activeGroup == index { active = groups.count }
            let selected = tabs.firstIndex(of: group.tabs[group.selected]) ?? 0
            groups.append(.init(tabs: tabs, selected: selected, frame: group.frame))
        }
        return .init(groups: groups, activeGroup: groups.isEmpty ? nil : active ?? 0)
    }
    static func safeFrame(_ frame: CanvasTabSession.Frame, screens: [NSRect]) -> NSRect {
        let desired = NSRect(x: frame.x, y: frame.y, width: frame.width, height: frame.height)
        let available = screens.filter { !$0.isEmpty && !$0.isInfinite }
        let screen = available.max { a, b in
            let ia = a.intersection(desired), ib = b.intersection(desired)
            return (ia.isNull ? 0 : ia.width * ia.height) < (ib.isNull ? 0 : ib.width * ib.height)
        } ?? NSRect(x: 0, y: 0, width: 1200, height: 800)
        let width = min(max(desired.width, 100), screen.width), height = min(max(desired.height, 100), screen.height)
        return NSRect(x: min(max(desired.minX, screen.minX), screen.maxX - width),
                      y: min(max(desired.minY, screen.minY), screen.maxY - height), width: width, height: height)
    }
}
