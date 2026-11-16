import AppKit
import Combine
import UserNotifications
import KwipMLCore

/// Live operations only. History and restored queue receipts never populate this registry.
@MainActor final class RunFeedbackController: ObservableObject {
    enum State: String { case queued, running, cancelling, completed, failed, cancelled
        var terminal: Bool { [.completed, .failed, .cancelled].contains(self) }
    }
    struct Activity: Equatable { let id: UUID; var state: State }
    static let shared = RunFeedbackController()
    @Published private(set) var activities: [Activity] = []
    @Published private(set) var latestOutcome: State?
    @Published private(set) var deliveryMessage = "Notifications are off."
    private var preferences = RunFeedbackPreferences.factory
    private let deliver: ((State) -> Void)?
    private var statusItem: NSStatusItem?
    private let notificationDelegate = RunFeedbackNotificationDelegate()
    private var started = false
    private let permissionQuery: ((@escaping (UNAuthorizationStatus) -> Void) -> Void)?
    private let permissionRequest: ((@escaping (Error?) -> Void) -> Void)?
    private let permissionTimeoutNanoseconds: UInt64
    private var permissionGeneration: UUID?
    private var permissionWatchdog: Task<Void, Never>?
    var onOpenSettings: (() -> Void)?
    init(deliver: ((State) -> Void)? = nil,
         permissionQuery: ((@escaping (UNAuthorizationStatus) -> Void) -> Void)? = nil,
         permissionRequest: ((@escaping (Error?) -> Void) -> Void)? = nil,
         permissionTimeoutNanoseconds: UInt64 = 8_000_000_000) {
        self.deliver = deliver; self.permissionQuery = permissionQuery; self.permissionRequest = permissionRequest
        self.permissionTimeoutNanoseconds = permissionTimeoutNanoseconds
    }
    func start(preferences: RunFeedbackPreferences) {
        started = true
        notificationCenter?.delegate = notificationDelegate
        configure(preferences)
    }
    func configure(_ value: RunFeedbackPreferences) {
        preferences = value; render()
        if !value.notifyCompleted && !value.notifyFailed {
            permissionGeneration = nil; permissionWatchdog?.cancel(); permissionWatchdog = nil
            deliveryMessage = "Notifications are off."
        } else { refreshAuthorization() }
    }
    func begin(_ id: UUID, state: State = .running) {
        guard !state.terminal, !activities.contains(where: { $0.id == id }) else { return }
        activities.append(.init(id: id, state: state)); render()
    }
    func transition(_ id: UUID, state: State) {
        guard let index = activities.firstIndex(where: { $0.id == id }) else { return }
        if state.terminal {
            activities.remove(at: index); latestOutcome = state
            if (state == .completed && preferences.notifyCompleted) || (state == .failed && preferences.notifyFailed) {
                if let deliver { deliver(state) } else { sendNotification(state) }
            }
        } else { activities[index].state = state }
        render()
    }
    func forget(_ id: UUID) { activities.removeAll { $0.id == id }; render() }
    var summary: String {
        let running = activities.filter { $0.state == .running }.count
        let cancelling = activities.filter { $0.state == .cancelling }.count
        let queued = activities.filter { $0.state == .queued }.count
        if running > 0 || queued > 0 || cancelling > 0 {
            return "\(running) running · \(queued) queued" + (cancelling > 0 ? " · \(cancelling) cancelling" : "")
        }
        switch latestOutcome {
        case .completed: return "Last run completed"
        case .failed: return "Last run failed · review History"
        case .cancelled: return "Last run cancelled"
        default: return "No active runs"
        }
    }
    private var notificationCenter: UNUserNotificationCenter? {
        // Foundation's notification service requires a real app bundle; CLI/tests are not apps.
        guard started, Bundle.main.bundleURL.pathExtension == "app", Bundle.main.bundleIdentifier != nil else { return nil }
        return .current()
    }
    private var notificationsEnabled: Bool { preferences.notifyCompleted || preferences.notifyFailed }
    private func pendingAuthorization(_ message: String, timeoutMessage: String) -> UUID {
        permissionWatchdog?.cancel()
        let generation = UUID(); permissionGeneration = generation; deliveryMessage = message
        permissionWatchdog = Task { [weak self, permissionTimeoutNanoseconds] in
            do { try await Task.sleep(nanoseconds: permissionTimeoutNanoseconds) } catch { return }
            guard let self, self.permissionGeneration == generation, self.notificationsEnabled else { return }
            self.deliveryMessage = timeoutMessage
            // A later real response may still resolve the pending state. New requests and
            // disabled preferences invalidate this generation, so stale replies cannot win.
        }
        return generation
    }
    private func acceptPermissionReply(_ generation: UUID) -> Bool {
        guard permissionGeneration == generation, notificationsEnabled else { return false }
        permissionWatchdog?.cancel(); permissionWatchdog = nil; permissionGeneration = nil
        return true
    }
    func refreshAuthorization() {
        guard notificationsEnabled else { deliveryMessage = "Notifications are off."; return }
        guard permissionQuery != nil || notificationCenter != nil else {
            permissionGeneration = nil; permissionWatchdog?.cancel(); permissionWatchdog = nil
            deliveryMessage = "System notifications require the packaged app. Run outcomes remain in History."; return
        }
        let generation = pendingAuthorization("Checking notification permission…", timeoutMessage: "macOS has not returned notification permission status. Delivery is unconfirmed; run outcomes remain in History.")
        let reply: (UNAuthorizationStatus) -> Void = { [weak self] status in
            Task { @MainActor in
                guard let self, self.acceptPermissionReply(generation) else { return }
                switch status {
                case .authorized, .provisional: self.deliveryMessage = "System notifications enabled. Focus and macOS notification settings control delivery."
                case .denied: self.deliveryMessage = "Notifications are blocked in macOS Settings. Run outcomes remain in History."
                case .notDetermined: self.deliveryMessage = "Notification permission has not been granted. Disable and re-enable a notification option to request it."
                default: self.deliveryMessage = "System notification delivery is unavailable. Run outcomes remain in History."
                }
            }
        }
        if let permissionQuery { permissionQuery(reply) }
        else { notificationCenter?.getNotificationSettings { reply($0.authorizationStatus) } }
    }
    /// Called only after the user explicitly enables a notification setting.
    func requestAuthorization() {
        guard notificationsEnabled else { return }
        guard permissionRequest != nil || notificationCenter != nil else { refreshAuthorization(); return }
        let generation = pendingAuthorization("Requesting notification permission…", timeoutMessage: "Notification permission request is still awaiting macOS. Check for a system prompt or macOS notification settings. Delivery is unconfirmed; run outcomes remain in History.")
        let reply: (Error?) -> Void = { [weak self] error in
            Task { @MainActor in
                guard let self, self.acceptPermissionReply(generation) else { return }
                if let error { self.deliveryMessage = "Notification permission could not be requested: " + error.localizedDescription }
                else { self.refreshAuthorization() }
            }
        }
        if let permissionRequest { permissionRequest(reply) }
        else { notificationCenter?.requestAuthorization(options: [.alert]) { _, error in reply(error) } }
    }
    private func sendNotification(_ state: State) {
        guard let center = notificationCenter else { return }
        center.getNotificationSettings { [weak self] settings in
            guard [.authorized, .provisional].contains(settings.authorizationStatus) else {
                Task { @MainActor in self?.refreshAuthorization() }; return
            }
            Task { @MainActor in
                guard let self, (state == .completed && self.preferences.notifyCompleted) || (state == .failed && self.preferences.notifyFailed) else { return }
                let content = UNMutableNotificationContent()
                content.title = state == .completed ? "\(ProductBrand.name) run completed" : "\(ProductBrand.name) run failed"
                content.body = state == .completed ? "Open History to inspect the accepted output." : "Open History to review the error and available diagnostic evidence."
                center.add(UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)) { [weak self] error in
                    if let error { Task { @MainActor in
                        guard let self, (state == .completed && self.preferences.notifyCompleted) || (state == .failed && self.preferences.notifyFailed) else { return }
                        self.deliveryMessage = "Notification could not be submitted: " + error.localizedDescription
                    } }
                }
            }
        }
    }
    private func render() {
        guard started else { return }
        guard preferences.showStatusIcon else {
            if let statusItem { NSStatusBar.system.removeStatusItem(statusItem); self.statusItem = nil }; return
        }
        if statusItem == nil { statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength) }
        let symbol = activities.isEmpty ? (latestOutcome == .failed ? "exclamationmark.circle" : latestOutcome == .completed ? "checkmark.circle" : latestOutcome == .cancelled ? "stop.circle" : "point.3.connected.trianglepath.dotted") : activities.allSatisfy { $0.state == .queued } ? "clock" : "play.circle"
        statusItem?.button?.image = NSImage(systemSymbolName: symbol, accessibilityDescription: "KwipML: " + summary)
        statusItem?.button?.toolTip = "\(ProductBrand.name) · " + summary
        let menu = NSMenu()
        let title = NSMenuItem(title: "\(ProductBrand.name) · " + summary, action: nil, keyEquivalent: ""); title.isEnabled = false; menu.addItem(title)
        menu.addItem(.separator())
        let settings = NSMenuItem(title: "Run feedback settings…", action: #selector(openSettings), keyEquivalent: "")
        settings.target = self; settings.isEnabled = onOpenSettings != nil; menu.addItem(settings)
        statusItem?.menu = menu
    }
    @objc private func openSettings() { onOpenSettings?() }
}

/// Explicit opt-in feedback also appears while KwipML is foreground. No sound is requested.
private final class RunFeedbackNotificationDelegate: NSObject, UNUserNotificationCenterDelegate {
    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification,
                                withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void) {
        completionHandler([.banner, .list])
    }
}
