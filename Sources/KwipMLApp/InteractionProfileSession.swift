import AppKit
import KwipMLCore

/// A single outstanding probe prevents a busy main queue accumulating diagnostics.
final class InteractionMainQueueProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = false
    private var active = true
    private let timer: DispatchSourceTimer
    init() {
        timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "kwipml.profile.probe", qos: .utility))
        timer.schedule(deadline: .now(), repeating: .milliseconds(20))
        timer.setEventHandler { [weak self] in self?.tick() }
        timer.resume()
    }
    private func tick() {
        lock.lock()
        guard active, !pending else { lock.unlock(); return }
        pending = true; lock.unlock()
        let token = InteractionProfiler.shared.begin(.mainQueueDelay)
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.lock.lock(); let active = self.active; self.pending = false; self.lock.unlock()
            if active { InteractionProfiler.shared.end(token) }
        }
    }
    func stop() { lock.lock(); active = false; lock.unlock(); timer.cancel() }
    deinit { timer.cancel() }
}

/// Explicit environment opt-in on the development identity only. No settings,
/// project content, worker logs or user-authored strings enter the report.
@MainActor enum InteractionProfileSession {
    private static var probe: InteractionMainQueueProbe?
    private static var deadline: DispatchWorkItem?
    private static var termination: NSObjectProtocol?
    static func request(bundleIdentifier: String?, environment: [String: String]) -> (url: URL, seconds: Double)? {
        guard bundleIdentifier == "info.kwip.KwipML.POC",
              let path = environment["KWIPML_INTERACTION_PROFILE"], path.hasPrefix("/"), !path.contains("\0"),
              let seconds = Double(environment["KWIPML_INTERACTION_PROFILE_SECONDS"] ?? "60"),
              seconds.isFinite, (1...300).contains(seconds) else { return nil }
        return (URL(fileURLWithPath: path), seconds)
    }
    static func startFromEnvironment() {
        guard let request = request(bundleIdentifier: Bundle.main.bundleIdentifier, environment: ProcessInfo.processInfo.environment),
              !FileManager.default.fileExists(atPath: request.url.path), probe == nil else { return }
        InteractionProfiler.shared.start()
        probe = InteractionMainQueueProbe()
        let finish = DispatchWorkItem {
            MainActor.assumeIsolated { finishCapture(to: request.url) }
        }
        deadline = finish
        DispatchQueue.main.asyncAfter(deadline: .now() + request.seconds, execute: finish)
        termination = NotificationCenter.default.addObserver(forName: NSApplication.willTerminateNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { finishCapture(to: request.url) }
        }
    }
    private static func finishCapture(to url: URL) {
        guard let running = probe else { return }
        running.stop(); probe = nil; deadline?.cancel(); deadline = nil
        if let termination { NotificationCenter.default.removeObserver(termination) }; termination = nil
        let report = InteractionProfiler.shared.stop()
        // Bounded report is written once, including on normal app termination.
        do { try writeCapture(report, to: url) }
        catch { NSLog("Kwip ML interaction profile could not be saved; no profile receipt was produced.") }
    }
    static func writeCapture(_ report: InteractionProfiler.Report, to url: URL) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let temporary = url.deletingLastPathComponent().appendingPathComponent(UUID().uuidString + ".profile-tmp")
        defer { try? FileManager.default.removeItem(at: temporary) }
        try encoder.encode(report).write(to: temporary, options: .withoutOverwriting)
        // Exclusive publication exposes complete bytes and refuses an existing file.
        try FileManager.default.linkItem(at: temporary, to: url)
    }
}
