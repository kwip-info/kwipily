import Foundation

/// Local developer diagnostics only. Fixed labels deliberately cannot carry graph,
/// source, console, port, file or user-authored content. Durations are host wall time,
/// not display-frame or pointer-to-paint measurements. Disabled by default.
public final class InteractionProfiler: @unchecked Sendable {
    public static let shared = InteractionProfiler()
    public enum Metric: String, Codable, CaseIterable {
        case canvasBody, canvasScene, geometry, projection, tensorInference, authoringCommand
        case selection, moveCommit, connectionCheck, connectionCommit, undo, redo
        case dragUpdate, wheelHandler, workerEvent, mainQueueDelay, hostLayout, scenarioAction
    }
    public enum Scenario: String, Codable, CaseIterable {
        case manual, idle, selection, move, configuration, connection, hover, undoRedo, insertionDeletion, panZoom, panDrag, wheelZoom, workerBurst
    }
    public struct Sample: Codable {
        public let metric: Metric
        public let scenario: Scenario
        public let milliseconds: Double
    }
    public struct Report: Codable {
        public let schemaVersion: Int
        public let samples: [Sample]
        public let droppedSamples: Int
    }
    public struct Token {
        fileprivate let session: UUID
        fileprivate let start: UInt64
        fileprivate let metric: Metric
        fileprivate let scenario: Scenario
    }
    private let lock = NSLock()
    private let clock: () -> UInt64
    private var session: UUID?
    private var samples: [Sample] = []
    private var dropped = 0
    private var limit = 20_000
    private var currentScenario = Scenario.manual

    public init(clock: @escaping () -> UInt64 = { DispatchTime.now().uptimeNanoseconds }) { self.clock = clock }
    public var scenario: Scenario {
        get { lock.lock(); defer { lock.unlock() }; return currentScenario }
        set { lock.lock(); defer { lock.unlock() }; currentScenario = newValue }
    }
    public func start(maximumSamples: Int = 20_000) {
        lock.lock(); defer { lock.unlock() }
        session = UUID(); samples = []; dropped = 0
        limit = max(1, min(maximumSamples, 100_000)); currentScenario = .manual
    }
    public func begin(_ metric: Metric) -> Token? {
        lock.lock(); defer { lock.unlock() }
        guard let session else { return nil }
        return Token(session: session, start: clock(), metric: metric, scenario: currentScenario)
    }
    public func end(_ token: Token?) {
        guard let token else { return }
        lock.lock(); defer { lock.unlock() }
        guard session == token.session else { return }
        let finish = clock()
        guard finish >= token.start else { return }
        guard samples.count < limit else { dropped += 1; return }
        samples.append(.init(metric: token.metric, scenario: token.scenario,
                             milliseconds: Double(finish - token.start) / 1_000_000))
    }
    public func measure<T>(_ metric: Metric, _ operation: () throws -> T) rethrows -> T {
        let token = begin(metric); defer { end(token) }
        return try operation()
    }
    public func stop() -> Report {
        lock.lock(); defer { lock.unlock() }
        session = nil
        let report = Report(schemaVersion: 1, samples: samples, droppedSamples: dropped)
        samples = []; dropped = 0
        return report
    }
}
