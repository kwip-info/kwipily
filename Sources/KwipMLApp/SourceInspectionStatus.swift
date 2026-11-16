import Foundation

/// Cached, source-scoped inspection presentation. This value performs no I/O.
struct SourceInspectionStatus: Equatable {
    enum Phase: Equatable { case inspecting, failed, cancelled, verified, stale }
    let phase: Phase
    let title: String
    let detail: String
    let action: String?
    let csvDiagnostic: CSVSourceDiagnostic?

    init(phase: Phase, detail: String) {
        self.phase = phase
        csvDiagnostic = phase == .failed ? CSVSourceDiagnostic.decodeTransport(detail) : nil
        self.detail = csvDiagnostic?.detail ?? String(detail.prefix(1000))
        switch phase {
        case .inspecting:
            title = "Inspecting source"
            action = nil
        case .failed:
            title = "Source preview unavailable"
            action = "Resolve the issue above, then Refresh preview. For file access, check read permissions or use Choose file… to grant access again."
        case .cancelled:
            title = "Source inspection cancelled"
            action = "Refresh preview when you are ready to inspect this source again."
        case .verified:
            title = "Source preview verified"
            action = nil
        case .stale:
            title = "Source inspection is out of date"
            action = "Refresh preview to inspect the current source."
        }
    }

    var isProblem: Bool { phase == .failed || phase == .stale }
    var symbolName: String {
        switch phase {
        case .inspecting: return "arrow.triangle.2.circlepath"
        case .failed: return "exclamationmark.triangle"
        case .cancelled: return "stop.circle"
        case .verified: return "checkmark.circle"
        case .stale: return "clock.arrow.circlepath"
        }
    }
}
