import Foundation
import KwipMLCore

struct ExperimentRecord: Codable, Identifiable {
    var id: String
    var status: String
    var startedAt: Date
    var finishedAt: Date
    var graph: GraphDocument
    var epoch: Int
    var trainingLoss: Double?
    var validationLoss: Double?
    var accuracy: Double?
    var device: String
    var directory: String
    var error: String?
    var sourceHash: String?
    var splitHash: String?
    var trainingSamples: Int?
    var validationSamples: Int?
    var diagnosticReport: DiagnosticArchiveReference? = nil
    var diagnosticError: String? = nil
    var events: [String]? = nil
    var settings: ExecutionSettings { graph.execution ?? ExecutionSettings() }
    var label: String { "\(finishedAt.formatted(date: .abbreviated, time: .shortened)) · \(status) · \(epoch) epochs · split \(settings.splitSeed) · \(id.prefix(6))" }
}

extension WorkspaceStore {
    var experimentIndex: URL? {
        guard let id = activeProjectID, UUID(uuidString: id) != nil else { return nil }
        return workspaceDirectory.appendingPathComponent("experiment-index/\(id).json")
    }
    func refreshExperiments() {
        experiments = []; comparisonA = nil; comparisonB = nil
        guard let url = experimentIndex, FileManager.default.fileExists(atPath: url.path) else { return }
        do {
            experiments = try readExperimentIndex(url)
            comparisonA = experiments.first?.id; comparisonB = experiments.dropFirst().first?.id
        } catch { statusMessage = "Experiment history could not load: \(error.localizedDescription)" }
    }
    func recordExperiment(status: String, directory: URL, error: String?) {
        guard let index = experimentIndex else { return }
        let now = Date(), metric = latestReceivedMetric ?? metrics.last
        func finite(_ value: Double?) -> Double? { value.flatMap { $0.isFinite ? $0 : nil } }
        var record = ExperimentRecord(id: UUID().uuidString, status: status, startedAt: runStartedAt ?? now, finishedAt: now, graph: graph, epoch: metric?.epoch ?? 0, trainingLoss: finite(metric?.trainLoss), validationLoss: finite(metric?.validationLoss), accuracy: finite(metric?.accuracy), device: resolvedDevice, directory: directory.path, error: error)
        if status == "completed", let projectID = activeProjectID, let data = try? DiagnosticPersistence.completedManifest(record: record, projectID: projectID, root: workspaceDirectory),
           let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let provenance = manifest["provenance"] as? [String: Any] {
            record.sourceHash = provenance["sourceHash"] as? String; record.splitHash = provenance["splitHash"] as? String
            record.trainingSamples = provenance["trainingSamples"] as? Int; record.validationSamples = provenance["validationSamples"] as? Int
        }
        record.events = eventLog + (status == "cancelled" ? ["Cancelled by user"] : [])
        do {
            // Disk is authoritative: a failed refresh must never turn into an empty-history overwrite.
            var updated = FileManager.default.fileExists(atPath: index.path) ? try readExperimentIndex(index) : []
            var observed = metrics
            if let latest = latestReceivedMetric {
                let last = observed.last
                if last?.epoch != latest.epoch || last?.trainLoss.bitPattern != latest.trainLoss.bitPattern || last?.validationLoss.bitPattern != latest.validationLoss.bitPattern || last?.accuracy?.bitPattern != latest.accuracy?.bitPattern { observed.append(latest) }
            }
            do {
                record.diagnosticReport = try DiagnosticPersistence.save(record: record, projectID: activeProjectID!, metrics: observed, root: workspaceDirectory)
            } catch { record.diagnosticError = "Original diagnostics could not be saved: " + String(error.localizedDescription.prefix(1000)) }
            updated.insert(record, at: 0)
            let data = try JSONEncoder().encode(updated)
            guard data.count <= 8_000_000 else { throw NSError(domain: "KwipML", code: 1, userInfo: [NSLocalizedDescriptionKey: "Experiment history would exceed 8 MB. Existing history was preserved."]) }
            try FileManager.default.createDirectory(at: index.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: index, options: .atomic)
            experiments = updated; comparisonB = comparisonA; comparisonA = record.id
        } catch { eventLog.append("History save failed: \(error.localizedDescription). Run artifacts remain at \(directory.path).") }
    }
    func useExperimentSettings(_ record: ExperimentRecord) {
        guard !isBusy else { return }
        setExecution { $0 = record.settings }; activeSection = "Settings"
        statusMessage = "Loaded run settings as a new draft. Existing experiment remains unchanged."
    }
}

private func readExperimentIndex(_ url: URL) throws -> [ExperimentRecord] {
    let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
    guard size <= 8_000_000 else { throw NSError(domain: "KwipML", code: 1, userInfo: [NSLocalizedDescriptionKey: "Experiment index exceeds the local 8 MB limit; it was preserved for recovery."]) }
    let data = try Data(contentsOf: url)
    guard data.count <= 8_000_000 else { throw NSError(domain: "KwipML", code: 1, userInfo: [NSLocalizedDescriptionKey: "Experiment index exceeds the local 8 MB limit; it was preserved for recovery."]) }
    return try JSONDecoder().decode([ExperimentRecord].self, from: data)
}
