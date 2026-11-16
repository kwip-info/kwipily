import XCTest
import Foundation
import KwipMLCore
@testable import KwipMLApp

/// Real worker artifacts generated separately; no synthetic gradients or metric substitutions.
final class MeasuredDiagnosticTrainedCorpusTests: XCTestCase {
    private struct Pin: Decodable { let path: String; let sha256: String }
    private struct FilePin: Decodable { let path: String; let bytes: Int; let sha256: String }
    private struct Runtime: Decodable { let python: String; let pytorch: String; let device: String; let threads: Int }
    private struct Configuration: Decodable {
        let epochs: Int; let initialEpoch: Int; let trainedEpochs: Int
        let learningRate: Double; let maxCaptures: Int; let batchSize: Int; let device: String; let seed: Int; let validationPercent: Int
    }
    private struct CasePins: Decodable {
        let gradientSHA256: String; let modelManifestSHA256: String; let metricsSHA256: String
        let preparationManifestSHA256: String; let sourceSHA256: String; let modelGraphSHA256: String; let captureConfigurationSHA256: String
    }
    private struct Growth: Decodable { let triggered: Bool; let withheld: String? }
    private struct Expectations: Decodable {
        let expectedFindings: [String]; let validationRows: Int; let capturedBatches: Int
        let growth: Growth; let withheldAlways: [String]
    }
    private struct Fixture: Decodable {
        let id: String; let runID: String; let configuration: Configuration; let pins: CasePins
        let modelManifest: String; let gradientArtifact: String; let metricsArtifact: String
        let expectations: Expectations
    }
    private struct Corpus: Decodable {
        let schemaVersion: Int; let corpusID: String; let runtime: Runtime
        let sourcePins: [Pin]; let files: [FilePin]; let cases: [Fixture]
    }
    private struct MetricWire: Decodable {
        let epoch: Int; let trainLoss: Double; let validationLoss: Double; let accuracy: Double
        var metric: EpochMetric { .init(epoch: epoch, trainLoss: trainLoss, validationLoss: validationLoss, accuracy: accuracy) }
    }
    private struct TrainingManifest: Decodable {
        struct Settings: Decodable {
            let epochs: Int; let initialEpoch: Int; let epochsExecuted: Int
            let optimizer: String; let learningRate: Double; let requestedBatchSize: Int
            let deterministicAlgorithms: Bool; let partitionMode: String
        }
        struct Provenance: Decodable {
            let trainingSamples: Int; let validationSamples: Int
            let trainingRowIDs: [String]; let validationRowIDs: [String]
        }
        struct Lineage: Decodable { let mode: String; let parentRunID: String? }
        struct WorkerRuntime: Decodable { let python: String; let torch: String; let device: String }
        let status: String; let runID: String; let seed: Int
        let settings: Settings; let provenance: Provenance; let lineage: Lineage; let runtime: WorkerRuntime
    }
    private func checkedBytes(_ relative: String, directory: URL, pins: [String: FilePin]) throws -> Data {
        guard !relative.hasPrefix("/"), !relative.split(separator: "/").contains(".."),
              let pin = pins[relative] else { throw NSError(domain: "Corpus", code: 1) }
        let path = directory.appendingPathComponent(relative)
        guard path.resolvingSymlinksInPath().path.hasPrefix(directory.resolvingSymlinksInPath().path + "/"),
              try path.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey]).isRegularFile == true,
              try path.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink != true else {
            throw NSError(domain: "Corpus", code: 2)
        }
        let bytes = try Data(contentsOf: path)
        guard bytes.count == pin.bytes, WorkflowArchiveCodec.hash(bytes) == pin.sha256 else {
            throw NSError(domain: "Corpus", code: 3, userInfo: [NSLocalizedDescriptionKey: "Corpus artifact hash/size mismatch: " + relative])
        }
        return bytes
    }
    func testSixActualTrainingCasesReplayStrictGradientsAndMeasuredRegistry() throws {
        guard let path = ProcessInfo.processInfo.environment["KWIPML_DIAGNOSTIC_CORPUS"] else {
            throw XCTSkip("Generate scripts/measured_diagnostic_training_fixtures.py and supply KWIPML_DIAGNOSTIC_CORPUS.")
        }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        let bytes = try Data(contentsOf: directory.appendingPathComponent("corpus.json"))
        let expectedPin = ProcessInfo.processInfo.environment["KWIPML_DIAGNOSTIC_CORPUS_SHA256"]
            ?? "fa461fa54f7bf3af6cd8e89161fdfb32e3a06aab147d4048272d9138d493c45f"
        XCTAssertEqual(WorkflowArchiveCodec.hash(bytes), expectedPin, "A replacement corpus requires its explicit reviewed pin.")
        let corpus = try JSONDecoder().decode(Corpus.self, from: bytes)
        XCTAssertEqual(corpus.schemaVersion, 1)
        XCTAssertEqual(corpus.corpusID, "kwip.cpu-trained-diagnostic-controls.v1")
        XCTAssertEqual(corpus.runtime.device, "cpu"); XCTAssertEqual(corpus.runtime.threads, 1)
        XCTAssertEqual(Set(corpus.cases.map(\.id)), Set(["ordinary-baseline", "gradient-growth", "lowered-rate-control", "tiny-holdout", "insufficient-gradient-window", "exact-continuation"]))
        XCTAssertEqual(corpus.cases.count, 6)
        XCTAssertEqual(Set(corpus.files.map(\.path)).count, corpus.files.count)
        let pins = Dictionary(uniqueKeysWithValues: corpus.files.map { ($0.path, $0) })
        for file in corpus.files { _ = try checkedBytes(file.path, directory: directory, pins: pins) }
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for pin in corpus.sourcePins {
            XCTAssertEqual(WorkflowArchiveCodec.hash(try Data(contentsOf: repo.appendingPathComponent(pin.path))), pin.sha256,
                           "Source changed since the corpus was generated: " + pin.path)
        }
        for fixture in corpus.cases {
            let modelBytes = try checkedBytes(fixture.modelManifest, directory: directory, pins: pins)
            let gradientBytes = try checkedBytes(fixture.gradientArtifact, directory: directory, pins: pins)
            let metricBytes = try checkedBytes(fixture.metricsArtifact, directory: directory, pins: pins)
            XCTAssertEqual(WorkflowArchiveCodec.hash(modelBytes), fixture.pins.modelManifestSHA256)
            XCTAssertEqual(WorkflowArchiveCodec.hash(gradientBytes), fixture.pins.gradientSHA256)
            XCTAssertEqual(WorkflowArchiveCodec.hash(metricBytes), fixture.pins.metricsSHA256)
            XCTAssertEqual(WorkflowArchiveCodec.hash(try checkedBytes("observations.csv", directory: directory, pins: pins)), fixture.pins.sourceSHA256)
            XCTAssertEqual(WorkflowArchiveCodec.hash(try checkedBytes(fixture.id + "/graph.json", directory: directory, pins: pins)), fixture.pins.modelGraphSHA256)
            XCTAssertEqual(WorkflowArchiveCodec.hash(try checkedBytes("preparation-\(fixture.configuration.validationPercent)/manifest.json", directory: directory, pins: pins)), fixture.pins.preparationManifestSHA256)
            let gradientObject = try XCTUnwrap(JSONSerialization.jsonObject(with: gradientBytes) as? [String: Any])
            let configurationBytes = try JSONSerialization.data(withJSONObject: XCTUnwrap(gradientObject["config"]), options: [.sortedKeys])
            XCTAssertEqual(WorkflowArchiveCodec.hash(configurationBytes), fixture.pins.captureConfigurationSHA256)
            let manifest = try JSONDecoder().decode(TrainingManifest.self, from: modelBytes)
            let modelObject = try XCTUnwrap(JSONSerialization.jsonObject(with: modelBytes) as? [String: Any])
            let metadata = try GradientCaptureCodec.decodeMetadata(JSONSerialization.data(withJSONObject: XCTUnwrap(modelObject["gradientCapture"])))
            let artifact = try GradientCaptureCodec.decodeArtifact(gradientBytes, expectedRunID: fixture.runID, expectedDevice: "cpu", metadata: metadata)
            let metrics = try JSONDecoder().decode([MetricWire].self, from: metricBytes)
            XCTAssertEqual(manifest.status, "completed"); XCTAssertEqual(manifest.runID, fixture.runID)
            XCTAssertEqual(manifest.runtime.python, corpus.runtime.python); XCTAssertEqual(manifest.runtime.torch, corpus.runtime.pytorch)
            XCTAssertEqual(manifest.runtime.device, "cpu"); XCTAssertTrue(manifest.settings.deterministicAlgorithms)
            XCTAssertEqual(manifest.seed, fixture.configuration.seed)
            XCTAssertEqual(manifest.settings.initialEpoch, fixture.configuration.initialEpoch)
            XCTAssertEqual(manifest.settings.epochs, fixture.configuration.epochs)
            XCTAssertEqual(manifest.settings.epochsExecuted, fixture.configuration.trainedEpochs)
            XCTAssertEqual(manifest.settings.optimizer, "Adam"); XCTAssertEqual(manifest.settings.partitionMode, "prepared")
            XCTAssertEqual(manifest.settings.learningRate, fixture.configuration.learningRate)
            XCTAssertEqual(manifest.settings.requestedBatchSize, fixture.configuration.batchSize)
            XCTAssertEqual(artifact.config.maxCaptures, fixture.configuration.maxCaptures)
            XCTAssertEqual(artifact.initialEpoch, manifest.settings.initialEpoch)
            XCTAssertEqual(artifact.finalEpoch, manifest.settings.epochs)
            XCTAssertEqual(Set(manifest.provenance.trainingRowIDs).count, manifest.provenance.trainingSamples)
            XCTAssertEqual(Set(manifest.provenance.validationRowIDs).count, manifest.provenance.validationSamples)
            XCTAssertTrue(Set(manifest.provenance.trainingRowIDs).isDisjoint(with: Set(manifest.provenance.validationRowIDs)))
            XCTAssertEqual(metrics.map(\.epoch), Array(1...manifest.settings.epochs), "Exact continuation includes its retained historical metrics.")
            XCTAssertTrue(metrics.allSatisfy { $0.trainLoss.isFinite && $0.trainLoss >= 0 && $0.validationLoss.isFinite && $0.validationLoss >= 0 && $0.accuracy.isFinite && (0...1).contains($0.accuracy) })
            // Local replay identity uses the pinned actual model manifest, not an invented worker request.
            var input = MeasuredDiagnosticInput(runID: manifest.runID, requestSHA256: fixture.pins.modelManifestSHA256, metrics: metrics.map(\.metric))
            input.validationRows = manifest.provenance.validationSamples
            input.continuation = manifest.lineage.mode != "newTraining"
            try input.installGradients(artifact, expectedWorkerRunID: manifest.runID)
            XCTAssertEqual(input.gradients.count, fixture.expectations.capturedBatches)
            XCTAssertEqual(input.validationRows, fixture.expectations.validationRows)
            XCTAssertTrue(input.gradients.allSatisfy { $0.nonfiniteElements == 0 && $0.missingParameters == 0 })
            let report = try MeasuredDiagnostics.evaluate(input)
            XCTAssertEqual(Set(report.findings.map(\.ruleID)), Set(fixture.expectations.expectedFindings), fixture.id)
            for id in fixture.expectations.withheldAlways {
                XCTAssertEqual(report.checks.first { $0.ruleID == id }?.status, .withheld, fixture.id + " / " + id)
            }
            if fixture.expectations.growth.withheld != nil {
                XCTAssertEqual(report.checks.first { $0.ruleID == "gradient.growth.v1" }?.status, .withheld, fixture.id)
            }
            XCTAssertEqual(try MeasuredDiagnosticReport.decode(JSONEncoder().encode(report)), report)
            XCTAssertThrowsError(try GradientCaptureCodec.decodeArtifact(gradientBytes, expectedRunID: UUID().uuidString, expectedDevice: "cpu", metadata: metadata))
            XCTAssertThrowsError(try GradientCaptureCodec.decodeArtifact(gradientBytes + Data([10]), expectedRunID: manifest.runID, expectedDevice: "cpu", metadata: metadata), "Pinned metadata refuses modified bytes, even valid whitespace.")
            var uncaptured = input
            uncaptured.gradients = [] // Explicitly unavailable capture, without substituting zero measurements.
            let missing = try MeasuredDiagnostics.evaluate(uncaptured)
            for id in ["gradient.growth.v1", "gradient.nonfinite.v1", "gradient.missing.v1", "gradient.zero.v1"] {
                XCTAssertEqual(missing.checks.first { $0.ruleID == id }?.status, .withheld)
            }
            if fixture.id == "exact-continuation" {
                XCTAssertEqual(manifest.lineage.mode, "exactResume"); XCTAssertNotNil(manifest.lineage.parentRunID)
                XCTAssertEqual(artifact.summaries.map(\.epoch), [21, 22, 23])
                XCTAssertEqual(report.checks.first { $0.ruleID == "optimization.plateau.v1" }?.status, .withheld)
            }
            if fixture.id == "insufficient-gradient-window" {
                XCTAssertEqual(artifact.summaries.last?.status, .budgetExhausted)
                XCTAssertEqual(input.gradients.count, 2)
            }
        }
    }
}
