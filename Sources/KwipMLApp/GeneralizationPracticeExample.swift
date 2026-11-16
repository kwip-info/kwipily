import Foundation
import KwipMLCore

/// Public, designed comparison. Reversed validation labels demonstrate a warning
/// pattern and its known cause; this is not proof of memorization or overfitting.
enum GeneralizationPracticeExample {
    static func resourceName(control: Bool) -> String {
        control ? "generalization-consistent" : "generalization-label-shift"
    }
    static func observationsURL(control: Bool = false) throws -> URL {
        guard let url = Bundle.module.url(forResource: resourceName(control: control), withExtension: "csv") else {
            throw DataFlowContractError("The public generalization observations are missing from this build.")
        }
        return url
    }
    static func document(control: Bool = false, projectID: String = UUID().uuidString) throws -> WorkflowDocument {
        var document = try WorkflowCodec.migrateToVersion2(WorkflowExamples.preparedParallelClassifier(projectID: projectID, device: "cpu"))
        document.name = control ? "Consistent-label comparison" : "Deliberate held-out label shift"
        document.nodes.removeAll { $0.address.domain == .model }
        document.edges.removeAll { $0.from.address.domain == .model || $0.to.address.domain == .model }
        document.definitions = []
        func address(_ domain: WorkflowDomain, _ id: String) -> WorkflowNodeAddress { .init(domain: domain, localID: id) }
        document.nodes += [
            .init(address: address(.model, "input"), name: "Two scaled measurements", x: 80, y: 530,
                  payload: .modelPrimitive(type: "input", configurationVersion: 2, config: ["features": 2])),
            .init(address: address(.model, "affine-0"), name: "Learn two class scores", x: 340, y: 530,
                  payload: .modelPrimitive(type: "linear", configurationVersion: 2, config: ["in_features": 2, "out_features": 2, "bias": true])),
            .init(address: address(.model, "output"), name: "Class scores", x: 600, y: 530,
                  payload: .modelPrimitive(type: "output", configurationVersion: 2, config: [:]))
        ]
        for (id, source, target) in [("practice-input-head", "input", "affine-0"), ("practice-head-output", "affine-0", "output")] {
            document.edges.append(.init(id: id, from: .init(address: address(.model, source), port: "out"), to: .init(address: address(.model, target), port: "in"), kind: .tensor))
        }
        document.edges.append(.init(id: "practice-model-source", from: .init(address: address(.model, "output"), port: "definition"), to: .init(address: address(.control, "model"), port: "source"), kind: .modelDefinition))
        for index in document.nodes.indices {
            let node = document.nodes[index]
            if node.address == WorkflowExamples.sourceAddress {
                document.nodes[index].name = control ? "80 public rows; consistent labels" : "80 public rows; validation labels deliberately reversed"
            }
            if node.address == address(.data, "split") {
                document.nodes[index].name = "48 training; 16 validation; 16 reserved future rows"
                document.nodes[index].payload = .dataOperator(operatorID: "kwip.data.partition", config: ["strategy":.string("time"),
                    "timeColumn": .string("observed_at"), "timeEncoding": .string("iso8601Offset"),
                    "trainingEnd": .string("2026-01-01T00:47:00Z"), "validationEnd": .string("2026-01-01T01:03:00Z"),
                    "embargoMilliseconds": 0, "entityColumns": .array([]), "entityPolicy": .string("allowAcrossTime")])
            }
            if case .runSetup(var config) = node.payload {
                config["epochs"] = 20; config["batchSize"] = 48; config["seed"] = 42
                config["learningRate"] = .number(0.03); config["device"] = .string("cpu")
                config["preparationCachePolicy"] = .string("materialize")
                document.nodes[index].name = "20 epochs; same comparison budget"
                document.nodes[index].payload = .runSetup(config: config)
            }
            if case .modelDefinition = node.payload { document.nodes[index].name = "One trainable class-score layer" }
        }
        try WorkflowCodec.validate(document)
        return document
    }
}

extension WorkspaceStore {
    @discardableResult func loadGeneralizationPracticeExample(control: Bool = false) -> Bool {
        guard !isBusy else { return false }
        var rollback: [URL] = []
        var installed = false
        defer {
            if !installed {
                for url in rollback.reversed() { try? FileManager.default.removeItem(at: url) }
                if !rollback.isEmpty { refreshProjects() }
            }
        }
        do {
            let document = try WorkflowTrainRunSetup.migrate(GeneralizationPracticeExample.document(control: control))
            let operators = try workflowDataRegistry()
            _ = try WorkflowCompiler.compile(document, train: WorkflowExamples.trainAddress,
                modelRegistry: registry, dataRegistry: operators,
                contractVersion: max(2, WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema: document.schemaVersion)))
            let source = try GeneralizationPracticeExample.observationsURL(control: control)
            let bytes = try DiagnosticFileIO.read(root: source.deletingLastPathComponent(), parts: [source.lastPathComponent], limit: 10_000_000)
            if hasUnsavedChanges && !saveManagedProject() { return false }
            guard flushWorkflowCanvasBeforeTransition(), flushChildCanvasBeforeTransition(), learningProjects.beforeProjectTransition() else { return false }
            let projectURL = try projectLibrary.documentURL(id: document.projectID)
            let sourceDirectory = workspaceDirectory.appendingPathComponent("example-sources").appendingPathComponent(document.projectID)
            let bindingDirectory = workspaceDirectory.appendingPathComponent("workflow-sources").appendingPathComponent(document.projectID)
            for url in [projectURL, sourceDirectory, bindingDirectory] {
                guard !FileManager.default.fileExists(atPath: url.path) else { throw DataFlowContractError("Example identity already exists. Open a new comparison.") }
            }
            rollback = [projectURL, sourceDirectory, bindingDirectory]
            _ = try projectLibrary.saveSource(.workflow(document))
            let owner = try WorkflowAuthoringStore(library: projectLibrary, projectID: document.projectID, modelRegistry: registry, dataRegistry: operators)
            defer { owner.invalidate() }
            let bindings = try WorkflowPrivateSourceBindings(owner: owner, directory: workspaceDirectory)
            let parts = ["example-sources", document.projectID, GeneralizationPracticeExample.resourceName(control: control) + ".csv"]
            try DiagnosticFileIO.writeAtomic(bytes, root: workspaceDirectory, parts: parts)
            let durableSource = parts.reduce(workspaceDirectory) { $0.appendingPathComponent($1) }
            guard bindings.setPath(durableSource.path, for: WorkflowExamples.sourceAddress, lease: owner.lease(), expectedRevision: bindings.revision) else {
                throw DataFlowContractError(bindings.error ?? "Could not bind the public observations.")
            }
            refreshProjects()
            if let openProjectInWindow {
                let record = try owner.currentProjectRecord()
                installed = true
                openProjectInWindow(record, "Build")
            } else {
                guard beginWorkflowProject(id: document.projectID) else { return false }
                installed = true
            }
            statusMessage = control
                ? "Opened consistent-label comparison · same 20-epoch budget; 16 future rows remain unused."
                : "Opened deliberate validation label shift · known label noise, not proof of overfitting; 16 future rows remain unused."
            return true
        } catch {
            statusMessage = "Comparison could not be opened: \(error.localizedDescription)"
            return false
        }
    }
}
