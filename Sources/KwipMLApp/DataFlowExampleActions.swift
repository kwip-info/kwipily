import Foundation
import KwipMLCore

extension DataFlowWorkspaceStore {
    var bundledDataExampleDirectory: URL {
        if let runtime = workspace.runtime {
            return (runtime.kind == .bundled ? runtime.workingDirectoryURL.appendingPathComponent("runner") : runtime.entryURL.deletingLastPathComponent()).appendingPathComponent("dataflow_examples")
        }
        return workspace.projectRoot.appendingPathComponent("runner/dataflow_examples")
    }
    var fittedLessonText: String {
        let name = project.graph.nodes.contains { $0.operatorID == "kwip.data.filterRows" } ? "sensor-lesson.md" : "lesson.md"
        return (try? String(contentsOf: bundledDataExampleDirectory.appendingPathComponent(name), encoding: .utf8)) ?? "This runtime does not include this preparation lesson."
    }
    func loadBundledFittedExample() {
        guard requirePreparationOperation() else {return}
        let placements: [(String, Double, Double)] = [("source",30,190),("columns",250,190),("partition",470,190),("prepare",690,190),("training",910,70),("validation",910,290),("test",910,510)]
        loadBundledPreparation(graphFile: "preparation.json", sourceFile: "observations.csv", name: "Missing values & categorical features", placements: placements)
    }
    func loadBundledSensorExample() {
        guard requirePreparationOperation() else {return}
        let placements: [(String, Double, Double)] = [("source",30,190),("columns",250,190),("quality-filter",470,190),("range-filter",690,190),("model-columns",910,190),("partition",1130,190),("prepare",1350,190),("training",1570,70),("validation",1570,290),("test",1570,510)]
        loadBundledPreparation(graphFile: "sensor-preparation.json", sourceFile: "sensor-readings.csv", name: "Sensor selection & filtering", placements: placements)
    }
    private func loadBundledPreparation(graphFile: String, sourceFile: String, name: String, placements: [(String, Double, Double)]) {
        guard requirePreparationOperation() else {return}
        guard !workspace.isBusy, !isBusy else { return }
        do {
            let directory = bundledDataExampleDirectory
            let graph = try DataFlowCodec.decodeGraph(WorkspaceDocumentReader.read(directory.appendingPathComponent(graphFile)))
            let source = directory.appendingPathComponent(sourceFile)
            guard graph.nodes.allSatisfy({ Self.editableTypes.contains($0.operatorID) }), FileManager.default.fileExists(atPath: source.path) else { throw DataFlowContractError("This runtime's example uses unavailable components or has no observations.") }
            let stableSource = try workspace.sourceAccess.snapshot(to: source.path, directory: workspace.workspaceDirectory.appendingPathComponent("dataflow/example-sources"))
            var next = DataFlowProject(name: name, graph: graph)
            for (id, x, y) in placements { next.editor[id] = DataFlowEditorNode(name: id.replacingOccurrences(of: "-", with: " ").capitalized, x: x, y: y) }
            for node in graph.nodes where node.operatorID == "kwip.data.sourceCSV" { next.sourceBindings[node.id] = stableSource }
            historySessionRevision = UUID()
            let before = project; project = next; history.record(before: before, after: next)
            clearSelection(); sourceHeaders = [:]; previews = [:]; manifestURL = nil
            if let sourceID = graph.nodes.first(where: { $0.operatorID == "kwip.data.sourceCSV" })?.id { inspectSource(id: sourceID) }
            statusMessage = "Loaded bundled observations and preparation example. Prepare data, inspect outputs, then choose model mappings explicitly."
        } catch { statusMessage = error.localizedDescription }
    }
}
