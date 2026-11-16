import Foundation
import AppKit
import KwipMLCore

extension WorkspaceStore {
    func isManagedLesson(_ record: ProjectRecord) -> Bool {
        record.guidedCourseID != nil || lessonReading.entry(projectID: record.id).map {lessonReading.record($0).savedProjectID==record.id} == true || learningProjects.hasLessonMarker(projectID:record.id)
    }
    var modelProjects: [ProjectRecord] { projects.filter { !isManagedLesson($0) && lessonReading.entry(projectID:$0.id) == nil } }
    var lessonProjects: [ProjectRecord] { projects.filter { isManagedLesson($0) } }
    var projectCatalogIdentity: URL { workspaceDirectory.appendingPathComponent("projects") }
    func projectCatalogRecords() throws -> [ProjectRecord] {
        try RestorationCatalogSnapshot.shared.records(identity: projectCatalogIdentity) { try projectLibrary.listSources() }
    }
    func invalidateProjectCatalogSnapshot() {
        RestorationCatalogSnapshot.shared.invalidate(identity: projectCatalogIdentity)
    }
    func refreshProjects() {
        do {
            projects = try projectCatalogRecords(); externalProjects = try externalFlowCatalog.list()
            let opened=(try? RecentProjectHistory(directory:applicationDirectory).load()) ?? [:]
            recentProjects=RecentProjectCatalog.entries(managed:modelProjects,external:externalProjects,opened:opened)
        }
        catch { statusMessage = "Project library: \(error.localizedDescription)" }
    }
    func permitsManagedProjectSave() -> Bool {
        guard workflowSession?.owner.hasPendingInspectorEdits != true else {
            let message = "Apply or revert the pending configuration edits before saving or closing this project."
            saveError = message; statusMessage = message
            return false
        }
        return true
    }
    @discardableResult func saveManagedProject() -> Bool {
        guard permitsManagedProjectSave() else { return false }
        do {
            if let owner = workflowSession?.owner {
                try owner.saveCanvasVersion()
                let record = try owner.currentProjectRecord()
                activeProjectID = record.id; projectURL = URL(fileURLWithPath: record.path)
                invalidateProjectCatalogSnapshot()
                refreshProjects(); persistWorkspaceView()
                statusMessage = "Saved locally · \(owner.document.name)"
                hasUnsavedChanges = false; saveError = nil; projectCatalogChanged?(); return true
            }
            if let owner = modelAuthoring {
                let record = try projectLibrary.saveDocument(owner.document, id: owner.projectID)
                activeProjectID = record.id; projectURL = URL(fileURLWithPath: record.path)
                invalidateProjectCatalogSnapshot()
                refreshProjects(); persistWorkspaceView()
                statusMessage = "Saved locally · \(modelSourceName)"
                hasUnsavedChanges = false; saveError = nil; projectCatalogChanged?(); return true
            }
            let current = try GraphMigration.toCurrent(graph, registry: registry)
            let record = try projectLibrary.save(current, id: activeProjectID)
            if graph != current { graph = current }
            activeProjectID = record.id; projectURL = URL(fileURLWithPath: record.path)
            invalidateProjectCatalogSnapshot()
            refreshProjects()
            persistWorkspaceView()
            statusMessage = "Saved locally · \(graph.name)"
            hasUnsavedChanges = false; saveError = nil; projectCatalogChanged?(); return true
        } catch { saveError = "Save failed: \(error.localizedDescription)"; statusMessage = saveError!; return false }
    }
    func openManagedProject(_ record: ProjectRecord, section: String? = nil) {
        if let openProjectInWindow { openProjectInWindow(record, section); return }
        guard !isBusy else { return }
        do {
            let opened: Bool
            switch try projectLibrary.loadSource(id: record.id) {
            case .legacyFlat(let graph): opened = beginProject(graph, id: record.id)
            case .blockModel: opened = beginOwnedProject(id: record.id)
            case .workflow: opened = beginWorkflowProject(id: record.id)
            }
            if opened {
                if isLearningLessonActive { activeSection = "Learn" }
                else if let section { activeSection = section }
            }
        }
        catch { statusMessage = "Open failed: \(error.localizedDescription)" }
    }
    func setExecution(_ update: (inout ExecutionSettings) -> Void) {
        guard workflowSession == nil else { statusMessage = "Configure execution on the workflow components."; return }
        if let owner = modelAuthoring {
            reportModelEdit(owner.editMetadata({ _, execution in
                var settings = execution ?? ExecutionSettings(); update(&settings); execution = settings
            }, lease: owner.lease()))
            return
        }
        edit { document in var settings = document.execution ?? ExecutionSettings(); update(&settings); document.execution = settings }
    }
    func renameProject(_ value: String) {
        if let owner = workflowSession?.owner { reportWorkflowEdit(owner.rename(String(value.prefix(80)), lease: owner.lease())); return }
        if let owner = modelAuthoring {
            reportModelEdit(owner.editMetadata({ name, _ in name = String(value.prefix(80)) }, lease: owner.lease()))
            return
        }
        edit { $0.name = String(value.prefix(80)) }
    }
    func rememberLastRun(_ path: String) {
        guard let id = activeProjectID else { return }
        var paths = UserDefaults.standard.dictionary(forKey: "lastRunByProject") as? [String: String] ?? [:]
        paths[id] = path; UserDefaults.standard.set(paths, forKey: "lastRunByProject")
    }
    func restoreLastRun() {
        guard !isBusy else { return }
        gradientCapture.reset()
        lastCompleted = false; outputDirectory = nil; resultSignature = nil
        generatedCode = ""; metrics = []; lastRunSourceHash = nil
        guard let id = activeProjectID, let paths = UserDefaults.standard.dictionary(forKey: "lastRunByProject") as? [String: String], let path = paths[id] else { return }
        let directory = URL(fileURLWithPath: path)
        // Validate every required artifact before publishing any completed-run state.
        func regularNonempty(_ name: String) -> Bool {
            guard let values = try? directory.appendingPathComponent(name).resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]) else { return false }
            return values.isRegularFile == true && values.isSymbolicLink != true && (values.fileSize ?? 0) > 0
        }
        guard ["manifest.json", "checkpoint.pt", "model.py", "metrics.json"].allSatisfy(regularNonempty),
              let data = try? DiagnosticFileIO.read(root: directory, parts: ["manifest.json"], limit: 20_000_000),
              let manifest = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              manifest["status"] as? String == "completed",
              let graphObject = manifest["graph"],
              let graphData = try? JSONSerialization.data(withJSONObject: graphObject),
              let snapshot = try? JSONDecoder().decode(GraphDocument.self, from: graphData),
              [1, 2].contains(snapshot.schemaVersion),
              GraphValidator.validate(snapshot, registry: registry).isValid,
              let signature = semanticSignature(snapshot),
              let code = try? String(contentsOf: directory.appendingPathComponent("model.py"), encoding: .utf8),
              !code.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let metricData = try? Data(contentsOf: directory.appendingPathComponent("metrics.json")),
              let rows = try? JSONDecoder().decode([RestoredEpoch].self, from: metricData),
              !rows.isEmpty,
              rows.allSatisfy({ $0.epoch > 0 && $0.trainLoss.isFinite && $0.trainLoss >= 0 && $0.validationLoss.isFinite && $0.validationLoss >= 0 && $0.accuracy.isFinite && (0...1).contains($0.accuracy) }) else {
            eventLog = ["Previous run is incomplete or unreadable. Run the model again to create verified artifacts."]
            return
        }
        lastRunRecipeVersion = manifest["recipeVersion"] as? Int ?? 1
        outputDirectory = directory; lastCompleted = true; resultSignature = signature
        lastRunSourceHash = (manifest["provenance"] as? [String: Any])?["sourceHash"] as? String
        generatedCode = code
        resolvedDevice = (manifest["runtime"] as? [String: Any])?["device"] as? String ?? "Unknown"
        metrics = rows.map { EpochMetric(epoch: $0.epoch, trainLoss: $0.trainLoss, validationLoss: $0.validationLoss, accuracy: $0.accuracy) }
        gradientCapture.restore(directory: directory, manifestData: data)
        refreshVerifiedCheckpointParent()
        eventLog = ["Restored last completed run from its manifest. Tensor capture requires a new run."]
    }
    func startCourse() {
        _ = loadGuidedWorkflowCourse()
    }
    func advanceCourse() {
        guard let host=workflowSession,let progress=host.owner.document.guidedBuild else {return}
        _ = host.advanceGuidedCourse(expected:progress)
    }
}

private struct RestoredEpoch: Decodable {
    let epoch: Int
    let trainLoss: Double
    let validationLoss: Double
    let accuracy: Double
}


extension WorkspaceStore {
    func openRecentProject(_ entry:RecentProjectEntry,section:String="Build") {
        switch entry.source {
        case .managed(let record):openManagedProject(record,section:section)
        case .external(let record):openExternalProjectInWindow?(record)
        }
    }
}
