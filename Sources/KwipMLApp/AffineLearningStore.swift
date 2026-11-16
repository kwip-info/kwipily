import Combine
import CryptoKit
import Foundation
import KwipMLCore

@MainActor final class AffineLearningStore: ObservableObject {
  @Published private(set) var draft: AffineLearningDraft?
  @Published private(set) var statusMessage = "Build the requested function and enter any explicit parameters."
  @Published private(set) var latestGrade: LearningGrade?
  @Published private(set) var attempts: [LearningAttempt] = []
  @Published private(set) var resultSaved = false
  @Published private(set) var isRestoreFailure = false
  @Published private(set) var parameterRevision = UUID()
  @Published var pendingNumericEdits: Set<String> = []
  func setNumericEditPending(id: String, pending: Bool) {
    if pending { pendingNumericEdits.insert(id) } else { pendingNumericEdits.remove(id) }
  }
  private weak var workspace: WorkspaceStore?
  private let measuredSession = MeasuredLearningSession()
  private var subscriptions = Set<AnyCancellable>()
  private var receipt: AffineLearningReceipt?
  var latestAttempt: LearningAttempt? { receipt?.attempt }
  /// Only a verified, current formative receipt may expose its numerical fixtures.
  /// Switching the current mode never changes the mode recorded by an attempt.
  var formativeProjection: AffineFormativeProjection? {
    guard mode != .assessment, latestResultIsCurrent, let receipt,
      receipt.attempt.mode != .assessment, let evidence = receipt.evidence,
      evidence.submissionDigest == receipt.submissionSHA256,
      let identity = try? receipt.resolvedLessonIdentity(),
      let recordedPlan = try? receipt.replayPlan(),
      (try? LearningAssessor.evaluate(recordedPlan, evidence: evidence)) == receipt.attempt.grade
    else { return nil }
    return AffineFormativeProjection.make(
      plan: recordedPlan, evidence: evidence, recordedMode: receipt.attempt.mode,
      lessonIdentity: identity)
  }
  private var directory: URL { workspace!.workspaceDirectory.appendingPathComponent("learning") }
  private var progress: LearningProgressStore { LearningProgressStore(directory: directory) }
  init(workspace: WorkspaceStore) {
    self.workspace = workspace
    workspace.learningProjects.attachAffine(self)
    workspace.$legacyGraph.sink { [weak self, weak workspace] graph in
      guard let self, let workspace, workspace.learningProjects.owns(self), var draft = self.draft,
        draft.projectID == workspace.activeProjectID,
        workspace.affineLessonProjectID == draft.projectID
      else { return }
      let configurationChanged =
        draft.graph.nodes.count != graph.nodes.count
        || graph.nodes.contains { node in
          guard let previous = draft.graph.nodes.first(where: { $0.id == node.id }) else {
            return true
          }
          return previous.type != node.type || previous.config != node.config
        }
      if configurationChanged {
        self.parameterRevision = UUID()
        self.pendingNumericEdits = []
      }
      draft.graph = graph
      self.draft = draft
      self.persistDraft()
    }.store(in: &subscriptions)
  }
  var isActive: Bool { draft != nil && draft?.projectID == workspace?.activeProjectID && workspace?.learningProjects.owns(self) == true }
  var isBusy: Bool { workspace?.isBusy ?? false }
  var mode: LearningMode { draft?.mode ?? .guided }
  var lessonTask: AffineLessonTask? {
    guard let identity = try? draft?.resolvedLessonIdentity() else { return nil }
    return try? AffineLessonTasks.publicTask(identity: identity)
  }
  private var plan: LearningAssessmentPlan? {
    guard let identity = try? draft?.resolvedLessonIdentity() else { return nil }
    return try? AffineLessonTasks.plan(
      identity: identity, runtimeVersion: workspace?.runtimeContractVersion ?? "unavailable",
      fixtureSeed: 1)
  }
  var publicTask: LearningPublicTask? { isActive ? plan?.publicTask(mode: mode) : nil }
  var linearNodes: [GraphNode] { workspace?.graph.nodes.filter { $0.type == "linear" } ?? [] }
  var parameterIssues: [String] {
    do {
      _ = try currentSubmission()
      return []
    } catch { return [error.localizedDescription] }
  }
  var lessonRuntimeIssue: String? {
    guard let identity = try? draft?.resolvedLessonIdentity() else { return nil }
    if [.tanh, .relu].contains(identity) {
      return workspace?.runtimeExecutionContract == nil || workspace?.runtimeContractVersion == nil ? "This \(identity == .relu ? "ReLU" : "Tanh") lesson requires a verified runtime execution contract. Refresh runtime status or select a compatible runtime." : nil
    }
    if identity == .sigmoid {
      return workspace?.sigmoidRuntimeCapability == nil ? "This Sigmoid lesson requires verified Sigmoid runtime support. Refresh runtime status or select a compatible runtime." : nil
    }
    guard identity.geluApproximateTanh != nil else { return nil }
    return workspace?.geluRuntimeCapability == nil ? "This GELU lesson requires verified GELU runtime support. Refresh runtime status or select a compatible runtime." : nil
  }
  var canSubmit: Bool {
    isActive && workspace?.runtime != nil && !isBusy && pendingNumericEdits.isEmpty
      && parameterIssues.isEmpty && lessonRuntimeIssue == nil
  }
  var hasCurrentCompletion: Bool {
    guard let plan else { return false }
    return (try? progress.hasCurrentCompletion(for: plan)) ?? false
  }
  var latestAssessmentIsCurrent: Bool {
    guard let attempt = latestAttempt, let plan else { return false }
    return attempt.contentVersion == plan.contentVersion
      && attempt.assessmentVersion == plan.assessmentVersion
  }
  var latestResultIsCurrent: Bool {
    guard let receipt, let current = try? currentSubmission(),
      let recordedIdentity = try? receipt.resolvedLessonIdentity(),
      let currentIdentity = try? draft?.resolvedLessonIdentity(),
      recordedIdentity == currentIdentity else { return false }
    guard let currentIdentity = signature(current),
      let receiptIdentity = signature(receipt.submission)
    else { return false }
    return currentIdentity == receiptIdentity
  }
  private struct ConstructionIdentity: Encodable {
    let graph: Data
    let parameters: [String: LinearSubmittedParameters]
  }
  private func signature(_ submission: AffineLearningSubmission) -> Data? {
    guard let registry = workspace?.registry,
      let graph = try? GraphSemantics.canonicalData(submission.graph, registry: registry)
    else { return nil }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    return try? encoder.encode(
      ConstructionIdentity(graph: graph, parameters: submission.parameters))
  }
  private func currentSubmission() throws -> AffineLearningSubmission {
    guard var draft, isActive, let workspace else {
      throw LearningContractError("Open a lesson project first.")
    }
    draft.graph = workspace.graph
    return try draft.submission(registry: workspace.registry)
  }
  var startInWindow: ((AffineLessonIdentity, LearningMode) -> Void)?
  func start(identity: AffineLessonIdentity = .composition, mode: LearningMode = .guided) {
    if let startInWindow { startInWindow(identity, mode); return }
    guard let workspace, !workspace.isBusy, workspace.learningProjects.owns(self) else { return }
    guard let task = try? AffineLessonTasks.publicTask(identity: identity) else {
      statusMessage = "This lesson task is unsupported. The current project is unchanged."
      return
    }
    guard (workspace.startsIndependentLesson && workspace.activeProjectID == nil) || workspace.saveManagedProject() else { return }
    let previous = workspace.activeProjectID
    let graph = GraphDocument(
      name: "Learn · \(task.title)",
      nodes: [
        GraphNode(
          id: "input", type: "input", name: identity == .backwardChain ? "Upstream gradient" : "Input", x: 80, y: 160, config: ["features": 2]),
        GraphNode(id: "output", type: "output", name: identity == .backwardChain ? "Input gradient" : "Result", x: 620, y: 160),
      ], edges: [])
    guard workspace.beginProject(graph), let id = workspace.activeProjectID else { return }
    isRestoreFailure = false
    draft = AffineLearningDraft(
      projectID: id, graph: graph, mode: mode, returnProjectID: previous,
      lessonIdentity: identity)
    workspace.learningProjects.activateAffine(projectID: id)
    workspace.activeSection = "Learn"
    parameterRevision = UUID()
    pendingNumericEdits = []
    receipt = nil
    latestGrade = nil
    resultSaved = false
    statusMessage = "Build the requested function and enter any explicit parameters."
    save()
  }
  func reopenCurrentProject() {
    guard let workspace, workspace.learningProjects.owns(self) else { return }
    workspace.learningProjects.restoreCurrentProject()
  }
  func clearProjectContext() {
    draft = nil; receipt = nil; latestGrade = nil; resultSaved = false
    isRestoreFailure = false; parameterRevision = UUID(); pendingNumericEdits = []
  }
  func persistForProjectTransition() -> Bool {
    guard isActive else { return true }
    draft?.graph = workspace!.graph
    return persistDraft()
  }
  func restoreProjectContext() {
    guard let workspace, !workspace.isBusy, workspace.learningProjects.owns(self) else { return }
    guard draft?.projectID != workspace.activeProjectID else { return }
    clearProjectContext()
    guard let id = workspace.activeProjectID else { return }
    let url = draftURL(id)
    guard FileManager.default.fileExists(atPath: url.path) else { return }
    // An unreadable lesson still blocks ordinary training; its explicit parameters must never be silently ignored.
    if workspace.activeSection == "Build" { workspace.activeSection = "Learn" }
    do {
      let value = try AffineLearningCodec.decodeDraft(read(url, limit: 2_000_000))
      guard value.projectID == id else {
        throw LearningContractError("Lesson project identity mismatch.")
      }
      draft = value
      draft?.graph = workspace.graph
      if workspace.activeSection == "Build" { workspace.activeSection = "Learn" }
      attempts = try progress.attempts()
      restoreReceipt(projectID: id)
    } catch {
      isRestoreFailure = true
      statusMessage = "Lesson could not be restored: \(error.localizedDescription)"
    }
  }
  private func restoreReceipt(projectID: String) {
    var remaining = 16_000_000
    func accept(_ data: Data) -> Bool {
      guard let candidate = try? JSONDecoder().decode(AffineLearningReceipt.self, from: data),
        [1, 2].contains(candidate.schemaVersion), candidate.projectID == projectID,
        let submitted = try? AffineLearningCodec.encodeSubmission(candidate.submission),
        candidate.submissionSHA256 == hash(submitted),
        attempts.contains(where: { $0 == candidate.attempt })
      else { return false }
      guard
        let expected = try? candidate.replayPlan(),
        let recordedIdentity = try? candidate.resolvedLessonIdentity(),
        let currentIdentity = try? draft?.resolvedLessonIdentity(),
        recordedIdentity == currentIdentity,
        candidate.runtime == expected.runtime,
        candidate.fixtureDigest == expected.fixtureDigest,
        candidate.attempt.objectiveID == expected.objectiveID,
        candidate.attempt.contentVersion == expected.contentVersion,
        candidate.attempt.assessmentVersion == expected.assessmentVersion,
        candidate.attempt.grade.fixtureDigest.map({ $0 == candidate.fixtureDigest }) ?? true
      else { return false }
      if let evidence = candidate.evidence {
        guard evidence.runtime == candidate.runtime,
          evidence.submissionDigest == candidate.submissionSHA256,
          evidence.fixtureDigest == candidate.fixtureDigest,
          (try? LearningAssessor.evaluate(expected, evidence: evidence)) == candidate.attempt.grade
        else { return false }
      } else if candidate.attempt.grade.status == .passed
        || candidate.attempt.grade.status == .failed
      {
        return false
      }
      receipt = candidate
      latestGrade = candidate.attempt.grade
      resultSaved = true
      return true
    }
    // Progress is the durable index; latest.json is only an optional convenience copy.
    for attempt in attempts.reversed().prefix(100) {
      let url = directory.appendingPathComponent("attempts/\(attempt.id).json")
      guard remaining > 0, let data = try? read(url, limit: min(4_000_000, remaining)) else {
        continue
      }
      remaining -= data.count
      if accept(data) { return }
    }
    let latest = directory.appendingPathComponent("projects/\(projectID)/latest.json")
    if let data = try? read(latest, limit: 4_000_000),
      let candidate = try? JSONDecoder().decode(AffineLearningReceipt.self, from: data),
      UUID(uuidString: candidate.attempt.id) != nil,
      let immutable = try? read(
        directory.appendingPathComponent("attempts/\(candidate.attempt.id).json"), limit: 4_000_000),
      accept(immutable)
    {
      return
    }
    if !attempts.isEmpty {
      statusMessage =
        "No verified receipt for this project was found within the bounded recent-attempt index. Saved progress is preserved."
    }
  }
  private func draftURL(_ id: String) -> URL {
    directory.appendingPathComponent("projects/\(id)/draft.json")
  }
  private func read(_ url: URL, limit: Int) throws -> Data {
    let info = try url.resourceValues(forKeys: [
      .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
    ])
    guard info.isRegularFile == true, info.isSymbolicLink != true,
      (info.fileSize ?? Int.max) <= limit
    else { throw LearningContractError("Lesson file is not a bounded regular file.") }
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    let data = try handle.read(upToCount: limit + 1) ?? Data()
    guard data.count <= limit else { throw LearningContractError("Lesson file exceeds bounds.") }
    return data
  }
  @discardableResult private func persistDraft() -> Bool {
    guard let draft else { return true }
    do {
      let url = draftURL(draft.projectID)
      try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
      try AffineLearningCodec.encodeDraft(draft).write(to: url, options: .atomic)
      return true
    } catch { statusMessage = "Lesson save failed: \(error.localizedDescription)";return false }
  }
  @discardableResult func save() -> Bool {
    guard isActive, let workspace else { return false }
    draft?.graph = workspace.graph
    guard persistDraft() else { workspace.statusMessage = statusMessage;return false }
    return workspace.saveManagedProject()
  }
  func setMode(_ value: LearningMode) {
    guard isActive, !isBusy else { return }
    draft?.mode = value
    persistDraft()
  }
  func setNotes(_ value: String) {
    guard isActive, !isBusy else { return }
    draft?.notes = String(value.prefix(8000))
    persistDraft()
  }
  func revealHint(_ id: String) {
    guard isActive, !isBusy, mode != .assessment, !id.isEmpty, id.count <= 128,
      (draft?.hintIDs.count ?? 16) < 16, draft?.hintIDs.contains(id) == false
    else { return }
    draft?.hintIDs.append(id)
    persistDraft()
  }
  func parameters(for id: String) -> LinearParameterValues? { draft?.parameters[id] }
  func resetParameters(nodeID: String) {
    guard isActive, !isBusy, let node = linearNodes.first(where: { $0.id == nodeID }),
      let input = node.config["in_features"]?.doubleValue,
      let output = node.config["out_features"]?.doubleValue,
      input.isFinite, output.isFinite, (1...512).contains(input), (1...512).contains(output),
      input.rounded() == input, output.rounded() == output
    else { return }
    do {
      draft?.parameters[nodeID] = try .zeros(input: Int(input), output: Int(output))
      parameterRevision = UUID()
      persistDraft()
    } catch { statusMessage = error.localizedDescription }
  }
  func setWeight(nodeID: String, row: Int, column: Int, value: Double, revision: UUID? = nil)
    -> Bool
  {
    update(nodeID, revision: revision) { parameters in
      guard row >= 0, column >= 0, row < parameters.weight.shape[0],
        column < parameters.weight.shape[1], value.isFinite, abs(value) <= 1_000_000
      else { return false }
      parameters.weight.values[row * parameters.weight.shape[1] + column] = value
      return true
    }
  }
  func setBias(nodeID: String, index: Int, value: Double, revision: UUID? = nil) -> Bool {
    guard linearNodes.first(where: { $0.id == nodeID })?.config["bias"]?.boolValue != false else {
      return false
    }
    return update(nodeID, revision: revision) { parameters in
      guard index >= 0, index < parameters.bias.values.count, value.isFinite,
        abs(value) <= 1_000_000
      else { return false }
      parameters.bias.values[index] = value
      return true
    }
  }
  private func update(_ id: String, revision: UUID?, _ edit: (inout LinearParameterValues) -> Bool)
    -> Bool
  {
    guard isActive, !isBusy, revision == nil || revision == parameterRevision,
      workspace?.selectedNodeID == id, let node = linearNodes.first(where: { $0.id == id }),
      var values = draft?.parameters[id],
      let input = node.config["in_features"]?.integerValue,
      let output = node.config["out_features"]?.integerValue,
      (1...512).contains(input), (1...512).contains(output),
      (try? values.validate(input: input, output: output))
        != nil, edit(&values)
    else { return false }
    draft?.parameters[id] = values
    persistDraft()
    return true
  }
  func cancel() { measuredSession.cancel() }
  func submit() {
    if let issue = lessonRuntimeIssue { statusMessage = issue; return }
    guard canSubmit, let workspace, workspace.runtime != nil, let draft else {
      statusMessage = "Connect a valid graph, enter parameters, and ensure the Python runtime is available."
      return
    }
    do {
      let submission = try currentSubmission()
      let data = try AffineLearningCodec.encodeSubmission(submission)
      let identity = try draft.resolvedLessonIdentity()
      guard save() else { return }
      resultSaved = false
      receipt = nil
      latestGrade = nil
      statusMessage = "Evaluating measured outputs…"
      let accepted = measuredSession.start(workspace:workspace,projectID:draft.projectID,submission:data,runtimeVersionKey:"pytorch",makePlan:{ version,seed in
        try AffineLessonTasks.plan(identity:identity,runtimeVersion:version,fixtureSeed:seed)
      },onComplete:{ [weak self,weak workspace] result in
        guard let self,let workspace,workspace.activeProjectID == draft.projectID,
              self.draft?.projectID == draft.projectID,workspace.learningProjects.owns(self) else {return}
        let grade=result.grade
        self.latestGrade=grade
        self.statusMessage=grade.summary
        // Cancellation/discovery/execution failures are outcomes, not measured receipts.
        guard let plan=result.plan,let evidence=result.evidence,[.passed,.failed].contains(grade.status) else {return}
        let attempt = LearningAttempt(
          objectiveID:plan.objectiveID,contentVersion:plan.contentVersion,
          assessmentVersion:plan.assessmentVersion,mode:draft.mode,hintIDs:draft.hintIDs,grade:grade)
        let receipt = AffineLearningReceipt(
          attempt:attempt,projectID:draft.projectID,submission:submission,
          submissionSHA256:self.hash(data),runtime:plan.runtime,fixtureSeed:plan.fixtureSeed,
          fixtureDigest:plan.fixtureDigest,notes:draft.notes,evidence:evidence,lessonIdentity:identity)
        self.receipt=receipt
          do {
            let encoded = try JSONEncoder().encode(receipt)
            guard encoded.count <= 4_000_000 else {
              throw LearningContractError("Receipt exceeds bounds.")
            }
            let folder = self.directory.appendingPathComponent("attempts")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try encoded.write(
              to: folder.appendingPathComponent(attempt.id + ".json"), options: .withoutOverwriting)
            try self.progress.append(attempt)
            self.resultSaved = true
            self.attempts = (try? self.progress.attempts()) ?? (self.attempts + [attempt])
            self.statusMessage = grade.summary
            do {
              try encoded.write(
                to: self.directory.appendingPathComponent(
                  "projects/\(draft.projectID)/latest.json"), options: .atomic)
            } catch {
              self.statusMessage =
                "Attempt saved; latest receipt index could not be updated: \(error.localizedDescription). Reopening recovers verified recent receipts."
            }
          } catch { self.statusMessage = "Evaluated but not saved: \(error.localizedDescription)" }
      })
      if !accepted {statusMessage=measuredSession.startFailureMessage ?? "The lesson assessment could not start; wait for the active operation and retry."}
    } catch {statusMessage=error.localizedDescription}
  }
    var openProjectBrowser: (() -> Void)?
    var leaveActionTitle: String { draft?.returnProjectID == nil ? "Projects" : "Return to project" }
  func leave(returnToPreviousProject: Bool) {
    guard !isBusy, let workspace, workspace.learningProjects.owns(self) else { return }
    guard returnToPreviousProject else { return }
    guard let previous = draft?.returnProjectID else {
      guard let openProjectBrowser, save() else { return }
      openProjectBrowser()
      statusMessage = "Projects opened. This saved lesson stays in its own tab."
      return
    }
    do {
      let document = try workspace.projectLibrary.loadDocument(id: previous)
      guard save() else { return }
      let opened: Bool
      switch document {
      case .legacyFlat(let graph): opened = workspace.beginProject(graph, id: previous)
      case .blockModel: opened = workspace.beginOwnedProject(id: previous)
      }
      guard opened else {
        statusMessage = workspace.statusMessage + " This lesson remains active."
        return
      }
      pendingNumericEdits = []
    } catch {
      statusMessage =
        "Could not open the return project: \(error.localizedDescription). Open Projects or create a new flow; this lesson remains protected."
    }
  }

  private func hash(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
}
