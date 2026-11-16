import Foundation
import Combine
import CryptoKit
import KwipMLCore

/// Draft ownership and verified measured results for the normalization lesson.
@MainActor final class NormalizationLearningStore:ObservableObject {
    @Published private(set) var draft:NormalizationLearningDraft?
    @Published private(set) var editor:[String:DataFlowEditorNode]=[:]
    @Published private(set) var flowEditor:DataFlowWorkspaceStore?
    @Published private(set) var statusMessage="Build a training-only normalization flow. Select the three outputs to check your construction."
    @Published private(set) var isRestoreFailure=false
    @Published private(set) var latestGrade:LearningGrade?
    @Published private(set) var attempts:[LearningAttempt]=[]
    @Published private(set) var resultSaved=false
    private var receipt:NormalizationLearningReceipt?
    private var verifiedAttemptIDs:Set<String>=[]
    private let session=MeasuredLearningSession()
    private var pendingObservation:AnyCancellable?
    private var directory:URL {workspace!.workspaceDirectory.appendingPathComponent("learning")}
    private var progress:LearningProgressStore {LearningProgressStore(directory:directory)}
    var latestAttempt:LearningAttempt? {receipt?.attempt}
    // Kept for historical draft and receipt reading; current preparation is Preprocess.
    var canSubmit:Bool {false}
    var latestAssessmentIsCurrent:Bool {latestAttempt?.assessmentVersion == NormalizationLessonTask.currentVersion && latestAttempt?.contentVersion == ContractVersion()}
    var latestResultIsCurrent:Bool {
        guard isActive,let receipt,let draft,receipt.attempt.assessmentVersion == draft.assessmentVersion,receipt.attempt.contentVersion == draft.contentVersion,
              let current=try? draft.executableSubmission().semanticDigest(),let recorded=try? receipt.submission.semanticDigest() else {return false}
        return current == recorded
    }
    var formativeProjection:NormalizationFormativeProjection? {
        guard latestResultIsCurrent,let draft,let receipt else {return nil}
        return receipt.formativeProjection(expectedProjectID:draft.projectID,currentSubmission:draft.submission,currentMode:mode)
    }
    var hasCurrentCompletion:Bool {
        isActive && attempts.contains { verifiedAttemptIDs.contains($0.id) && $0.objectiveID == NormalizationLessonTask.objectiveID && $0.contentVersion == ContractVersion() && $0.assessmentVersion == NormalizationLessonTask.currentVersion && $0.mode == .assessment && $0.grade.status == .passed }
    }
    private weak var workspace:WorkspaceStore?
    private var editorObservation:AnyCancellable?
    init(workspace:WorkspaceStore) {self.workspace=workspace;workspace.learningProjects.attachNormalization(self)}
    var isActive:Bool {draft != nil && draft?.projectID == workspace?.activeProjectID && workspace?.learningProjects.owns(self) == true}
    var isBusy:Bool {workspace?.isBusy ?? false}
    var mode:LearningMode {draft?.mode ?? .guided}
    var lessonProject:DataFlowProject? {
        guard let draft else {return nil}
        return DataFlowProject(name:"Training-only standardization",graph:draft.submission.graph,editor:editor)
    }
    var constructionIssues:[String] {
        guard let draft else {return ["Open a normalization lesson first."]}
        var issues:[String]=[]
        if flowEditor?.hasPendingNumericEdits == true {issues.append("Apply or correct pending numeric edits in the canvas inspector.")}
        if workspace?.runtime == nil {issues.append("Configure an available Python runtime before checking the flow.")}
        do {_ = try draft.executableSubmission()} catch {issues.append(error.localizedDescription)}
        issues += flowEditor?.validation.issues.map(\.message) ?? ["Lesson editor is unavailable."]
        return issues
    }
    var startInWindow: ((LearningMode) -> Void)?
    func start(mode:LearningMode = .guided) {
        if let startInWindow { startInWindow(mode); return }
        guard let workspace,!workspace.isBusy,workspace.learningProjects.owns(self), ((workspace.startsIndependentLesson && workspace.activeProjectID == nil) || workspace.saveManagedProject()) else {return}
        let previous=workspace.activeProjectID
        guard workspace.beginProject(GraphDocument(name:"Learn · Training-only standardization")),let id=workspace.activeProjectID else {return}
        let source=DataFlowNode(id:"source",operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])
        draft=NormalizationLearningDraft(projectID:id,submission:NormalizationLearningSubmission(graph:DataFlowDocument(nodes:[source]),outputs:[:]),mode:mode,returnProjectID:previous)
        editor=["source":DataFlowEditorNode(name:"Fixture source",x:80,y:120)]
        isRestoreFailure=false
        workspace.learningProjects.activateNormalization(projectID:id)
        workspace.activeSection="Learn"
        attachEditor()
        statusMessage="Configure and connect your flow, then select the three outputs."
        _ = save()
        do {attempts=try progress.attempts();restoreReceipt(projectID:id)}
        catch {statusMessage="Construction created; progress could not be read: " + error.localizedDescription}
    }
    private func attachEditor() {
        pendingObservation?.cancel();pendingObservation=nil
        editorObservation?.cancel();editorObservation=nil;flowEditor=nil
        guard let workspace,let draft,let project=lessonProject,workspace.activeProjectID == draft.projectID else {return}
        let value=DataFlowWorkspaceStore(workspace:workspace,lessonProject:project)
        let projectID=draft.projectID
        flowEditor=value
        pendingObservation=value.$pendingNumericEdits.dropFirst().sink { [weak self,weak value] _ in
            guard let self,let value,self.flowEditor === value,self.draft?.projectID == projectID else {return}
            self.objectWillChange.send()
        }
        editorObservation=value.$project.dropFirst().sink { [weak self,weak value] updated in
            guard let self,let value,self.flowEditor === value,self.draft?.projectID == projectID,self.workspace?.activeProjectID == projectID else {return}
            // Published values arrive before the editor property changes; use this value directly.
            _ = self.updateConstruction(graph:updated.graph,editor:updated.editor)
        }
    }
    @discardableResult func updateConstruction(graph:DataFlowDocument,editor:[String:DataFlowEditorNode])->Bool {
        guard isActive,!isBusy,var candidate=draft else {return false}
        guard candidate.submission.graph != graph || self.editor != editor else {return true}
        candidate.submission.graph=graph
        do {try NormalizationLessonDocument(draft:candidate,editor:editor).validate()}
        catch {statusMessage="Construction could not be accepted: " + error.localizedDescription;return false}
        draft=candidate;self.editor=editor
        return persistDraft()
    }
    @discardableResult func selectOutput(role:String,endpoint:DataFlowEndpoint?)->Bool {
        guard isActive,!isBusy,["state","training","validation"].contains(role),var candidate=draft else {return false}
        candidate.submission.outputs[role]=endpoint
        do {try candidate.validate()} catch {statusMessage=error.localizedDescription;return false}
        draft=candidate
        return persistDraft()
    }
    func setMode(_ value:LearningMode) {guard isActive,!isBusy else {return};draft?.mode=value;_ = persistDraft()}
    func setNotes(_ value:String) {guard isActive,!isBusy else {return};draft?.notes=String(value.prefix(8000));_ = persistDraft()}
    func revealHint(_ id:String) {
        guard isActive,!isBusy,mode != .assessment,!id.isEmpty,id.count<=128,(draft?.hintIDs.count ?? 16)<16,draft?.hintIDs.contains(id)==false else {return}
        draft?.hintIDs.append(id);_ = persistDraft()
    }
    @discardableResult func save()->Bool {
        guard isActive,let workspace else {return false}
        guard persistDraft() else {workspace.statusMessage=statusMessage;return false}
        // Recents derive from the managed document timestamp; this also persists the workspace view.
        guard workspace.saveManagedProject() else {statusMessage=workspace.statusMessage;return false}
        statusMessage="Lesson construction saved."
        return true
    }
    func persistForProjectTransition()->Bool {guard isActive else {return true};return save()}
    func clearProjectContext() {
        pendingObservation?.cancel();pendingObservation=nil
        editorObservation?.cancel();editorObservation=nil;flowEditor=nil;draft=nil;editor=[:];isRestoreFailure=false;receipt=nil;latestGrade=nil;resultSaved=false;attempts=[];verifiedAttemptIDs=[]
    }
    func restoreProjectContext() {
        guard let workspace,!workspace.isBusy,workspace.learningProjects.owns(self),let id=workspace.activeProjectID else {return}
        guard draft?.projectID != id else {return}
        clearProjectContext()
        do {
            let value=try NormalizationLessonDocument.decode(read(markerURL(id)))
            guard value.draft.projectID == id else {throw LearningContractError("Normalization project identity differs from its folder.")}
            draft=value.draft;editor=value.editor
            workspace.activeSection="Learn"
            attachEditor()
            statusMessage="Lesson construction restored."
            do {attempts=try progress.attempts();restoreReceipt(projectID:id)}
            catch {statusMessage="Construction restored; progress could not be read: " + error.localizedDescription}
        } catch {
            isRestoreFailure=true;statusMessage="Normalization lesson could not be restored: " + error.localizedDescription
        }
    }
    func cancel() {session.cancel()}
    func submit() {
        guard canSubmit else {statusMessage="This historical normalization lesson is read-only. Use the current Preprocess practice.";return}
        guard canSubmit,let workspace,let draft else {statusMessage="Finish all connections, output selections and numeric edits before checking the flow.";return}
        do {
            let submission=try draft.executableSubmission(),data=try NormalizationLearningCodec.encodeSubmission(submission)
            guard save() else {return}
            let previous=(receipt,latestGrade,resultSaved)
            receipt=nil;latestGrade=nil;resultSaved=false;statusMessage="Evaluating measured outputs…"
            let accepted=session.start(workspace:workspace,projectID:draft.projectID,submission:data,runtimeVersionKey:"numpy",makePlan:{version,seed in
                try NormalizationLessonTask.plan(runtimeVersion:version,seed:seed,assessmentVersion:draft.assessmentVersion,contentVersion:draft.contentVersion)
            },onComplete:{[weak self] result in
                guard let self,self.isActive,self.draft?.projectID == draft.projectID,result.projectID == draft.projectID else {return}
                self.latestGrade=result.grade;self.statusMessage=result.grade.summary
                guard [.passed,.failed].contains(result.grade.status),let plan=result.plan,let evidence=result.evidence else {return}
                let attempt=LearningAttempt(objectiveID:plan.objectiveID,contentVersion:plan.contentVersion,assessmentVersion:plan.assessmentVersion,mode:draft.mode,hintIDs:draft.hintIDs,grade:result.grade)
                let value=NormalizationLearningReceipt(attempt:attempt,projectID:draft.projectID,submission:submission,submissionSHA256:SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined(),runtime:plan.runtime,fixtureSeed:plan.fixtureSeed,fixtureDigest:plan.fixtureDigest,evidence:evidence)
                do {
                    let encoded=try NormalizationLearningCodec.encodeReceipt(value)
                    self.receipt=value
                    let folder=self.directory.appendingPathComponent("attempts")
                    try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
                    try encoded.write(to:folder.appendingPathComponent(attempt.id+".json"),options:.withoutOverwriting)
                    try self.progress.append(attempt)
                    self.resultSaved=true;self.attempts=try self.progress.attempts();self.verifiedAttemptIDs.insert(attempt.id)
                    do {try encoded.write(to:self.latestURL(draft.projectID),options:.atomic)}
                    catch {self.statusMessage="Attempt saved; latest index could not be updated. Reopening recovers verified recent receipts."}
                } catch {self.statusMessage="Evaluated but not saved: " + error.localizedDescription}
            })
            if !accepted {receipt=previous.0;latestGrade=previous.1;resultSaved=previous.2;statusMessage=session.startFailureMessage ?? "Assessment could not start; the previous result is preserved."}
        } catch {statusMessage=error.localizedDescription}
    }
    private func latestURL(_ id:String)->URL {directory.appendingPathComponent("projects/\(id)/normalization-latest.json")}
    private func restoreReceipt(projectID:String) {
        var remaining=16_000_000
        func accept(_ data:Data)->Bool {
            guard let envelope=try? JSONDecoder().decode(NormalizationLearningReceipt.self,from:data),
                  let candidate=try? NormalizationLearningCodec.decodeReceipt(data,expectedProjectID:envelope.projectID),attempts.contains(candidate.attempt) else {return false}
            verifiedAttemptIDs.insert(candidate.attempt.id)
            if candidate.projectID == projectID && (receipt == nil || (attempts.firstIndex(of:candidate.attempt) ?? -1) > (attempts.firstIndex(where:{$0.id == receipt?.attempt.id}) ?? -1)) {
                receipt=candidate;latestGrade=candidate.attempt.grade;resultSaved=true;statusMessage="Verified measured result restored."
            }
            return true
        }
        let lessonAttempts=attempts.reversed().filter{$0.objectiveID == NormalizationLessonTask.objectiveID}
        let recent=Array(lessonAttempts.prefix(100))
        // Older assessment passes remain discoverable after many formative checks.
        let completions=Array(lessonAttempts.filter{$0.mode == .assessment && $0.grade.status == .passed && $0.assessmentVersion == NormalizationLessonTask.currentVersion}.prefix(100))
        for attempt in completions + recent.filter({!completions.contains($0)}) {
            guard remaining>0,let data=try? read(directory.appendingPathComponent("attempts/\(attempt.id).json"),limit:min(2_000_000,remaining)) else {continue}
            remaining-=data.count
            _ = accept(data)
        }
        if receipt != nil {return}
        if let data=try? read(latestURL(projectID)),let candidate=try? NormalizationLearningCodec.decodeReceipt(data,expectedProjectID:projectID),
           let immutable=try? read(directory.appendingPathComponent("attempts/\(candidate.attempt.id).json")),accept(immutable) {return}
        if !attempts.isEmpty {statusMessage="No verified result for this construction was found in the bounded recent-attempt index. Saved progress is preserved."}
    }
    var openProjectBrowser: (() -> Void)?
    var leaveActionTitle: String { draft?.returnProjectID == nil ? "Projects" : "Return to project" }
    func leave(returnToPreviousProject:Bool) {
        guard let workspace,!isBusy,workspace.learningProjects.owns(self),returnToPreviousProject else {return}
        guard let previous=draft?.returnProjectID else {
            guard let openProjectBrowser,save() else {return}
            openProjectBrowser()
            statusMessage="Projects opened. This saved lesson stays in its own tab."
            return
        }
        do {
            let document=try workspace.projectLibrary.loadDocument(id:previous)
            guard save() else {return}
            let opened:Bool
            switch document {
            case .legacyFlat(let graph): opened=workspace.beginProject(graph,id:previous)
            case .blockModel: opened=workspace.beginOwnedProject(id:previous)
            }
            if !opened {statusMessage=workspace.statusMessage + " This lesson remains active."}
        } catch {statusMessage="Could not open the return project: " + error.localizedDescription}
    }
    private func markerURL(_ id:String)->URL {workspace!.workspaceDirectory.appendingPathComponent("learning/projects/\(id)/normalization.json")}
    @discardableResult private func persistDraft()->Bool {
        guard let draft else {return true}
        do {
            let data=try NormalizationLessonDocument(draft:draft,editor:editor).encoded()
            let url=markerURL(draft.projectID)
            try FileManager.default.createDirectory(at:url.deletingLastPathComponent(),withIntermediateDirectories:true)
            try data.write(to:url,options:.atomic)
            return true
        } catch {statusMessage="Normalization lesson save failed: " + error.localizedDescription;return false}
    }
    private func read(_ url:URL,limit:Int=2_000_000)throws->Data {
        let info=try url.resourceValues(forKeys:[.isRegularFileKey,.isSymbolicLinkKey,.fileSizeKey])
        guard info.isRegularFile == true,info.isSymbolicLink != true,(info.fileSize ?? Int.max)<=limit else {throw LearningContractError("Lesson document must be a bounded regular file.")}
        let handle=try FileHandle(forReadingFrom:url);defer {try? handle.close()}
        let data=try handle.read(upToCount:limit+1) ?? Data()
        guard data.count<=limit else {throw LearningContractError("Lesson document exceeds bounds.")}
        return data
    }
}

/// One atomic local document keeps the Core draft and nonsemantic editor metadata together.
private struct NormalizationLessonDocument:Codable {
    var schemaVersion=1
    var kind="normalizationLesson"
    var draft:NormalizationLearningDraft
    var editor:[String:DataFlowEditorNode]
    func validate()throws {
        guard schemaVersion == 1,kind == "normalizationLesson" else {throw LearningContractError("Unsupported normalization lesson document.")}
        try draft.validate()
        try DataFlowCodec.validate(DataFlowProject(graph:draft.submission.graph,editor:editor))
    }
    func encoded()throws->Data {
        try validate()
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        let data=try encoder.encode(self)
        guard data.count<=2_000_000 else {throw LearningContractError("Lesson document exceeds 2 MB.")}
        return data
    }
    static func decode(_ data:Data)throws->Self {
        guard data.count<=2_000_000,case .object(let root)=try JSONDecoder().decode(JSONValue.self,from:data),Set(root.keys)==["schemaVersion","kind","draft","editor"],let rawDraft=root["draft"],case .object(let editor)=root["editor"] else {throw LearningContractError("Invalid normalization lesson envelope.")}
        _ = try NormalizationLearningCodec.decodeDraft(JSONEncoder().encode(rawDraft))
        for value in editor.values {guard case .object(let node)=value,Set(node.keys)==["name","x","y"] else {throw LearningContractError("Invalid normalization editor metadata.")}}
        let value=try JSONDecoder().decode(Self.self,from:data);try value.validate();return value
    }
}
