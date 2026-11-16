import Foundation
import Combine
import KwipMLCore

/// Tensor lesson lifecycle with locally verified measured-result receipts.
@MainActor final class TensorLearningStore:ObservableObject {
    let drafts:TensorLearningDraftSession
    @Published private(set) var canvas:TensorLearningCanvasAdapter?
    @Published private(set) var compositionCanvas:ModuleCompositionLearningCanvasSession?
    @Published private(set) var grade:LearningGrade?
    @Published private(set) var resultSaved=false
    private let receipts:TensorLearningReceiptRepository
    @Published private(set) var statusMessage="Build a tensor transformation and predict its values and gradients."
    private weak var workspace:WorkspaceStore?
    private let measurement=MeasuredLearningSession()
    private var submitted:Data?
    private var receipt:TensorLearningReceipt?
    struct CompositionResults {
        let task:ModuleCompositionLearningTask
        let projection:ModuleCompositionFormativeProjection
        let metadata:[ModuleCompositionLearningMetadata]
        let nodeLabels:[String:String]
    }
    private var compositionResultCache:(attempt:String,bytes:Data,mode:LearningMode,value:CompositionResults)?
    private var observation:AnyCancellable?
    private var workspaceObservation:AnyCancellable?
    init(workspace:WorkspaceStore) {
        self.workspace=workspace;drafts=workspace.learningProjects.tensorDraftSession
        receipts=TensorLearningReceiptRepository(directory:workspace.workspaceDirectory.appendingPathComponent("learning/projects"))
        workspace.learningProjects.attachTensor(self)
        observation=drafts.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()}
        workspaceObservation=workspace.objectWillChange.sink {[weak self] _ in
            self?.objectWillChange.send()
            self?.canvas?.objectWillChange.send()
        }
    }
    var draft:TensorLearningDraft? {drafts.draft}
    var isActive:Bool {
        guard let workspace,workspace.learningProjects.owns(self),let draft else {return false}
        return workspace.activeProjectID == draft.projectID && workspace.lessonContext?.kind == .tensor && workspace.lessonContext?.restoreState == .ready
    }
    var isBusy:Bool {workspace?.isBusy ?? false}
    var runtimeReadinessError:String? {
        guard let operation=draft?.operation else {return nil}
        if RetiredTrainingLessonDestination.contains(operation) {return "This historical construction lesson is read-only. Use the current Train or layer practice."}
        if DataLoadingLearningTask(rawValue:operation) != nil {return workspace?.dataLoadingRuntimeCapabilities.contains("dataset_source") == true && workspace?.dataLoadingRuntimeCapabilities.contains("data_loader") == true ? nil:"This lesson requires verified Dataset and DataLoader support together. Refresh the lesson runtime below."}
        if ProgramExportLearningTask(rawValue:operation) != nil {return workspace?.programExportRuntimeCapability.isAvailable == true ? nil:"This lesson requires verified executable-program capture and reload support. Refresh the lesson runtime below."}
        if ModuleCompositionLearningTask(rawValue:operation) != nil {
            guard let runtime=workspace?.runtime,let proof=workspace?.moduleCompositionDiscoveryEvidence,
                  proof.runtime == runtime else {return "This lesson requires verified reusable-module support for the selected runtime. Refresh the lesson runtime below."}
            return nil
        }
        if GradientLearningTask(rawValue:operation) != nil {return workspace?.gradientControlRuntimeCapabilities.contains("gradient_policy") == true && workspace?.gradientControlRuntimeCapabilities.contains("trainability") == true && workspace?.optimizerRuntimeCapabilities.contains("optimizer_sgd") == true ? nil:"This lesson requires verified gradient policy, trainability and supplied SGD support. Lesson runtime: \(workspace?.capabilityStatus ?? "Unavailable"). Refresh the lesson runtime below."}
        if InitializationLearningTask(rawValue:operation) != nil {return workspace?.initializationRuntimeCapabilities.contains("initialization") == true && workspace?.optimizerRuntimeCapabilities.contains("optimizer_sgd") == true ? nil:"This lesson requires verified starting-value and supplied SGD support. Lesson runtime: \(workspace?.capabilityStatus ?? "Unavailable"). Refresh the lesson runtime below."}
        if CheckpointLearningTask(rawValue:operation) != nil {return workspace?.lifecycleRuntimeCapability.isAvailable == true ? nil:"This lesson requires verified checkpoint save and restore support. Refresh the lesson runtime below."}
        if let task=LossLearningTask(rawValue:operation) {return workspace?.lossRuntimeCapabilities.contains(task.componentID) == true ? nil:"This lesson requires verified objective and typed target support. Refresh the lesson runtime below."}
        if let task=AttentionLearningTask(rawValue:operation) {return workspace?.attentionRuntimeCapabilities.contains(task.componentID) == true && workspace?.attentionRuntimeCapabilities.contains("attention_mask") == true ? nil:"This lesson requires verified attention and blocked-connection support. Lesson runtime: \(workspace?.capabilityStatus ?? "Unavailable"). Refresh the lesson runtime below."}
        if let task=RecurrentLearningTask(rawValue:operation) {return workspace?.recurrentRuntimeCapabilities.contains(task.componentID) == true ? nil:"This lesson requires verified sequence and named-state support. Lesson runtime: \(workspace?.capabilityStatus ?? "Unavailable"). Refresh the lesson runtime below."}
        if let task=SchedulerLearningTask(rawValue:operation) {return workspace?.schedulerRuntimeCapabilities.contains(task.componentID) == true && workspace?.optimizerRuntimeCapabilities.contains("optimizer_sgd") == true ? nil:"This lesson requires verified schedule and supplied SGD support. Lesson runtime: \(workspace?.capabilityStatus ?? "Unavailable"). Refresh the lesson runtime below."}
        if let task=OptimizerLearningTask(rawValue:operation) {return workspace?.optimizerRuntimeCapabilities.contains(task.componentID) == true ? nil:"This lesson requires verified training-recipe and update-state support. Lesson runtime: \(workspace?.capabilityStatus ?? "Unavailable"). Refresh the lesson runtime below."}
        if let task=StatefulNormalizationLearningTask(rawValue:operation) {
            return workspace?.statefulNormalizationRuntimeCapabilities.contains(task.componentID) == true ? nil : "Channel-normalization lessons require verified grouping and saved-statistics support. Refresh runtime status in Settings."
        }
        if let task=SpatialPatchLearningTask(rawValue:operation) {
            return workspace?.spatialPatchRuntimeCapabilities.contains(task.componentID) == true ? nil : "Patch lessons require verified window geometry and overlapping addition support. Refresh runtime status in Settings."
        }
        if let task=TensorRoutingLearningTask(rawValue:operation) {
            return workspace?.tensorRoutingRuntimeCapabilities.contains(task.componentID) == true ? nil : "Routing lessons require verified named-part, selector and arrangement support. Refresh runtime status in Settings."
        }
        if let task=RemainingActivationLearningTask(rawValue:operation) {
            return workspace?.remainingActivationRuntimeCapabilities.contains(task.componentID) == true ? nil : "This lesson requires verified runtime support for learned slopes, sample directions or training/evaluation behavior. Refresh runtime status in Settings."
        }
        if let task=TensorCompositionLearningTask(rawValue:operation) {
            return workspace?.tensorCompositionRuntimeCapabilities.contains(task.componentID) == true ? nil : "Composition lessons require verified runtime support. Refresh runtime status in Settings."
        }
        if let task=AdaptivePoolingLearningTask(rawValue:operation) {
            return workspace?.poolingRuntimeCapabilities.contains(task.componentID) == true ? nil : "Averaging/adaptive pooling lessons require verified runtime support. Refresh runtime status in Settings."
        }
        if let task=ConfigurableActivationLearningTask(rawValue:operation) {
            return workspace?.configurableActivationRuntimeCapabilities.contains(task.componentID) == true ? nil : "Activation lessons require verified runtime support. Refresh runtime status in Settings."
        }
        if let task=PoolingLearningTask(rawValue:operation) {
            return workspace?.poolingRuntimeCapabilities.contains(task.componentID) == true ? nil : "Pooling lessons require verified runtime support. Refresh runtime status in Settings."
        }
        if let task=ConvolutionLearningTask(rawValue:operation) {
            let supported = task.isTransposed
                ? (task.dimensions == 1 ? workspace?.convTranspose1dRuntimeCapability != nil : task.dimensions == 2 ? workspace?.convTranspose2dRuntimeCapability != nil : workspace?.convTranspose3dRuntimeCapability != nil)
                : (task == .one ? workspace?.conv1dRuntimeCapability != nil : task == .two ? workspace?.conv2dRuntimeCapability != nil : workspace?.conv3dRuntimeCapability != nil)
            return supported ? nil : "\(task.componentID) lessons require verified runtime support. Refresh runtime status in Settings."
        }
        if operation == LazyLinearAssessment.operation {
            return workspace?.lazyLinearRuntimeCapability == nil ? "LazyLinear lessons require verified inferred-width and state support. Refresh runtime status in Settings.":nil
        }
        if BilinearLearningTask(rawValue:operation) != nil {
            return workspace?.bilinearRuntimeCapability == nil ? "Bilinear lessons require verified runtime support. Refresh runtime status in Settings.":nil
        }
        if MatrixLearningTask(rawValue:operation) != nil {
            return workspace?.matrixMultiplyRuntimeCapability == nil ? "Matrix lessons require verified runtime support. Refresh runtime status in Settings.":nil
        }
        if EmbeddingBagLearningTask(rawValue:operation) != nil {
            return workspace?.embeddingBagRuntimeCapabilities.isComplete == true ? nil : "Bag lessons require verified Bag Input and EmbeddingBag support together. Refresh the lesson runtime below."
        }
        if EmbeddingLearningTask(rawValue:operation) != nil {
            return workspace?.embeddingRuntimeCapability == nil ? "Embedding lessons require verified runtime support. Refresh runtime status in Settings.":nil
        }
        if LayerNormLearningTask(rawValue:operation) != nil {
            return workspace?.layerNormRuntimeCapability == nil ? "LayerNorm lessons require verified runtime support. Refresh runtime status in Settings.":nil
        }
        if TensorLearningTask.softmaxAxis(for:operation) != nil {
            return workspace?.softmaxRuntimeCapability == nil ? "Softmax lessons require verified runtime support. Refresh runtime status in Settings.":nil
        }
        return nil
    }
    var canRefreshLessonRuntime:Bool {!RetiredTrainingLessonDestination.contains(draft?.operation) && isActive && (DataLoadingLearningTask(rawValue:draft?.operation ?? "") != nil || ProgramExportLearningTask(rawValue:draft?.operation ?? "") != nil || EmbeddingBagLearningTask(rawValue:draft?.operation ?? "") != nil || ModuleCompositionLearningTask(rawValue:draft?.operation ?? "") != nil || GradientLearningTask(rawValue:draft?.operation ?? "") != nil || InitializationLearningTask(rawValue:draft?.operation ?? "") != nil || CheckpointLearningTask(rawValue:draft?.operation ?? "") != nil || LossLearningTask(rawValue:draft?.operation ?? "") != nil || AttentionLearningTask(rawValue:draft?.operation ?? "") != nil || RecurrentLearningTask(rawValue:draft?.operation ?? "") != nil || OptimizerLearningTask(rawValue:draft?.operation ?? "") != nil || SchedulerLearningTask(rawValue:draft?.operation ?? "") != nil) && !isBusy && workspace != nil}
    @discardableResult func refreshLessonRuntime()->Bool {
        guard canRefreshLessonRuntime,let workspace else {return false}
        workspace.refreshCapabilities()
        return true
    }
    private func currentSubmission()throws->Data {
        guard let draft else {throw LearningContractError("Open a lesson first.")}
        return try draft.submissionData(compositionRegistry:receipt?.compositionRegistry ?? workspace?.registry)
    }
    var canSubmit:Bool {runtimeReadinessError == nil && isActive && !isBusy && workspace?.runtime != nil && (try? currentSubmission()) != nil && (ProgramExportLearningTask(rawValue:draft?.operation ?? "") != nil || ModuleCompositionLearningTask(rawValue:draft?.operation ?? "") != nil || CheckpointLearningTask(rawValue:draft?.operation ?? "") != nil || canvas?.validation.isValid == true)}
    var resultIsCurrent:Bool {isActive && submitted != nil && submitted == (try? currentSubmission())}
    var compositionResults:CompositionResults? {
        guard isActive,!isBusy,let draft,let task=ModuleCompositionLearningTask(rawValue:draft.operation),
              draft.mode != .assessment,let receipt,receipt.attempt.mode == draft.mode,
              let registry=receipt.compositionRegistry,let bytes=try? currentSubmission(),bytes == submitted else {return nil}
        if let cache=compositionResultCache,cache.attempt == receipt.attempt.id,cache.bytes == bytes,cache.mode == draft.mode {return cache.value}
        guard let plan=try? receipt.verifiedReplay(expectedProjectID:draft.projectID),
              let projection=ModuleCompositionFormativeProjection.project(plan:plan,evidence:receipt.evidence,currentSubmission:bytes,registry:registry,mode:draft.mode,attemptMode:receipt.attempt.mode),
              let metadata=ModuleCompositionLearningMetadata.project(plan:plan,evidence:receipt.evidence,currentSubmission:bytes,registry:registry,mode:draft.mode,attemptMode:receipt.attempt.mode),
              let source=try? ModuleCompositionLearningSubmission.decode(bytes,task:task,registry:registry),
              let expanded=try? CompositeExpander.expand(source.source,registry:registry,context:.explicitObjective) else {return nil}
        let labels=Dictionary(uniqueKeysWithValues:expanded.graph.nodes.map {($0.id,$0.name.isEmpty ? $0.id:$0.name)})
        let value=CompositionResults(task:task,projection:projection,metadata:metadata,nodeLabels:labels)
        compositionResultCache=(receipt.attempt.id,bytes,draft.mode,value)
        return value
    }
    var recipeProjection:RecipeFormativeProjection? {
        guard isActive,resultIsCurrent,let receipt,let draft,let bytes=try? currentSubmission() else {return nil}
        return receipt.recipeFormativeProjection(expectedProjectID:draft.projectID,currentSubmission:bytes,currentMode:draft.mode)
    }
    var formativeProjection:TensorFormativeProjection? {
        guard isActive,resultIsCurrent,let receipt,let draft,let bytes=try? draft.submissionData() else {return nil}
        return receipt.formativeProjection(expectedProjectID:draft.projectID,currentSubmission:bytes,currentMode:draft.mode)
    }
    struct InitializationMeasuredFixture:Identifiable {
        let id:String
        let measured:[String:LearningTensor]
    }
    struct InitializationMeasuredProjection {
        let submissionDigest:String
        let fixtures:[InitializationMeasuredFixture]
    }
    var initializationProjection:InitializationMeasuredProjection? {
        guard isActive,resultIsCurrent,let draft,InitializationLearningTask(rawValue:draft.operation) != nil,
              draft.mode != .assessment,let receipt,receipt.attempt.mode != .assessment,
              let plan=try? receipt.verifiedReplay(expectedProjectID:draft.projectID) else {return nil}
        let keys:Set<String>=["startingSettings","predictions","firstAdjustment","repeatIsolation","savedReuse"]
        var fixtures:[InitializationMeasuredFixture]=[]
        for fixture in plan.fixtures {
            guard let measured=receipt.evidence.fixtures.first(where:{$0.fixtureID == fixture.id}),Set(measured.tensors.keys)==keys else {return nil}
            for key in keys {guard let tensor=measured.tensors[key],tensor.wellFormed,tensor.shape == fixture.expected[key]?.shape else {return nil}}
            fixtures.append(.init(id:fixture.id,measured:measured.tensors))
        }
        return .init(submissionDigest:receipt.submissionSHA256,fixtures:fixtures)
    }
    var gradientProjection:GradientFormativeProjection? {
        guard isActive,resultIsCurrent,let receipt,let draft,GradientLearningTask(rawValue:draft.operation) != nil,
              receipt.operation == draft.operation,let bytes=try? draft.submissionData(),
              let plan=try? receipt.verifiedReplay(expectedProjectID:draft.projectID) else {return nil}
        return GradientFormativeProjection.project(plan:plan,evidence:receipt.evidence,currentSubmission:bytes,mode:draft.mode,attemptMode:receipt.attempt.mode)
    }
    var checkpointProjection:CheckpointFormativeProjection? {
        guard isActive,resultIsCurrent,let receipt,let draft,let bytes=try? draft.submissionData() else {return nil}
        return receipt.checkpointFormativeProjection(expectedProjectID:draft.projectID,currentSubmission:bytes,currentMode:draft.mode)
    }
    var checkpointConfiguration:CheckpointLearningConfiguration? {draft?.checkpointConfiguration}
    func checkpointField(_ key:String)->String {
        if let pending=draft?.pendingFields["checkpoint."+key] {return pending}
        guard let configuration=checkpointConfiguration else {return ""}
        switch key {
        case "trainingMode":return configuration.trainingMode.rawValue
        case "parent":return configuration.parent.rawValue
        case "epochs":return String(configuration.epochs)
        case "batchSize":return String(configuration.batchSize)
        default:return ""
        }
    }
    @discardableResult func setCheckpointField(_ key:String,text:String)->Bool {
        guard isActive,!isBusy,CheckpointLearningTask(rawValue:draft?.operation ?? "") != nil,
              ["trainingMode","parent","epochs","batchSize"].contains(key),text.count<=512 else {return false}
        return drafts.update {$0.pendingFields["checkpoint."+key]=text}
    }
    @discardableResult func applyCheckpointFields()->Bool {
        guard isActive,!isBusy,CheckpointLearningTask(rawValue:draft?.operation ?? "") != nil else {return false}
        do {
            guard let mode=WorkflowTrainingMode(rawValue:checkpointField("trainingMode")),
                  let parent=CheckpointLearningParent(rawValue:checkpointField("parent")),
                  let epochs=Int(checkpointField("epochs")),let batch=Int(checkpointField("batchSize")) else {
                throw LearningContractError("Choose a training mode and saved parent, and enter whole numbers from 1 to 8.")
            }
            let configuration=try CheckpointLearningConfiguration(trainingMode:mode,parent:parent,epochs:epochs,batchSize:batch)
            guard drafts.update({$0.checkpointConfiguration=configuration;$0.pendingFields=$0.pendingFields.filter {!$0.key.hasPrefix("checkpoint.")}}) else {
                statusMessage=drafts.errorMessage ?? "Checkpoint controls could not be applied.";return false
            }
            statusMessage="Checkpoint controls applied. Check the exercise to measure their behavior."
            return true
        } catch {statusMessage=error.localizedDescription;return false}
    }
    @discardableResult func discardCheckpointFields()->Bool {
        guard isActive,!isBusy,CheckpointLearningTask(rawValue:draft?.operation ?? "") != nil else {return false}
        return drafts.update {$0.pendingFields=$0.pendingFields.filter {!$0.key.hasPrefix("checkpoint.")}}
    }
    var lossProjection:LossFormativeProjection? {
        guard isActive,resultIsCurrent,let receipt,let draft,let bytes=try? draft.submissionData() else {return nil}
        return receipt.lossFormativeProjection(expectedProjectID:draft.projectID,currentSubmission:bytes,currentMode:draft.mode)
    }
    var attentionProjection:AttentionFormativeProjection? {
        guard isActive,resultIsCurrent,let receipt,let draft,let bytes=try? draft.submissionData() else {return nil}
        return receipt.attentionFormativeProjection(expectedProjectID:draft.projectID,currentSubmission:bytes,currentMode:draft.mode)
    }
    var recurrentProjection:RecurrentFormativeProjection? {
        guard isActive,resultIsCurrent,let receipt,let draft,let bytes=try? draft.submissionData() else {return nil}
        return receipt.recurrentFormativeProjection(expectedProjectID:draft.projectID,currentSubmission:bytes,currentMode:draft.mode)
    }
    var schedulerProjection:SchedulerFormativeProjection? {
        guard isActive,resultIsCurrent,let receipt,let draft,let bytes=try? draft.submissionData() else {return nil}
        return receipt.schedulerFormativeProjection(expectedProjectID:draft.projectID,currentSubmission:bytes,currentMode:draft.mode)
    }
    var optimizerProjection:OptimizerFormativeProjection? {
        guard isActive,resultIsCurrent,let receipt,let draft,let bytes=try? draft.submissionData() else {return nil}
        return receipt.optimizerFormativeProjection(expectedProjectID:draft.projectID,currentSubmission:bytes,currentMode:draft.mode)
    }
    struct LazyStateRow: Identifiable {
        let id: String
        let check: String
        let measured: String
        let meaning: String
    }
    var lazyStatePresentation: [LazyStateRow]? {
        guard isActive, resultIsCurrent, let receipt, let draft, let submission = try? draft.submissionData() else { return nil }
        return Self.lazyStateRows(receipt: receipt, projectID: draft.projectID, submission: submission, mode: draft.mode)
    }
    /// Safe presentation of measured shapes/invariants only; never exposes private
    /// inputs, initialization seeds, learned values or expected fixture tables.
    static func lazyStateRows(receipt: TensorLearningReceipt, projectID: String, submission: Data, mode: LearningMode) -> [LazyStateRow]? {
        guard mode != .assessment, receipt.attempt.mode != .assessment,
              receipt.operation == LazyLinearAssessment.operation,
              receipt.submissionData == submission,
              (try? receipt.verifiedReplay(expectedProjectID: projectID)) != nil else { return nil }
        let definitions = [
            ("output-shape", "Output shape", "N means the number of observations; groups and results remain separate."),
            ("inferred-width", "Inferred input features", "The connected array’s final direction determines this width."),
            ("weight-shape", "Created weight shape", "One row of weights per output; one column per input feature."),
            ("parameter-count", "Parameter inventory matches selected settings", "For the requested four outputs: twelve weights plus four offsets when Learn bias is on; the checker compares the actual inventory."),
            ("repeat-state", "Repeated calculation preserves state", "The same weights and observations produce unchanged results."),
            ("reload-state", "Reload preserves state and results", "Loading the saved weights preserves the calculation."),
            ("rng-preserved", "Repeat and reload avoid fresh random draws", "Starting weights are created once; these checks reuse them.")]
        var rows: [LazyStateRow] = []
        for (key, title, meaning) in definitions {
            let values = receipt.evidence.fixtures.compactMap { $0.tensors[key]?.values }
            guard values.count == receipt.evidence.fixtures.count, let first = values.first else { return nil }
            if ["output-shape", "inferred-width", "weight-shape"].contains(key) {
                guard values.allSatisfy({ !$0.isEmpty && $0.count <= 8 && $0.allSatisfy { $0.isFinite && $0.rounded() == $0 && (1...4096).contains($0) } }) else { return nil }
            }
            let measured: String
            if key == "output-shape" {
                let samples = values.map { Array($0.dropFirst()) }
                measured = samples.allSatisfy { $0 == samples[0] }
                    ? "[N, " + first.dropFirst().map { String(Int($0)) }.joined(separator: ", ") + "]" : "Shapes differ across checks"
            } else if ["inferred-width", "weight-shape"].contains(key) {
                measured = values.allSatisfy { $0 == first }
                    ? (key == "inferred-width" ? String(Int(first[0])) : "[" + first.map { String(Int($0)) }.joined(separator: ", ") + "]") : "Varies across checks"
            } else { measured = values.allSatisfy { $0 == [1] } ? "Yes" : "No" }
            rows.append(.init(id: key, check: title, measured: measured, meaning: meaning))
        }
        return rows
    }
    var publicTask:LearningPublicTask? {
        guard let draft else {return nil}
        // Only the safe projection leaves this controller; private fixtures are never view state.
        let version=RemainingActivationLearningTask(rawValue:draft.operation)?.isDropout == true ? "2.14.1":"required"
        return try? TensorLearningTask.plan(operation:draft.operation,runtimeVersion:version,seed:0).publicTask(mode:draft.mode)
    }
    var startInWindow: ((String, LearningMode) -> Bool)?
    @discardableResult func start(operation:String,mode:LearningMode = .guided)->Bool {
        if let startInWindow { return startInWindow(operation, mode) }
        let example=CheckpointLearningExamples.example(id:operation)
        let initializationExample=InitializationLearningTask.allCases.first {"prepared-"+$0.rawValue == operation}
        let gradientExample=GradientLearningTask.allCases.first {"prepared-"+$0.rawValue == operation}
        let compositionExample=ModuleCompositionLearningTask.allCases.first {"prepared-"+$0.rawValue == operation}
        let resolvedOperation=compositionExample?.rawValue ?? gradientExample?.rawValue ?? initializationExample?.rawValue ?? example?.task.rawValue ?? operation
        guard let title=TensorLearningTask.title(for:resolvedOperation),let workspace,!workspace.isBusy,
              workspace.learningProjects.owns(self),((workspace.startsIndependentLesson && workspace.activeProjectID == nil) || workspace.saveManagedProject()),
              workspace.beginProject(GraphDocument(name:"Learn · "+title)),let id=workspace.activeProjectID else {return false}
        guard drafts.create(projectID:id,operation:resolvedOperation,mode:mode) else {statusMessage=drafts.errorMessage ?? "Lesson could not be created.";return false}
        if let task=DataLoadingLearningTask(rawValue:resolvedOperation) {
            do {let graph=try DataLoadingExamples.lessonGraph(task:task);guard drafts.update({$0.graph.nodes=[graph.nodes[2]]}),drafts.save() else {return false}}
            catch {statusMessage=error.localizedDescription;return false}
        }
        if let compositionExample {
            do {
                let model=try ModuleCompositionLearningDraft.model(projectID:id,task:compositionExample,prepared:true)
                guard drafts.update({$0.compositionModel=model}),drafts.save() else {return false}
            } catch {statusMessage=error.localizedDescription;return false}
        }
        if let task=LossLearningTask(rawValue:resolvedOperation),!drafts.update({$0.graph=LossLearningCanvas.preparedGraph(task:task)}) {statusMessage=drafts.errorMessage ?? "Objective construction could not be prepared.";return false}
        if let gradientExample {
            do {let prepared=try GradientLearningExamples.preparedGraph(task:gradientExample);guard drafts.update({$0.graph=prepared}),drafts.save() else {return false}}
            catch {statusMessage=error.localizedDescription;return false}
        }
        if let initializationExample {
            do {let prepared=try InitializationLearningExamples.preparedGraph(task:initializationExample)
                guard drafts.update({$0.graph=prepared}),drafts.save() else {return false}
            }catch {statusMessage=error.localizedDescription;return false}
        }
        if let example {
            guard drafts.update({$0.checkpointConfiguration=example.configuration}),drafts.save() else {statusMessage=drafts.errorMessage ?? "Worked example could not be prepared.";return false}
        }
        workspace.learningProjects.restoreCurrentProject()
        workspace.refreshProjects()
        return isActive
    }
    func restoreEditor() {
        guard let workspace,workspace.learningProjects.owns(self),draft?.projectID == workspace.activeProjectID else {return}
        if compositionCanvas != nil,ModuleCompositionLearningTask(rawValue:draft?.operation ?? "") != nil {return}
        if canvas?.historyContextIsActive == true {return}
        grade=nil;submitted=nil;resultSaved=false;receipt=nil
        if ModuleCompositionLearningTask(rawValue:draft?.operation ?? "") != nil {
            canvas=nil
            do {
                compositionCanvas=try ModuleCompositionLearningCanvasSession(drafts:drafts,registry:workspace.registry,
                    isCurrent:{[weak self] in self?.isActive == true},isBusy:{[weak self] in self?.isBusy ?? true})
                statusMessage="Open a Block to connect its parts. Apply or discard its changes before checking the model."
                if let draft,let saved=try receipts.latest(projectID:draft.projectID),saved.operation == draft.operation,saved.attempt.mode == draft.mode {
                    grade=saved.attempt.grade;submitted=saved.submissionData;resultSaved=true;receipt=saved
                    statusMessage=draft.mode == .assessment ? "Saved assessment verified from its captured model, registry and measured results.":"Saved example check verified from its captured model, registry and measured results."
                }
            } catch {statusMessage="The lesson could not be fully restored: "+error.localizedDescription}
            return
        }
        compositionCanvas?.invalidate();compositionCanvas=nil
        if CheckpointLearningTask(rawValue:draft?.operation ?? "") != nil || ProgramExportLearningTask(rawValue:draft?.operation ?? "") != nil {
            let isProgramExport=ProgramExportLearningTask(rawValue:draft?.operation ?? "") != nil
            canvas=nil
            statusMessage=isProgramExport
                ? "Export controls restored. Check reloaded results, accepted input sizes and saved-parent integrity."
                : "Checkpoint controls restored. Choose how to reuse the saved lesson run, then check the measured result."
            do {
                if let draft,let saved=try receipts.latest(projectID:draft.projectID) {
                    guard saved.operation == draft.operation else {throw LearningContractError(isProgramExport ? "Saved result belongs to another program-export exercise.":"Saved result belongs to another checkpoint exercise.")}
                    if saved.attempt.mode == draft.mode {
                        grade=saved.attempt.grade;submitted=saved.submissionData;resultSaved=true;receipt=saved
                        statusMessage="Saved "+(isProgramExport ? "program-export ":"checkpoint ")+(draft.mode == .assessment ? "assessment":"example check")+" verified from its controls and measured evidence."
                    }
                }
            } catch {statusMessage=(isProgramExport ? "Export controls restored; saved result could not be verified: ":"Checkpoint controls restored; saved result could not be verified: ")+error.localizedDescription}
            return
        }
        let editor=TensorLearningCanvasAdapter(session:drafts)
        editor.assessmentIsRunning = {[weak workspace] in workspace?.isBusy ?? true}
        editor.evidenceState = {[weak self] in
            guard let self,self.isActive,self.receipt != nil else {return .none}
            return self.resultIsCurrent ? .current:.stale
        }
        // Canvas availability checks share one verified projection per receipt.
        // Replaying scalar lesson gold for every node/port would block rendering.
        var lossCache:(attempt:String,fixture:String,bytes:Data,mode:LearningMode,projection:LossFormativeProjection?)?
        editor.lossProjectionProvider = {[weak self] in
            guard let self,self.isActive,self.resultIsCurrent,!self.isBusy,let receipt=self.receipt,let draft=self.draft,draft.mode != .assessment,let bytes=try? draft.submissionData() else {lossCache=nil;return nil}
            if let cache=lossCache,cache.attempt == receipt.attempt.id,cache.fixture == receipt.fixtureDigest,cache.bytes == bytes,cache.mode == draft.mode {return cache.projection}
            let projection=receipt.lossFormativeProjection(expectedProjectID:draft.projectID,currentSubmission:bytes,currentMode:draft.mode)
            lossCache=(receipt.attempt.id,receipt.fixtureDigest,bytes,draft.mode,projection);return projection
        }
        var attentionCache:(attempt:String,fixture:String,bytes:Data,mode:LearningMode,projection:AttentionFormativeProjection?)?
        editor.attentionProjectionProvider = {[weak self] in
            guard let self,self.isActive,self.resultIsCurrent,!self.isBusy,let receipt=self.receipt,let draft=self.draft,
                  draft.mode != .assessment,let bytes=try? draft.submissionData() else {attentionCache=nil;return nil}
            if let cache=attentionCache,cache.attempt == receipt.attempt.id,cache.fixture == receipt.fixtureDigest,cache.bytes == bytes,cache.mode == draft.mode {return cache.projection}
            let projection=receipt.attentionFormativeProjection(expectedProjectID:draft.projectID,currentSubmission:bytes,currentMode:draft.mode)
            attentionCache=(receipt.attempt.id,receipt.fixtureDigest,bytes,draft.mode,projection)
            return projection
        }
        canvas=editor
        statusMessage=GradientLearningTask(rawValue:draft?.operation ?? "") != nil
            ? "Training controls restored. Check influence, adjustment windows, fixed settings or saved-parent changes."
            : InitializationLearningTask(rawValue:draft?.operation ?? "") != nil
            ? "Starting settings restored. Check starting arrays, predictions, first adjustments and saved-weight reuse."
            : RecurrentLearningTask(rawValue:draft?.operation ?? "") != nil
            ? "Construction restored. Compare every time position, final state and measured influence."
            : SchedulerLearningTask(rawValue:draft?.operation ?? "") != nil
            ? "Schedule restored. Check rates, learned updates, pass timing and the resumed sequence."
            : OptimizerLearningTask(rawValue:draft?.operation ?? "") != nil
            ? "Recipe restored. Check measured setting changes, remembered quantities and parameters without gradients."
            : TensorRoutingLearningTask(rawValue:draft?.operation ?? "") != nil
            ? "Construction restored. Check named parts, selected positions and number-source influence in the measured tables."
            : TensorCompositionLearningTask(rawValue:draft?.operation ?? "") != nil
            ? "Construction restored. Check source order, output values and small-change influence on each source."
            : BilinearLearningTask(rawValue:draft?.operation ?? "") != nil
            ? "Construction restored. Check bilinear interactions and gradients for both inputs, the shared weight and active bias."
            : MatrixLearningTask(rawValue:draft?.operation ?? "") != nil
            ? "Construction restored. Check row–column products and gradients for both operands."
            : EmbeddingBagLearningTask(rawValue:draft?.operation ?? "") != nil
            ? "Construction restored. Check whole groups, real item weights, combined vectors and table influence. Category IDs and offsets have no gradient available."
            : EmbeddingLearningTask(rawValue:draft?.operation ?? "") != nil
            ? "Construction restored. Check gathered values, table gradients and forward table changes."
            : DataLoadingLearningTask(rawValue:draft?.operation ?? "") != nil
            ? "Recipe construction restored. Check selected rows, aligned desired outcomes and observed groups."
            : ProgramExportLearningTask(rawValue:draft?.operation ?? "") != nil
            ? "Export controls restored. Check reloaded results, accepted input sizes and saved-parent integrity."
            : "Construction restored. Check it to measure values and input gradients."
        do {
            if let draft,let receipt=try receipts.latest(projectID:draft.projectID) {
                guard receipt.operation == draft.operation else {throw LearningContractError("Saved receipt belongs to another tensor task.")}
                if receipt.attempt.mode == draft.mode {
                    grade=receipt.attempt.grade;submitted=receipt.submissionData;resultSaved=true;self.receipt=receipt
                    statusMessage=draft.mode == .assessment ? "Saved assessment verified from its task, submission and measured evidence.":"Saved example check verified from its task, submission and measured evidence."
                } else {statusMessage="Construction restored. A saved attempt exists in another learning mode."}
            }
        } catch {statusMessage="Construction restored; saved assessment could not be verified: " + error.localizedDescription}
    }
    func clearProjectContext() {compositionCanvas?.invalidate();compositionCanvas=nil;canvas=nil;grade=nil;submitted=nil;resultSaved=false;receipt=nil}
    @discardableResult func setMode(_ mode:LearningMode)->Bool {
        guard isActive,!isBusy else {return false}
        guard drafts.update({$0.mode=mode}) else {return false}
        grade=nil;submitted=nil;resultSaved=false;receipt=nil
        statusMessage=mode.rawValue.capitalized+" mode selected. Check the construction for a result in this mode."
        return true
    }
    @discardableResult func save()->Bool {
        guard isActive,!isBusy,let workspace else {return false}
        guard drafts.save() else {statusMessage=drafts.errorMessage ?? "Lesson save failed.";return false}
        guard workspace.saveManagedProject() else {statusMessage=workspace.statusMessage;return false}
        statusMessage="Lesson construction saved."
        return true
    }
    func submit() {
        guard !RetiredTrainingLessonDestination.contains(draft?.operation) else {statusMessage="This historical construction lesson is read-only. Use the current Train or layer practice.";return}
        if let error=runtimeReadinessError {statusMessage=error;return}
        guard canSubmit,let workspace,let draft else {statusMessage="Complete the graph and apply or discard pending fields before checking.";return}
        do {
            var compositionRegistry:ComponentRegistry?
            var compositionRegistryBytes:Data?
            var compositionInventory:WorkflowRuntimeEvidence?
            if ModuleCompositionLearningTask(rawValue:draft.operation) != nil {
                guard let runtime=workspace.runtime,let proof=workspace.moduleCompositionDiscoveryEvidence else {throw LearningContractError("Refresh reusable-module runtime support before checking.")}
                let before=try WorkflowRuntimeEvidence.capture(runtime)
                _ = try proof.verifiedDescriptor(for:runtime,evidence:before)
                compositionInventory=before
                let registryBytes=try Data(contentsOf:runtime.registryURL)
                guard registryBytes.count<=2_000_000 else {throw LearningContractError("The captured component registry exceeds the lesson limit.")}
                compositionRegistryBytes=registryBytes
                compositionRegistry=try JSONDecoder().decode(ComponentRegistry.self,from:registryBytes)
                guard try WorkflowRuntimeEvidence.capture(runtime) == before else {throw LearningContractError("The lesson runtime changed while capturing its registry. Refresh and retry.")}
            }
            let bytes=try draft.submissionData(compositionRegistry:compositionRegistry)
            guard save() else {return}
            grade=nil;submitted=nil;resultSaved=false;receipt=nil
            let identity=drafts.identity
            let accepted=measurement.start(workspace:workspace,projectID:draft.projectID,submission:bytes,runtimeVersionKey:"pytorch",compositionRegistry:compositionRegistryBytes,makePlan:{version,seed in
                if let expected=compositionInventory {
                    guard let selected=workspace.runtime,let proof=workspace.moduleCompositionDiscoveryEvidence,
                          try WorkflowRuntimeEvidence.capture(selected) == expected else {throw LearningContractError("The selected lesson runtime changed during discovery. Refresh and retry.")}
                    _ = try proof.verifiedDescriptor(for:selected,evidence:expected)
                }
                return try TensorLearningTask.plan(operation:draft.operation,runtimeVersion:version,seed:seed)
            },onComplete:{[weak self] result in
                guard let self,self.drafts.identity == identity,self.isActive else {return}
                defer {self.canvas?.objectWillChange.send()}
                self.grade=result.grade;self.submitted=bytes
                self.statusMessage=result.grade.summary
                if let plan=result.plan,let evidence=result.evidence,[.passed,.failed].contains(result.grade.status) {
                    do {
                        let receipt=try TensorLearningReceipt(projectID:draft.projectID,operation:draft.operation,mode:draft.mode,
                            submissionData:bytes,plan:plan,evidence:evidence,compositionRegistry:compositionRegistry)
                        self.receipt=receipt
                        try self.receipts.save(receipt)
                        self.resultSaved=true
                        self.statusMessage += " Verified result saved."
                    } catch {self.statusMessage += " Result could not be saved: " + error.localizedDescription}
                }
            })
            let measuring=DataLoadingLearningTask(rawValue:draft.operation) != nil ? "Measuring selected rows and groups…" : ProgramExportLearningTask(rawValue:draft.operation) != nil ? "Measuring reloaded computation…" : GradientLearningTask(rawValue:draft.operation) != nil ? "Measuring actual influence, adjustment windows, trainability and saved-parent comparisons…" : InitializationLearningTask(rawValue:draft.operation) != nil ? "Measuring starting settings, predictions, first adjustments, repeat isolation and saved-weight reuse…" : CheckpointLearningTask(rawValue:draft.operation) != nil ? "Saving the supplied run, reopening it and measuring the requested training behavior…" : LossLearningTask(rawValue:draft.operation) != nil ? "Measuring individual scores, combined objectives, typed desired roles and raw-input influence…" : RecurrentLearningTask(rawValue:draft.operation) != nil ? "Measuring sequence and final-state values, and weighted influence on supplied inputs, state and all learned tables…" : SchedulerLearningTask(rawValue:draft.operation) != nil ? "Measuring scheduled rates, learned updates and the saved-sequence continuation…" : OptimizerLearningTask(rawValue:draft.operation) != nil ? "Measuring configured training updates and remembered optimizer state…" : TensorCompositionLearningTask(rawValue:draft.operation) != nil ? "Measuring source order, output values and influence on each source…" : ConvolutionLearningTask(rawValue:draft.operation) != nil ? "Measuring shared-window outputs and influence on input, kernel and bias…" : draft.operation == LazyLinearAssessment.operation ? "Measuring inferred widths, created weights and repeat/reload state preservation…" : BilinearLearningTask(rawValue:draft.operation) != nil ? "Measuring bilinear outputs and gradients for both inputs, shared weight and active bias…" : MatrixLearningTask(rawValue:draft.operation) != nil ? "Measuring matrix products and both operand gradients…" : EmbeddingLearningTask(rawValue:draft.operation) != nil ? "Measuring gathered values, table gradients and forward table changes…" : "Measuring outputs and input gradients…"
            statusMessage=accepted ? measuring : (measurement.startFailureMessage ?? "Assessment could not start.")
        } catch {statusMessage=error.localizedDescription}
    }
    func cancel() {measurement.cancel()}
}
