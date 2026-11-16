import SwiftUI
import Combine
import KwipMLCore

enum TensorCanvasEvidenceState {case none,current,stale}

/// A project-bound adapter over the lesson source owner, using the common canvas and commands.
@MainActor final class TensorLearningCanvasAdapter:ObservableObject,CanvasWorkspace {
    let session:TensorLearningDraftSession
    private let projectID:String?
    private let sessionIdentity:UUID
    private var observation:AnyCancellable?
    private var bilinearCache:(graph:GraphDocument,task:BilinearLearningTask,projection:BilinearLearningCanvas.Projection)?
    private var gradientEditor:(nodeID:String,context:GradientPolicyInspectorContext)?
    private var trainabilityEditor:(nodeID:String,context:TrainabilityInspectorContext)?
    private var initializationEditor:(nodeID:String,context:InitializationRecipeInspectorContext)?
    private var validationCache:(graph:GraphDocument,operation:String,result:ValidationResult)?
    private struct HistoryState:Equatable {
        let editor:GraphEditorState
        var pending:[String:String]
        var parameters:[String:[String:LearningTensor]]?
        var gradientTarget:String?
    }
    private var history=EditHistory<HistoryState>()
    @Published private var selection=GraphSelection() {didSet {if oldValue != selection {initializationEditor=nil;gradientEditor=nil;trainabilityEditor=nil}}}
    var canvasViewState:CanvasViewState {
        get {session.identity == sessionIdentity && session.draft?.projectID == projectID ? session.draft?.canvasView ?? CanvasViewState():CanvasViewState()}
        set {
            guard session.identity == sessionIdentity,session.draft?.projectID == projectID else {return}
            _ = session.update {$0.canvasView=newValue}
        }
    }
    @Published var canvasViewRevision=UUID()
    @Published var pendingSourceID:String?
    @Published var statusMessage="Connect your construction. Shapes are inferred; values are measured only by assessment."

    var assessmentIsRunning:()->Bool = {false}
    var evidenceState:()->TensorCanvasEvidenceState = {.none}
    var lossProjectionProvider:()->LossFormativeProjection? = {nil}
    var attentionProjectionProvider:()->AttentionFormativeProjection? = {nil}
    init(session:TensorLearningDraftSession) {
        self.session=session;projectID=session.draft?.projectID;sessionIdentity=session.identity
        if GradientLearningTask(rawValue:session.draft?.operation ?? "") != nil {statusMessage="Connect Gradient policy, Trainability and Supplied SGD to Train. Checks measure differences from an independent reference; zero means match."}
        if InitializationLearningTask(rawValue:session.draft?.operation ?? "") != nil {statusMessage="Connect Starting settings and Supplied SGD to Train. Checks measure differences from an independent reference; zero means match."}
        if LossLearningTask(rawValue:session.draft?.operation ?? "") != nil {statusMessage="Connect produced values, desired roles and the objective recipe to Objective check. Inspect individual scores, their combined score and input influence."}
        if AttentionLearningTask(rawValue:session.draft?.operation ?? "") != nil {statusMessage="Connect the supplied sequences and Attention mask. Checks measure position results, connection weights and learned-value influence."}
        if RecurrentLearningTask(rawValue:session.draft?.operation ?? "") != nil {statusMessage="Connect the supplied measurements and starting state. Checks measure every named result and its influence."}
        if SchedulerLearningTask(rawValue:session.draft?.operation ?? "") != nil {statusMessage="Connect both recipes. Checks measure rates, learned-setting updates and the resumed sequence."}
        if OptimizerLearningTask(rawValue:session.draft?.operation ?? "") != nil {statusMessage="Connect your recipe. Checks measure learned-setting updates and remembered quantities."}
        observation=session.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()}
    }
    var historyContextID:String {"tensor-lesson:"+(projectID ?? "unavailable")+":"+sessionIdentity.uuidString}
    var historyContextIsActive:Bool {projectID != nil && session.identity == sessionIdentity && session.draft?.projectID == projectID && !isBusy}
    var isBusy:Bool {assessmentIsRunning()}
    var graph:GraphDocument {session.identity == sessionIdentity && session.draft?.projectID == projectID ? session.draft!.graph:GraphDocument()}
    private var allowsSoftmax:Bool {session.identity == sessionIdentity && session.draft?.projectID == projectID && TensorLearningTask.softmaxAxis(for:session.draft?.operation ?? "") != nil}
    private var operation:String {session.identity == sessionIdentity && session.draft?.projectID == projectID ? session.draft!.operation : ""}
    private var allowsLayerNorm:Bool {LayerNormLearningTask(rawValue:operation) != nil}
    private var gradientTask:GradientLearningTask? {GradientLearningTask(rawValue:operation)}
    var isGradientLesson:Bool {gradientTask != nil}
    private var initializationTask:InitializationLearningTask? {InitializationLearningTask(rawValue:operation)}
    var isInitializationLesson:Bool {initializationTask != nil}
    private var dataLoadingTask:DataLoadingLearningTask? {DataLoadingLearningTask(rawValue:operation)}
    var isDataLoadingLesson:Bool {dataLoadingTask != nil}
    private var optimizerTask:OptimizerLearningTask? {OptimizerLearningTask(rawValue:operation)}
    private var lossTask:LossLearningTask? {LossLearningTask(rawValue:operation)}
    var isLossLesson:Bool {lossTask != nil}
    private var attentionTask:AttentionLearningTask? {AttentionLearningTask(rawValue:operation)}
    var isAttentionLesson:Bool {attentionTask != nil}
    private var recurrentTask:RecurrentLearningTask? {RecurrentLearningTask(rawValue:operation)}
    var isRecurrentLesson:Bool {recurrentTask != nil}
    private var schedulerTask:SchedulerLearningTask? {SchedulerLearningTask(rawValue:operation)}
    var isSchedulerLesson:Bool {schedulerTask != nil}
    var isOptimizerLesson:Bool {optimizerTask != nil}
    var isTrainingControlLesson:Bool {isDataLoadingLesson || isGradientLesson || isInitializationLesson || isSchedulerLesson || isOptimizerLesson}
    func isFixedTrainingContext(_ node:GraphNode)->Bool {(isDataLoadingLesson || isGradientLesson || isInitializationLesson || isSchedulerLesson) && ["optimizer_sgd","train"].contains(node.type)}
    private var trainingControlSchemaVersion:Int {isDataLoadingLesson ? 2:isGradientLesson ? 10:isInitializationLesson ? 9:isSchedulerLesson ? 7:6}
    private var statefulNormalizationTask:StatefulNormalizationLearningTask? {StatefulNormalizationLearningTask(rawValue:operation)}
    private var spatialPatchTask:SpatialPatchLearningTask? {SpatialPatchLearningTask(rawValue:operation)}
    private var tensorRoutingTask:TensorRoutingLearningTask? {TensorRoutingLearningTask(rawValue:operation)}
    private var remainingActivationTask:RemainingActivationLearningTask? {RemainingActivationLearningTask(rawValue:operation)}
    private var bilinearTask:BilinearLearningTask? {BilinearLearningTask(rawValue:operation)}
    private var bilinearProjection:BilinearLearningCanvas.Projection? {
        guard let task=bilinearTask else {return nil}
        let current=graph
        if let cache=bilinearCache,cache.graph == current,cache.task == task {return cache.projection}
        let projection=BilinearLearningCanvas.inspect(current,task:task)
        bilinearCache=(current,task,projection)
        return projection
    }
    var axisExplanation:String {dataLoadingTask != nil ? "Each wire carries a recipe, not model values. Dataset chooses observations from the supplied training source; DataLoader arranges those observations into groups. The checker supplies the model and keeps inputs, desired outcomes and row identities together." : bagTask != nil ? "A bag is one observation’s group. Fixed layout uses rows; packed layout uses one item list with offsets marking group boundaries. IDs and offsets are whole numbers with no gradient. Real item weights can change influence. Combining returns one vector per observation, including empty groups." : gradientTask != nil ? "These wires carry gradient, trainability and optimizer instructions, with no tensor dimensions. The checker supplies two learned layers and observations. A gradient describes how a small change affects a selected result; freezing a setting prevents adjustment while input influence can still be measured." : initializationTask != nil ? "These wires carry starting-value and training instructions, with no tensor dimensions. The checker supplies two learned layers; choose their named weight or bias arrays in Configuration." : lossTask != nil ? "Direction 0 keeps observations separate. The objective recipe carries settings, not model numbers. Individual scores keep their original arrangement; the combined score is one value. Input influence describes a small change in the original produced values, before visible probability preparation." : attentionTask != nil ? "Direction 0 keeps observations separate. Sequences use position then value directions; scaled attention also has a comparison-group direction. A mask uses query position then source position: 1 blocks, 0 allows. Masks are whole numbers without learned influence." : recurrentTask != nil ? "Direction 0 keeps observations separate. Sequence measurements use time then feature directions. Starting and final sequence states use layer then hidden-width directions; a cell has only hidden-width. State never carries silently between runs." : schedulerTask != nil ? "Both wires carry training controls. The supplied SGD adjusts settings; the scheduler changes the size of its adjustments after a batch or pass. No tensor dimensions belong to these ports." : optimizerTask != nil ? "The connection carries an optimizer recipe, not model numbers. Train uses supplied starting settings and gradients; each update adjusts only parameters that have a gradient." : statefulNormalizationTask != nil ? "Direction 0 holds observations; direction 1 holds channels. BatchNorm compares channels across observations, InstanceNorm keeps each observation separate, and GroupNorm compares teams of channels within each observation." : spatialPatchTask != nil ? "Direction 0 keeps independent observations separate. Grid inputs have channel, row and column directions. Patch columns pack channel, patch row and patch column in that order; columns follow patch starts in reading order." : tensorRoutingTask != nil || compositionTask != nil ? "Axis means one array direction. Axis 0 holds independent observations; sample directions begin at 1. Negative direction numbers count from the end." : bilinearTask != nil ? "Inputs preserve observations on axis 0. Shared weights [O,H1,H2] and bias [O] have no batch axis. Layout tools preserve the first axis; negative axes count from the end." : "Batch is axis 0. Sample axes begin at 1; negative axes count from the end."}
    private var compositionTask:TensorCompositionLearningTask? {TensorCompositionLearningTask(rawValue:operation)}
    private var matrixTask:MatrixLearningTask? {MatrixLearningTask(rawValue:operation)}
    private var bagTask:EmbeddingBagLearningTask? {EmbeddingBagLearningTask(rawValue:operation)}
    private var allowsEmbedding:Bool {EmbeddingLearningTask(rawValue:operation) != nil}
    var registry:ComponentRegistry {
        if gradientTask != nil {
            var optimizer=BundledOptimizerComponents.definition("optimizer_sgd");optimizer.title="Supplied SGD"
            return .init(components:[BundledGradientControlComponents.definition("gradient_policy"),BundledGradientControlComponents.definition("trainability"),optimizer,.init(id:"train",title:"Train",category:"Training",inputs:["gradientPolicy","trainability","optimizer"],outputs:[])])
        }
        return initializationTask.map {InitializationLearningCanvas.registry(task:$0)} ?? lossTask.map {LossLearningCanvas.registry(task:$0)} ?? attentionTask.map {AttentionLearningCanvas.registry(task:$0)} ?? recurrentTask.map {RecurrentLearningCanvas.registry(task:$0)} ?? schedulerTask.map {SchedulerLearningCanvas.registry(task:$0)} ?? optimizerTask.map {OptimizerLearningCanvas.registry(task:$0)} ?? TensorLearningCanvas.registry(operation:operation)}
    var validation:ValidationResult {
        if let task=dataLoadingTask {return TensorLearningCanvas.inspectDataLoading(graph,task:task)}
        if bagTask != nil {return EmbeddingBagLearningCanvas.inspect(graph,registry:registry)}
        if let task=gradientTask {return GradientLearningCanvas.inspect(graph,task:task)}
        if let task=initializationTask {return InitializationLearningCanvas.inspect(graph,task:task)}
        if let task=lossTask {return LossLearningCanvas.inspect(graph,task:task)}
        if let task=attentionTask {return AttentionLearningCanvas.inspect(graph,task:task)}
        if let task=recurrentTask {return RecurrentLearningCanvas.inspect(graph,task:task)}
        if let task=schedulerTask {return SchedulerLearningCanvas.inspect(graph,task:task)}
        if let task=optimizerTask {return OptimizerLearningCanvas.inspect(graph,task:task)}
        if let projection=bilinearProjection {return projection.validation}
        let current=graph
        if let cache=validationCache,cache.graph == current,cache.operation == operation {return cache.result}
        let result=TensorLearningCanvas.inspect(current,allowSoftmax:allowsSoftmax,allowLayerNorm:allowsLayerNorm,allowEmbedding:allowsEmbedding,matrixTask:matrixTask,allowLazyLinear:operation == LazyLinearAssessment.operation,convolutionTask:ConvolutionLearningTask(rawValue:operation),poolingTask:PoolingLearningTask(rawValue:operation),activationTask:ConfigurableActivationLearningTask(rawValue:operation),adaptivePoolingTask:AdaptivePoolingLearningTask(rawValue:operation),compositionTask:compositionTask,remainingActivationTask:remainingActivationTask,tensorRoutingTask:tensorRoutingTask,spatialPatchTask:spatialPatchTask,statefulNormalizationTask:statefulNormalizationTask)
        validationCache=(current,operation,result)
        return result
    }
    var selectedNodeIDs:Set<String> {selection.nodeIDs}
    var selectedEdgeID:String? {selection.edgeIDs.sorted().first}
    var previewWeightLabel:String {bagTask != nil ? "Supplied category groups and shared table · checks combined values, table changes and real weight influence" : gradientTask != nil ? "Supplied two-layer model · measured influence, adjustment windows, active settings and saved-parent changes" : lossTask != nil ? "Supplied produced and desired values · individual scores, combined objectives and raw-input influence are measured" : attentionTask != nil ? (attentionTask == .scaled ? "Supplied sequences and blocked connections · no learned tables in scaled attention" : "Checker-supplied learned tables · position results and every source/table influence are measured") : recurrentTask != nil ? "Checker-supplied learned tables and starting states · every named output and input/state/parameter influence is measured" : schedulerTask != nil ? "Supplied settings, gradients and validation measurements · rates, updates and the resumed sequence are measured" : optimizerTask != nil ? "Supplied learned settings and gradients · measured updates and remembered optimizer quantities" : statefulNormalizationTask != nil ? "Supplied channel patterns · output, input influence and active learned adjustments are measured; remembered summaries belong to the checker" : spatialPatchTask != nil ? "Supplied patch data · output and input influence are measured automatically; overlapping uses add contributions" : tensorRoutingTask != nil ? "Fixture-owned sources and position selectors · output and number-source influence are measured automatically" : remainingActivationTask?.isPReLU == true ? "Explicit learned slopes · shared within each channel; values and influence are measured automatically" : remainingActivationTask != nil ? "Fixture measurements · training/evaluation behavior and influence are measured automatically" : compositionTask != nil ? "Fixture sources · values and small-change influence are measured automatically; observations stay separate" : bilinearTask != nil ? "Fixture operands and shared parameters · representative shapes; assessment varies dimensions and measures all gradients" : matrixTask != nil ? "Two fixture operands · representative shapes only; assessment varies dimensions and measures both gradients" : allowsEmbedding ? "Fixture-owned table · representative input [N,4]; assessment varies index axes and measures table gradients" : allowsLayerNorm ? "Explicit normalization parameters · assessment captures boundaries and selected parameter gradients" : "Parameter-free construction · assessment captures fixture boundaries, not intermediate tensors"}
    func previewStatus(for id:String)->String {
        guard session.identity == sessionIdentity,session.draft?.projectID == projectID,
              let node=graph.nodes.first(where:{$0.id == id}) else {return "Not measured"}
        if bagTask != nil {switch evidenceState() {
        case .none:return "Not measured"
        case .stale:return "Stale"
        case .current:
            if session.draft?.mode == .assessment {return "Private assessment · values withheld"}
            return node.type == "bag_input" ? "Supplied group rows":"Measured combined result"
        }}
        if lossTask != nil {switch evidenceState() {
        case .none:return "Not measured"
        case .stale:return "Stale"
        case .current:return session.draft?.mode == .assessment ? "Private assessment · values withheld":lossProjectionProvider() == nil ? "Not captured":node.type == "input" ? "Supplied fixture":node.type == "objective_check" || node.type == "output" ? "Measured result":"Configured recipe"
        }}
        if let task=attentionTask {
            switch evidenceState() {
            case .none:return "Not measured"
            case .stale:return "Stale"
            case .current:
                if session.draft?.mode == .assessment {return "Private assessment · values withheld"}
                if node.type == "attention_mask" {return attentionCaptureLabel(nodeID:id,port:"mask") != nil && attentionCaptureLabel(nodeID:id,port:"padding") != nil ? "Measured mask/positions":"Not captured"}
                if node.type == task.componentID {return task.outputRoles.allSatisfy {attentionCaptureLabel(nodeID:id,port:$0 == "result" ? "out":$0) != nil} ? "Measured results":"Not captured"}
                if node.type == "input" {return attentionCaptureLabel(nodeID:id,port:"out") != nil ? "Fixture input":"Not captured"}
                if node.type == "output" {return attentionCaptureLabel(nodeID:id,port:"out") != nil ? "Measured output":"Not captured"}
                return "Not captured"
            }
        }
        if isTrainingControlLesson {switch evidenceState() {case .none:return "Not measured";case .stale:return "Stale updates";case .current:return node.type == "train" ? "Measured updates":"Configured recipe"}}
        switch evidenceState() {
        case .none:return "Not measured"
        case .stale:return ["input","output"].contains(node.type) ? "Stale":"Not captured"
        case .current:
            if node.type == "input" {return "Fixture input"}
            if node.type == "output" {return "Measured output"}
            return "Not captured"
        }
    }
    func canvasSummary(_ node:GraphNode)->String {
        if bagTask != nil {
            if node.type == "bag_input",let c=try? BagSourceConfiguration(values:node.config) {return "\(c.layout.capitalized) groups · ≤\(c.maximumItemsPerBag) items"}
            if node.type == "embedding_bag",let c=try? EmbeddingBagConfiguration(values:node.config) {return "\(c.mode.capitalized) · \(c.embedding.embeddingDimension) values per group"}
            return "One vector per observation"
        }
        if gradientTask != nil {return node.type == "train" ? "Supplied training context":node.type == "optimizer_sgd" ? "Supplied SGD recipe":node.type == "gradient_policy" ? "Gradient policy recipe":"Trainability recipe"}
        if initializationTask != nil {return node.type == "train" ? "Supplied training context":node.type == "optimizer_sgd" ? "Supplied SGD recipe":"Starting settings recipe"}
        if let task=lossTask {return node.type == task.componentID ? "Objective recipe":node.type == "objective_check" ? "Scores & input influence":node.type == "log_softmax" ? "Visible probability preparation":node.type == "input" ? "Supplied "+(node.config["fixture"]?.stringValue ?? "role"):"Measured "+(node.config["role"]?.stringValue ?? "result")}
        if attentionTask != nil {return node.type == "input" ? "Supplied "+(node.config["fixture"]?.stringValue ?? "sequence") : node.type == "output" ? "Measured "+(node.config["role"]?.stringValue ?? "result") : node.type == "attention_mask" ? "Blocked connections":"Sequence comparisons"}
        if recurrentTask != nil {return node.type == "input" ? "Supplied "+(node.config["fixture"]?.stringValue ?? "source") : node.type == "output" ? "Measured "+(node.config["role"]?.stringValue ?? "result") : "Explicit sequence and state"}
        if schedulerTask != nil {return node.type == "train" ? "Supplied training context":node.type == "optimizer_sgd" ? "Supplied SGD recipe":"Scheduler recipe"}
        if optimizerTask != nil {return node.type == "train" ? "Supplied training context":"Optimizer recipe"}
        if tensorRoutingTask != nil,["split","chunk"].contains(node.type) {
            let count=validation.outputTensorContracts.keys.filter {$0.nodeID == node.id}.count
            return count == 0 ? "Shape unresolved":"\(count) parts"
        }
        if let projection=bilinearProjection {return projection.shapeDescription(node.id) ?? "Shape unresolved"}
        return validation.tensorLayouts[node.id]?.displayShape ?? "Shape unresolved"
    }
    func canvasPortDescriptor(nodeID:String,port:String,direction:PortDirection)->CanvasPortDescriptor {
        guard let node=graph.nodes.first(where:{$0.id == nodeID}),let component=registry.component(type:node.type),
              (direction == .input ? component.inputs:component.outputs).contains(port) else {return .unknown(direction:direction)}
        if dataLoadingTask != nil {return .init(kind:node.type == "dataset_source" || port == "dataset" ? .datasetRecipe:.batchRecipe,direction:direction,declared:["Configured observation recipe","No tensor dimensions","Train retains the supplied model and verified source"])}
        if bagTask != nil {return EmbeddingBagLearningCanvas.portDescriptor(graph,nodeID:nodeID,port:port,direction:direction)}
        if let task=lossTask {
            if node.type == task.componentID || (node.type == "objective_check" && direction == .input && port == "objective") {return .init(kind:.objectiveRecipe,direction:direction,declared:["Configured objective recipe","No tensor dimensions","Connect the objective recipe to Objective check"])}
            guard let p=LossLearningCanvas.presentation(graph,task:task,nodeID:nodeID,port:port,direction:direction) else {return .unknown(direction:direction,reason:"Connect supplied role to resolve shape")}
            var descriptor=CanvasPortDescriptor(kind:.tensor,direction:direction,declared:[p.displayShape,p.dtype.rawValue,p.observations ? "Independent observations":"One combined score · no observation direction"])
            if session.draft?.mode != .assessment,case .current=evidenceState(),lossProjectionProvider() != nil {descriptor.measured=["Verified measured fixture · inspect Results"]}
            return descriptor
        }
        if gradientTask != nil {
            let kind:CanvasPortKind
            if node.type == "gradient_policy" || (node.type == "train" && direction == .input && port == "gradientPolicy") {kind = .gradientPolicyRecipe}
            else if node.type == "trainability" || (node.type == "train" && direction == .input && port == "trainability") {kind = .trainabilityRecipe}
            else if node.type == "optimizer_sgd" || (node.type == "train" && direction == .input && port == "optimizer") {kind = .optimizerRecipe}
            else {return .unknown(direction:direction)}
            return .init(kind:kind,direction:direction,declared:["Configured training controls","No tensor dimensions","The checker supplies observations and native learned arrays"])
        }
        if initializationTask != nil {return .init(kind:node.type == "initialization" || port == "initialization" ? .initializationRecipe:.optimizerRecipe,direction:direction,declared:["Configured training controls","No tensor dimensions","Supplied model owners: first and second learned layers"])}
        if schedulerTask != nil {return .init(kind:node.type == schedulerTask?.componentID || port == "scheduler" ? .schedulerRecipe:.optimizerRecipe,direction:direction,declared:["Configured training controls","No tensor dimensions","The checker owns the batch, pass and validation sequence"])}
        if optimizerTask != nil {return .init(kind:.optimizerRecipe,direction:direction,declared:["Configured training controls","No tensor dimensions","The checker owns parameters and update sequence"])}
        if let task=attentionTask {
            if AttentionConfiguration.componentIDs.contains(node.type) {
                guard let config=try? AttentionConfiguration(componentID:node.type,values:node.config),
                      (direction == .input ? config.configuredInputs:config.outputPorts).contains(port) else {
                    return .unknown(direction:direction,reason:"Port disabled by lesson settings")
                }
            }
            func presented(_ descriptor:CanvasPortDescriptor)->CanvasPortDescriptor {
                var copy=descriptor
                let source=direction == .output ? GraphPortAddress(nodeID,port):graph.edges.first {$0.target == nodeID && $0.targetPort == port}.map {GraphPortAddress($0.source,$0.sourcePort)}
                if let source,let label=attentionCaptureLabel(nodeID:source.nodeID,port:source.portID) {copy.measured=[label]}
                return copy
            }
            let requirement=attentionPortRequirement(task:task,node:node,port:port,direction:direction)
            let incoming=graph.edges.first {$0.target == nodeID && $0.targetPort == port}
            let address=direction == .output ? GraphPortAddress(nodeID,port):incoming.map {GraphPortAddress($0.source,$0.sourcePort)}
            let contract=address.flatMap {validation.outputTensorContracts[$0]}
            let wired=direction == .input ? incoming != nil:graph.edges.contains {$0.target == nodeID}
            if let contract,wired {
                return presented(.init(kind:.tensor,direction:direction,role:requirement?.role,declared:["Inferred shape · "+contract.layout.displayShape,contract.dtype.rawValue,requirement?.semantic ?? "Independent observations"]))
            }
            if !wired,let requirement {
                return presented(.init(kind:.tensor,direction:direction,role:requirement.role,declared:["Required lesson shape · "+requirement.contract.layout.displayShape,requirement.contract.dtype.rawValue,requirement.semantic,"Connect the supplied roles to verify this requirement"]))
            }
            return presented(.init(kind:.tensor,direction:direction,role:requirement?.role,declared:["Shape unresolved","Type unresolved",requirement?.semantic ?? "Connect the supplied roles to resolve this port"]))
        }
        if tensorRoutingTask != nil || recurrentTask != nil {
            let address:GraphPortAddress?
            if direction == .output {address = .init(nodeID,port)}
            else {address=graph.edges.first(where:{$0.target == nodeID && $0.targetPort == port}).map {.init($0.source,$0.sourcePort)}}
            let contract=address.flatMap {validation.outputTensorContracts[$0]}
            let semantic=contract?.dtype == .int64 ? (attentionTask != nil ? "Blocked connections · 1 blocks, 0 allows · not learned values":"Whole-number positions · not learned values"):"Independent observations · floating number source"
            return .init(kind:.tensor,direction:direction,declared:[contract?.layout.displayShape ?? "Shape unresolved",contract?.dtype.rawValue ?? "Type unresolved",semantic])
        }
        if direction == .output,port == "indices",(node.type.hasPrefix("max_pool") || node.type == "adaptive_max_pool2d") {
            guard node.config["return_indices"] == .bool(true) else {return .init(kind:.tensor,direction:direction,declared:["Unavailable · enable Expose winning indices"])}
            return .init(kind:.tensor,direction:direction,declared:[validation.tensorLayouts[nodeID]?.displayShape ?? "Shape unresolved","int64","Zero-based spatial input positions within each channel"])
        }
        let source=direction == .output ? nodeID:graph.edges.first(where:{$0.target == nodeID && $0.targetPort == port})?.source
        if let projection=bilinearProjection {
            let shape=source.flatMap {projection.shapes[$0]}
            let semantic:String
            if direction == .input,node.type == "bilinear",["weight","bias"].contains(port) {semantic="Shared parameter · no batch axis"}
            else if let shape,case .sharedParameter = shape.leadingAxis {semantic="Shared parameter · no batch axis"}
            else {semantic="Independent observations"}
            return .init(kind:.tensor,direction:direction,declared:[shape?.description ?? "Shape unresolved","float64",semantic])
        }
        return .init(kind:.tensor,direction:direction,declared:[source.flatMap {validation.tensorLayouts[$0]?.displayShape} ?? "Shape unresolved",source.flatMap {validation.tensorContracts[$0]?.dtype.rawValue} ?? "Type unresolved"])
    }
    func canvasPortDescription(nodeID:String,port:String)->String {
        let direction:PortDirection=registry.component(type:graph.nodes.first {$0.id == nodeID}?.type ?? "")?.inputs.contains(port) == true ? .input:.output
        return canvasPortDescriptor(nodeID:nodeID,port:port,direction:direction).accessibilityDescription
    }
    private var state:GraphEditorState {.init(document:graph,selection:selection)}
    private func attentionCaptureLabel(nodeID:String,port:String)->String? {
        guard historyContextIsActive,let task=attentionTask,session.draft?.mode != .assessment,
              case .current=evidenceState(),let projection=attentionProjectionProvider(),!projection.fixtures.isEmpty,
              let node=graph.nodes.first(where:{$0.id == nodeID}) else {return nil}
        let key:String,label:String,isSource:Bool
        if node.type == "attention_mask",["mask","padding"].contains(port) {
            key=port == "mask" ? "blockedConnections":"blockedSourcePositions";label=port == "mask" ? "Measured blocked connections":"Measured blocked source positions";isSource=false
        } else if node.type == task.componentID,(try? AttentionConfiguration(componentID:node.type,values:node.config).outputPorts.contains(port)) == true {
            key=port == "out" ? "result":port;label=port == "out" ? "Measured position results":"Measured connection weights";isSource=false
        } else if node.type == "input",port == "out",let role=node.config["fixture"]?.stringValue,task.sourceRoles.contains(role) {
            key=role;label="Supplied fixture sequence";isSource=true
        } else if node.type == "output",port == "out",let role=node.config["role"]?.stringValue,task.outputRoles.contains(role) {
            key=role;label="Measured "+role;isSource=false
        } else {return nil}
        guard projection.fixtures.allSatisfy({fixture in
            guard let table=isSource ? fixture.inputs[key]:fixture.measured[key],table.wellFormed else {return false}
            if isSource {return true}
            guard table.shape == fixture.expected[key]?.shape else {return false}
            return task.checkpointSampleShapes[key] == nil || table.values.allSatisfy {$0 == 0 || $0 == 1}
        }) else {return nil}
        return label
    }
    /// Public task requirements are declarations, never measured fixture values.
    private func attentionPortRequirement(task:AttentionLearningTask,node:GraphNode,port:String,direction:PortDirection)->(contract:TypedTensorContract,role:String,semantic:String)? {
        var shape:[Int]?,role="",masked=false
        let labels=["query":"Query sequence","key":"Key sequence","value":"Value sequence","source":"Source sequence","target":"Requested sequence","memory":"Source memory","in":"Sequence values"]
        if node.type == "input",direction == .output,let source=node.config["fixture"]?.stringValue {
            shape=task.sampleShapes[source];role=labels[source] ?? source
        } else if node.type == "output",direction == .input,let result=node.config["role"]?.stringValue {
            shape=task.outputSampleShapes[result];role=result == "weights" ? "Connection shares":"Position mixture"
        } else if node.type == "attention_mask" {
            if direction == .input,let index=["query","key"].firstIndex(of:port) {
                shape=task.sampleShapes[task.maskReferenceRoles[index]];role=port == "query" ? "Query sequence reference":"Key sequence reference"
            } else if direction == .output {
                masked=true;shape=task.checkpointSampleShapes[port == "mask" ? "blockedConnections":"blockedSourcePositions"];role=port == "mask" ? "Blocked connections":"Blocked source positions"
            }
        } else if node.type == task.componentID {
            if direction == .output {shape=task.outputSampleShapes[port == "out" ? "result":port];role=port == "weights" ? "Connection shares":"Position mixture"}
            else if let source=task.sampleShapes[port] {shape=source;role=labels[port] ?? port}
            else {
                masked=true
                let query=task.checkpointSampleShapes["blockedConnections"]![0],source=task.checkpointSampleShapes["blockedConnections"]![1]
                if port.contains("padding") {
                    let length=port == "tgt_key_padding_mask" ? query:source
                    shape=[length];role="Blocked source positions"
                } else {
                    shape=port == "src_mask" ? [source,source]:port == "tgt_mask" ? [query,query]:[query,source];role="Blocked connections"
                }
            }
        }
        guard let shape,let contract=try? TypedTensorContract(sample:shape,dtype:masked ? .int64:.float64) else {return nil}
        return (contract,role,masked ? "Whole-number mask · 1 blocks, 0 allows · not learned values":"Independent observations · supplied sequence values")
    }
    private func result(_ command:GraphCommand)throws->GraphEditorState {
        // The shared editing envelope is schema2; persisted training-control lessons
        // retain schema6 or schema7 and their own strict validation before any source commit.
        var scratch=state
        if isTrainingControlLesson {scratch.document.schemaVersion=2}
        var after=try GraphCommandEngine.apply(command,to:scratch,registry:registry,connectionSemantics:{edge,document in
            var candidate=document;candidate.edges.append(edge)
            if self.isTrainingControlLesson {candidate.schemaVersion=self.trainingControlSchemaVersion}
            guard var draft=self.session.draft else {throw LearningContractError("Lesson session is unavailable.")}
            draft.graph=candidate;try draft.validate()
        }).after
        if isTrainingControlLesson {after.document.schemaVersion=trainingControlSchemaVersion}
        return after
    }
    private var historyState:HistoryState {.init(editor:state,pending:session.draft?.pendingFields ?? [:],parameters:session.draft?.parameters,gradientTarget:session.draft?.gradientNodeID)}
    private func install(_ proposed:GraphEditorState,record:Bool=true,pending:[String:String]?=nil,sourceState:HistoryState?=nil)->Bool {
        guard historyContextIsActive else {return false}
        let before=historyState
        let source=sourceState ?? before
        var validKeys=Set(proposed.document.nodes.flatMap {node in
            (registry.component(type:node.type)?.fields ?? []).map {pendingKey(node.id,$0.key)}
        })
        if isInitializationLesson {
            for node in proposed.document.nodes where node.type == "initialization" {validKeys.insert(pendingKey(node.id,"initializationEditor"))}
        }
        if isGradientLesson {
            for node in proposed.document.nodes {
                if node.type == "gradient_policy" {validKeys.insert(pendingKey(node.id,"gradientPolicyEditor"))}
                if node.type == "trainability" {validKeys.insert(pendingKey(node.id,"trainabilityEditor"))}
            }
        }
        let parkedIDs = Set(proposed.document.nodes.filter { statefulNormalizationTask?.componentID == $0.type }.map(\.id))
        for node in proposed.document.nodes where ownsEditableParameters(node) || parkedIDs.contains(node.id) {
            for (name,tensor) in source.parameters?[node.id] ?? [:] where name == "weight" || parkedIDs.contains(node.id) || node.config["bias"]?.boolValue == true {
                for index in tensor.values.indices {validKeys.insert(pendingKey(node.id,"parameter.\(name).\(index)"))}
            }
        }
        let retained=(pending ?? source.pending).filter {validKeys.contains($0.key)}
        let active=proposed.document.nodes.filter {ownsEditableParameters($0)}
        let ids=Set(active.map(\.id)).union(parkedIDs)
        let parameters=source.parameters.map { values in
            var retained=values.filter {ids.contains($0.key)}
            for node in active {
                if node.type == "prelu" {
                    if retained[node.id]?["weight"]?.shape != parameterShape(node) {retained.removeValue(forKey:node.id)}
                } else if !parkedIDs.contains(node.id), node.config["bias"]?.boolValue == false {retained[node.id]?.removeValue(forKey:"bias")}
            }
            return retained
        }
        let target=source.gradientTarget.flatMap {ids.contains($0) ? $0 : nil}
        guard session.update({$0.graph=proposed.document;$0.pendingFields=retained;$0.parameters=parameters;$0.gradientNodeID=target}) else {statusMessage=session.errorMessage ?? "Edit rejected.";return false}
        selection=proposed.selection
        if record {history.record(before:before,after:historyState)}
        return true
    }
    private func apply(_ command:GraphCommand) {
        guard historyContextIsActive else {return}
        do {if install(try result(command)) {pendingSourceID=nil}} catch {statusMessage=error.localizedDescription}
    }
    var canUndo:Bool {historyContextIsActive && history.canUndo}
    var canRedo:Bool {historyContextIsActive && history.canRedo}
    func undo() {
        guard canUndo else {return}
        var proposed=history
        if let value=proposed.undo(current:historyState),install(value.editor,record:false,pending:value.pending,sourceState:value) {history=proposed}
    }
    func redo() {
        guard canRedo else {return}
        var proposed=history
        if let value=proposed.redo(current:historyState),install(value.editor,record:false,pending:value.pending,sourceState:value) {history=proposed}
    }
    func selectNode(_ id:String,extending:Bool) {
        guard historyContextIsActive,graph.nodes.contains(where:{$0.id == id}) else {return}
        if extending {selection.nodeIDs.formSymmetricDifference([id]);selection.edgeIDs=[]}
        else {selection = .init(nodeIDs:[id])}
    }
    func selectEdge(_ id:String) {guard historyContextIsActive,graph.edges.contains(where:{$0.id == id}) else {return};selection = .init(edgeIDs:[id])}
    func clearSelection() {selection = .init()}
    func moveNodes(ids:Set<String>,dx:Double,dy:Double) {apply(.move(nodeIDs:ids,offset:.init(x:dx,y:dy)))}
    func duplicateSelection() {
        guard historyContextIsActive else {return}
        if lossTask != nil {statusMessage="Each supplied role, objective recipe and result appears once. Add its matching boundary to repair this lesson.";return}
        if attentionTask != nil {statusMessage="Each supplied sequence, attention block, mask and result role appears once. Add or remove its matching boundary to change this lesson.";return}
        if recurrentTask != nil {statusMessage="Each supplied and result role appears once. Add or remove its matching boundary to change this lesson.";return}
        if gradientTask != nil {statusMessage="This lesson uses one gradient policy, one trainability policy, supplied SGD and Train.";return}
        if schedulerTask != nil {statusMessage="This lesson uses one scheduler, supplied SGD and Train. Start another lesson to compare a different schedule.";return}
        if optimizerTask != nil {statusMessage="This lesson uses one optimizer recipe and one supplied Train context. Start a separate lesson to compare another recipe.";return}
        do {
            let result=try GraphCommandEngine.apply(.duplicate(nodeIDs:selection.nodeIDs,offset:.init(),semantics:.independent),to:state,registry:registry)
            var source=historyState
            for (original,copy) in result.copies.nodes {
                if let values=source.parameters?[original] {source.parameters?[copy]=values}
            }
            _ = install(result.after,sourceState:source)
        } catch {statusMessage=error.localizedDescription}
    }
    private func gradientLessonChoices()->(owners:[InitializationOwnerChoice],endpoints:[GradientEndpointChoice])? {
        guard let model=try? GradientLearningExamples.modelGraph(),let registry=try? ComponentRegistry.builtIn(),
              let raw=try? WorkflowLifecycleVerifier.optimizerParameterOwnership(model,context:.explicitObjective,capturedRegistry:registry,initializationOwnership:true) else {return nil}
        let owners:[InitializationOwnerChoice]=raw.compactMap {value in
            guard case .object(let row)=value,let node=row["node"]?.stringValue,let parameter=row["parameter"]?.stringValue,let name=row["name"]?.stringValue,case .array(let shape)=row["shape"] else {return nil}
            let dims=shape.compactMap {$0.integerValue.flatMap(Int.init(exactly:))}
            guard dims.count == shape.count else {return nil}
            return .init(node:node,parameter:parameter,name:name,shape:dims)
        }
        let validation=GraphValidator.validate(model,registry:registry,context:.explicitObjective)
        guard validation.isValid else {return nil}
        let endpoints=validation.outputTensorContracts.compactMap {address,contract -> GradientEndpointChoice? in
            guard let node=model.nodes.first(where:{$0.id == address.nodeID}),node.type != "output" else {return nil}
            return .init(node:node.id,port:address.portID,name:node.name,dtype:contract.dtype,shape:contract.layout.sample.dimensions,isModelInput:node.type == "input")
        }.sorted {$0.id < $1.id}
        return (owners,endpoints)
    }
    func gradientPolicyInspector(nodeID:String)->GradientPolicyInspectorContext? {
        guard isGradientLesson,selectedNodeIDs == [nodeID],let node=graph.nodes.first(where:{$0.id == nodeID && $0.type == "gradient_policy"}) else {return nil}
        if let cached=gradientEditor,cached.nodeID == nodeID,cached.context.expectedConfiguration == node.config {return cached.context}
        guard let choices=gradientLessonChoices() else {return nil}
        let identity=historyContextID+":"+nodeID,key=pendingKey(nodeID,"gradientPolicyEditor")
        weak var retained:GradientPolicyInspectorContext?
        let context=GradientPolicyInspectorContext(configuration:node.config,endpoints:choices.endpoints,owners:choices.owners,origin:identity,
            isCurrent:{[weak self] in guard let self else {return false};return self.historyContextIsActive && self.selectedNodeIDs == [nodeID] && self.historyContextID+":"+nodeID == identity && self.gradientEditor?.context === retained && self.graph.nodes.first(where:{$0.id == nodeID})?.config == retained?.expectedConfiguration},
            isBusy:{[weak self] in self?.isBusy ?? true},validate:{[weak self] values in
                guard let self,let task=self.gradientTask,
                      let companion=self.graph.nodes.first(where:{$0.type == "trainability"}) else {
                    throw LearningContractError("This lesson's controls are unavailable. Reopen the lesson before applying changes.")
                }
                if task == .keepSettings || task == .nextRun,
                   try GradientPolicyConfiguration(values:values) != GradientPolicyConfiguration(values:task.gradientValues) {
                    throw LearningContractError("This lesson keeps Gradient policy fixed. Restore it and configure Trainability instead.")
                }
                _ = try GradientLearningConfiguration(gradientValues:values,trainabilityValues:companion.config,task:task)
            },
            apply:{[weak self] expected,updated in guard let self,self.graph.nodes.first(where:{$0.id == nodeID})?.config == expected else {return false};return self.configure(nodeID:nodeID,values:updated)},
            pending:{[weak self] pending in
                guard let self,self.historyContextIsActive,self.selectedNodeIDs == [nodeID],self.gradientEditor?.context === retained,let context=retained else {return}
                if !pending {_ = self.session.update {$0.pendingFields.removeValue(forKey:key)};return}
                guard let data=try? context.snapshotData(),let text=String(data:data,encoding:.utf8) else {return}
                _ = self.session.update {$0.pendingFields[key]=text}
            })
        retained=context;gradientEditor=(nodeID,context)
        if let text=session.draft?.pendingFields[key],let data=text.data(using:.utf8) {_ = context.restorePending(data)}
        return context
    }
    func trainabilityInspector(nodeID:String)->TrainabilityInspectorContext? {
        guard isGradientLesson,selectedNodeIDs == [nodeID],let node=graph.nodes.first(where:{$0.id == nodeID && $0.type == "trainability"}) else {return nil}
        if let cached=trainabilityEditor,cached.nodeID == nodeID,cached.context.expectedConfiguration == node.config {return cached.context}
        guard let choices=gradientLessonChoices() else {return nil}
        let identity=historyContextID+":"+nodeID,key=pendingKey(nodeID,"trainabilityEditor")
        weak var retained:TrainabilityInspectorContext?
        let context=TrainabilityInspectorContext(configuration:node.config,owners:choices.owners,origin:identity,
            isCurrent:{[weak self] in guard let self else {return false};return self.historyContextIsActive && self.selectedNodeIDs == [nodeID] && self.historyContextID+":"+nodeID == identity && self.trainabilityEditor?.context === retained && self.graph.nodes.first(where:{$0.id == nodeID})?.config == retained?.expectedConfiguration},
            isBusy:{[weak self] in self?.isBusy ?? true},
            validate:{[weak self] values in
                guard let self,let task=self.gradientTask,
                      let companion=self.graph.nodes.first(where:{$0.type == "gradient_policy"}) else {
                    throw LearningContractError("This lesson's controls are unavailable. Reopen the lesson before applying changes.")
                }
                if task == .selectedInfluence || task == .combineSmallGroups,
                   try TrainabilityConfiguration(values:values) != TrainabilityConfiguration(values:task.trainabilityValues) {
                    throw LearningContractError("This lesson keeps Trainability fixed. Restore it and configure Gradient policy instead.")
                }
                _ = try GradientLearningConfiguration(gradientValues:companion.config,trainabilityValues:values,task:task)
            },
            apply:{[weak self] expected,updated in guard let self,self.graph.nodes.first(where:{$0.id == nodeID})?.config == expected else {return false};return self.configure(nodeID:nodeID,values:updated)},
            pending:{[weak self] pending in
                guard let self,self.historyContextIsActive,self.selectedNodeIDs == [nodeID],self.trainabilityEditor?.context === retained,let context=retained else {return}
                if !pending {_ = self.session.update {$0.pendingFields.removeValue(forKey:key)};return}
                guard let data=context.serializedDraft(),let text=String(data:data,encoding:.utf8) else {return}
                _ = self.session.update {$0.pendingFields[key]=text}
            })
        retained=context;trainabilityEditor=(nodeID,context)
        if let text=session.draft?.pendingFields[key],let data=text.data(using:.utf8) {_ = context.restorePending(data)}
        return context
    }
    /// Reuses the same typed rules form; this lesson owns only the recipe, never its supplied model.
    func initializationInspector(nodeID:String)->InitializationRecipeInspectorContext? {
        guard isInitializationLesson,selectedNodeIDs == [nodeID],let node=graph.nodes.first(where:{$0.id == nodeID && $0.type == "initialization"}) else {return nil}
        if let cached=initializationEditor,cached.nodeID == nodeID,cached.context.expectedConfiguration == node.config,!cached.context.isBusy {return cached.context}
        guard let model=try? InitializationLearningExamples.modelGraph(),let registry=try? ComponentRegistry.builtIn(),
              let ownership=try? WorkflowLifecycleVerifier.optimizerParameterOwnership(model,context:.explicitObjective,capturedRegistry:registry,initializationOwnership:true) else {return nil}
        let owners:[InitializationOwnerChoice]=ownership.compactMap {raw in
            guard case .object(let row)=raw,let owner=row["node"]?.stringValue,let parameter=row["parameter"]?.stringValue,let name=row["name"]?.stringValue,case .array(let shape)=row["shape"] else {return nil}
            let dimensions=shape.compactMap {$0.integerValue.flatMap(Int.init(exactly:))}
            guard dimensions.count == shape.count else {return nil}
            return .init(node:owner,parameter:parameter,name:name,shape:dimensions)
        }
        let identity=historyContextID+":"+nodeID,key=pendingKey(nodeID,"initializationEditor")
        weak var retained:InitializationRecipeInspectorContext?
        let context=InitializationRecipeInspectorContext(configuration:node.config,owners:owners,origin:identity,
            isCurrent:{[weak self] in
                guard let self,self.historyContextIsActive,self.selectedNodeIDs == [nodeID],self.historyContextID+":"+nodeID == identity else {return false}
                return self.initializationEditor?.context === retained && self.graph.nodes.first(where:{$0.id == nodeID})?.config == retained?.expectedConfiguration
            },isBusy:{[weak self] in self?.isBusy ?? true},
            apply:{[weak self] expected,updated in
                guard let self,self.graph.nodes.first(where:{$0.id == nodeID})?.config == expected else {return false}
                return self.configure(nodeID:nodeID,values:updated)
            },pending:{[weak self] value in
                guard let self,self.historyContextIsActive,self.selectedNodeIDs == [nodeID],let context=retained else {return}
                if !value {_ = self.session.update {$0.pendingFields.removeValue(forKey:key)};return}
                let raw:JSONValue = .object(["configuration":.object(context.draftConfiguration),"seedText":.string(context.seedText),"rawNumericDrafts":.object(context.rawNumericDrafts.mapValues(JSONValue.string))])
                guard let data=try? WorkflowArchiveCodec.encode(raw),data.count<=131072,let text=String(data:data,encoding:.utf8) else {return}
                _ = self.session.update {$0.pendingFields[key]=text}
            })
        retained=context;initializationEditor=(nodeID,context)
        if let text=session.draft?.pendingFields[key],let data=text.data(using:.utf8),data.count<=131072,
           case .object(let raw)=try? JSONDecoder().decode(JSONValue.self,from:data),Set(raw.keys)==["configuration","seedText","rawNumericDrafts"],
           case .object(let configuration)=raw["configuration"],let seed=raw["seedText"]?.stringValue,case .object(let numeric)=raw["rawNumericDrafts"],numeric.values.allSatisfy({$0.stringValue != nil}) {
            _ = context.restorePending(configuration:configuration,seedText:seed,rawNumericDrafts:numeric.compactMapValues(\.stringValue))
        }
        return context
    }
    func deleteSelection() {apply(.delete(selection))}
    func canAddType(_ type:String)->Bool {
        guard historyContextIsActive,registry.component(type:type) != nil else {return false}
        if bagTask != nil || dataLoadingTask != nil {return !graph.nodes.contains {$0.type == type}}
        if let task=lossTask {
            if type == "input" {return graph.nodes.filter {$0.type == "input"}.count<task.sourceRoles.count}
            if type == "output" {return graph.nodes.filter {$0.type == "output"}.count<3}
            return [task.componentID,"objective_check","log_softmax"].contains(type) && !graph.nodes.contains {$0.type == type}
        }
        if let task=attentionTask {
            if type == "input" {return graph.nodes.filter {$0.type == "input"}.count<task.sourceRoles.count}
            if type == "output" {return graph.nodes.filter {$0.type == "output"}.count<task.outputRoles.count}
            return [task.componentID,"attention_mask"].contains(type) && !graph.nodes.contains {$0.type == type}
        }
        if let task=recurrentTask {
            if type == "input" {return graph.nodes.filter {$0.type == "input"}.count<task.sourceRoles.count}
            if type == "output" {return graph.nodes.filter {$0.type == "output"}.count<task.outputRoles.count}
            return type == task.componentID && !graph.nodes.contains {$0.type == type}
        }
        if isTrainingControlLesson {return !graph.nodes.contains {$0.type == type}}
        if let task=bilinearTask {
            if type == "input" {return graph.nodes.filter {$0.type == "input"}.count < (task == .withBias ? 4:3)}
            if type == "bilinear" {return !graph.nodes.contains {$0.type == "bilinear"}}
        }
        if type == "input",let task=tensorRoutingTask {return graph.nodes.filter {$0.type == "input"}.count < task.sourceRoles.count}
        if type == "input",let task=compositionTask {return graph.nodes.filter {$0.type == "input"}.count < (task.isBinary ? 2:1)}
        if type == "input",matrixTask != nil {return graph.nodes.filter {$0.type == "input"}.count<2}
        return !(type == "input" || type == "output") || !graph.nodes.contains {$0.type == type}
    }
    private func node(type:String,x:Double,y:Double)->GraphNode? {
        guard canAddType(type),let component=registry.component(type:type) else {return nil}
        var config=Dictionary(uniqueKeysWithValues:component.fields.map {($0.key,$0.defaultValue)})
        var name=component.title
        if let task=lossTask {
            if type == task.componentID {config=task.configuration;name=task.title}
            if type == "log_softmax" {config=["dim":-1];name="Prepare probabilities"}
            if type == "input" {guard let role=task.sourceRoles.first(where:{r in !graph.nodes.contains {$0.type == "input" && $0.config["fixture"]?.stringValue == r}}) else {return nil};config=["fixture":.string(role)];name="Supplied "+role}
            if type == "output" {guard let role=LossLearningCanvas.resultRoles.first(where:{r in !graph.nodes.contains {$0.type == "output" && $0.config["role"]?.stringValue == r}}) else {return nil};config=["role":.string(role)];name=role == "raw" ? "Individual scores":role == "objective" ? "Combined score":"Input influence"}
        }
        if let task=attentionTask {
            if type == task.componentID {config=task.configuration;name=task.componentTitle}
            if type == "attention_mask" {config=task.maskConfiguration;name="Blocked connections"}
            if type == "input" {guard let role=task.sourceRoles.first(where:{r in !graph.nodes.contains {$0.type == "input" && $0.config["fixture"]?.stringValue == r}}) else {return nil};config=["fixture":.string(role)];name="Supplied "+role}
            if type == "output" {guard let role=task.outputRoles.first(where:{r in !graph.nodes.contains {$0.type == "output" && $0.config["role"]?.stringValue == r}}) else {return nil};config=["role":.string(role)];name=role.capitalized+" Output"}
        }
        if let task=recurrentTask {
            if type == task.componentID {config=task.configuration}
            if type == "input" {guard let role=task.sourceRoles.first(where:{r in !graph.nodes.contains {$0.type == "input" && $0.config["fixture"]?.stringValue == r}}) else {return nil};config=["fixture":.string(role)];name="Supplied "+role}
            if type == "output" {guard let role=task.outputRoles.first(where:{r in !graph.nodes.contains {$0.type == "output" && $0.config["role"]?.stringValue == r}}) else {return nil};config=["role":.string(role)];name=role.capitalized+" Output"}
        }
        if isTrainingControlLesson,type == "train" {config=[:];name="Supplied training context"}
        if let task=gradientTask {
            if type == "optimizer_sgd" {config=task.optimizerValues;name="Supplied SGD"}
            if ["gradient_policy","trainability"].contains(type) {config=(try? GradientLearningExamples.defaultGraph(task:task))?.nodes.first {$0.type == type}?.config ?? [:]}
        }
        if initializationTask != nil,type == "optimizer_sgd" {config=InitializationLearningTask.suppliedOptimizerValues;name="Supplied SGD"}
        if let task=initializationTask,type == "initialization" {config=(try? InitializationLearningExamples.defaultGraph(task:task))?.nodes.first {$0.type == "initialization"}?.config ?? [:];name="Starting settings"}
        if schedulerTask != nil,type == "optimizer_sgd" {config=SchedulerLearningTask.suppliedOptimizerValues;name="Supplied SGD"}
        if type == "input",spatialPatchTask != nil {
            // The single reviewed source is implicit; patch boundaries have no settings.
            name="Supplied patch data"
        }
        if type == "input",let task=tensorRoutingTask {
            guard let role=task.sourceRoles.first(where:{role in !graph.nodes.contains {$0.type == "input" && $0.config["fixture"]?.stringValue == role}}) else {return nil}
            config["fixture"] = .string(role)
            name=["x":"Number source","indices":"Position selectors","left":"First table","right":"Second table"][role]!
        }
        if type == "input",let task=bilinearTask {
            let roles=["left","right","weight"]+(task == .withBias ? ["bias"]:[])
            guard let role=roles.first(where:{role in !graph.nodes.contains {$0.type == "input" && $0.config["fixture"]?.stringValue == role}}) else {return nil}
            config["fixture"] = .string(role)
            name=["left":"Left operand","right":"Right operand","weight":"Shared weights","bias":"Shared bias"][role]!
        }
        if type == "input",compositionTask != nil || matrixTask != nil {
            let hasLeft=graph.nodes.contains {$0.type == "input" && $0.config["fixture"]?.stringValue == "left"}
            config["fixture"] = .string(hasLeft ? "right":"left")
            name=compositionTask != nil ? (hasLeft ? "Second source":"First source") : (hasLeft ? "Right operand":"Left operand")
        }
        if dataLoadingTask != nil,type == "dataset_source" {guard let canonical=try? DatasetConfiguration(values:config).values else {return nil};config=canonical}
        if dataLoadingTask != nil,type == "data_loader" {guard let canonical=try? DataLoaderConfiguration(values:config).values else {return nil};config=canonical}
        return .init(type:type,name:name,x:x,y:y,config:config)
    }
    func addNode(type:String,x:Double,y:Double) {
        guard let node=node(type:type,x:x,y:y) else {return}
        var proposed=state;proposed.document.nodes.append(node);proposed.selection = .init(nodeIDs:[node.id]);_ = install(proposed)
    }
    func canConnect(source:String,sourcePort:String,target:GraphNode,port:String)->Bool {
        historyContextIsActive && (try? result(.connect(source:.init(source,sourcePort),target:.init(target.id,port)))) != nil
    }
    func connect(source:String,sourcePort:String,target:String,targetPort:String) {apply(.connect(source:.init(source,sourcePort),target:.init(target,targetPort)))}
    func reconnect(edgeID:String,source:String,sourcePort:String,target:String,targetPort:String) {apply(.reconnect(edgeID:edgeID,source:.init(source,sourcePort),target:.init(target,targetPort)))}
    func insertNode(type:String,edgeID:String,x:Double,y:Double) {
        guard let node=node(type:type,x:x,y:y) else {return}
        apply(.insertOnWire(edgeID:edgeID,node:node,inputPort:"in",outputPort:"out"))
    }
    @discardableResult func configureGradientPolicy(nodeID:String,configuration:GradientPolicyConfiguration)->Bool {
        guard isGradientLesson,graph.nodes.first(where:{$0.id == nodeID})?.type == "gradient_policy" else {return false}
        return configure(nodeID:nodeID,values:configuration.values)
    }
    @discardableResult func configureTrainability(nodeID:String,configuration:TrainabilityConfiguration)->Bool {
        guard isGradientLesson,graph.nodes.first(where:{$0.id == nodeID})?.type == "trainability" else {return false}
        return configure(nodeID:nodeID,values:configuration.values)
    }
    @discardableResult func setTensorOperation(_ type:String,nodeID:String,expectedType:String,expected:[String:JSONValue])->Bool {
        guard historyContextIsActive,selectedNodeIDs == [nodeID],let index=graph.nodes.firstIndex(where:{$0.id==nodeID}),
              graph.nodes[index].type==expectedType,graph.nodes[index].config==expected else {return false}
        guard !(session.draft?.pendingFields.keys.contains(where:{$0.hasPrefix(pendingKey(nodeID,""))}) ?? false) else {statusMessage="Apply or discard this field draft before changing method.";return false}
        guard type != expectedType else {return true}
        do {
            let values=try TensorOperationAuthoring.defaults(type,registry:registry)
            let changed=try TensorOperationAuthoring.switching(graph.nodes[index],configurationVersion:2,to:type,configuration:values,
                connectedInputs:Set(graph.edges.filter {$0.target==nodeID}.map(\.targetPort)),connectedOutputs:Set(graph.edges.filter {$0.source==nodeID}.map(\.sourcePort)),
                inputs:TensorOperationAuthoring.incoming(nodeID:nodeID,graph:graph,validation:validation),registry:registry)
            var proposed=state;proposed.document.nodes[index]=changed;return install(proposed)
        } catch {statusMessage=error.localizedDescription;return false}
    }
    @discardableResult func setReshape(_ variant:ReshapeAuthoringConfiguration.Variant,nodeID:String,expectedType:String,expected:[String:JSONValue])->Bool {
        guard historyContextIsActive,selectedNodeIDs == [nodeID],let index=graph.nodes.firstIndex(where:{$0.id==nodeID}),
              graph.nodes[index].type==expectedType,graph.nodes[index].config==expected else {return false}
        guard !(session.draft?.pendingFields.keys.contains(where:{$0.hasPrefix(pendingKey(nodeID,""))}) ?? false) else {
            statusMessage="Apply or discard this component's field draft before changing method.";return false
        }
        do {
            let changed=try ReshapeAuthoringConfiguration.switching(graph.nodes[index],configurationVersion:2,to:variant,registry:registry)
            let incoming=graph.edges.filter {$0.target==nodeID && $0.targetPort=="in"}
            if incoming.count==1,let edge=incoming.first,let input=validation.tensorContracts[edge.source]?.layout {
                _=try ReshapeAuthoringConfiguration.outputLayout(variant:variant,configuration:changed.config,input:input)
            }
            var proposed=state;proposed.document.nodes[index]=changed
            return install(proposed)
        } catch {statusMessage=error.localizedDescription;return false}
    }
    @discardableResult func configure(nodeID:String,values:[String:JSONValue])->Bool {
        guard historyContextIsActive,let index=graph.nodes.firstIndex(where:{$0.id == nodeID}) else {return false}
        if isFixedTrainingContext(graph.nodes[index]) {statusMessage="The checker supplies these public controls. Configure the scheduler to change the rate sequence.";return false}
        if let task=bilinearTask,graph.nodes[index].type == "bilinear",values["bias"] != .bool(task == .withBias) {
            statusMessage="Bias mode belongs to this task. Start the other Bilinear lesson to change modes.";return false
        }
        var proposed=state
        if dataLoadingTask != nil,graph.nodes[index].type == "dataset_source" {
            do {proposed.document.nodes[index].config=try DatasetConfiguration(values:values).values}
            catch {statusMessage=error.localizedDescription;return false}
        } else if dataLoadingTask != nil,graph.nodes[index].type == "data_loader" {
            do {proposed.document.nodes[index].config=try DataLoaderConfiguration(values:values).values}
            catch {statusMessage=error.localizedDescription;return false}
        } else if gradientTask != nil,graph.nodes[index].type == "gradient_policy" {
            do {proposed.document.nodes[index].config=try GradientPolicyConfiguration(values:values).values}
            catch {statusMessage=error.localizedDescription;return false}
        } else if gradientTask != nil,graph.nodes[index].type == "trainability" {
            do {proposed.document.nodes[index].config=try TrainabilityConfiguration(values:values).values}
            catch {statusMessage=error.localizedDescription;return false}
        } else if initializationTask != nil,graph.nodes[index].type == "initialization" {
            do {proposed.document.nodes[index].config=try InitializationLearningConfiguration(values:values).values}
            catch {statusMessage=error.localizedDescription;return false}
        } else if let task=lossTask,graph.nodes[index].type == task.componentID {
            do {proposed.document.nodes[index].config=try LossConfiguration(componentID:task.componentID,values:values).values}
            catch {statusMessage=error.localizedDescription;return false}
        } else if let task=attentionTask,[task.componentID,"attention_mask"].contains(graph.nodes[index].type) {
            do {proposed.document.nodes[index].config=try AttentionConfiguration(componentID:graph.nodes[index].type,values:values).values}
            catch {statusMessage=error.localizedDescription;return false}
        } else if let task=recurrentTask,graph.nodes[index].type == task.componentID {
            do {proposed.document.nodes[index].config=try RecurrentConfiguration(componentID:task.componentID,values:values).values}
            catch {statusMessage=error.localizedDescription;return false}
        } else if let task=schedulerTask,graph.nodes[index].type == task.componentID {
            do {proposed.document.nodes[index].config=try SchedulerConfiguration(componentID:task.componentID,values:values).values}
            catch {statusMessage=error.localizedDescription;return false}
        } else if let task=optimizerTask,graph.nodes[index].type == task.componentID {
            do {proposed.document.nodes[index].config=try OptimizerConfiguration(componentID:task.componentID,values:values).values}
            catch {statusMessage=error.localizedDescription;return false}
        } else if let task=statefulNormalizationTask,graph.nodes[index].type == task.componentID {
            do {proposed.document.nodes[index].config=try StatefulNormalizationConfiguration(componentID:task.componentID,values:values).values}
            catch {statusMessage=error.localizedDescription;return false}
        } else if let task=spatialPatchTask,graph.nodes[index].type == task.componentID {
            do {proposed.document.nodes[index].config=try SpatialPatchConfiguration(componentID:task.componentID,values:values).values}
            catch {statusMessage=error.localizedDescription;return false}
        } else if let task=tensorRoutingTask,graph.nodes[index].type == task.componentID {
            do {proposed.document.nodes[index].config=try TensorRoutingConfiguration(componentID:task.componentID,values:values).values}
            catch {statusMessage=error.localizedDescription;return false}
        } else {proposed.document.nodes[index].config=values}
        guard install(proposed) else {return false}
        statusMessage="Configuration applied."
        return true
    }
    private func ownsEditableParameters(_ node:GraphNode)->Bool {
        (statefulNormalizationTask?.componentID == node.type && node.config["affine"]?.boolValue == true)
            || (allowsLayerNorm && node.type == "layer_norm" && node.config["elementwise_affine"]?.boolValue == true)
            || (remainingActivationTask?.isPReLU == true && node.type == "prelu")
    }
    private func parameterShape(_ node:GraphNode)->[Int]? {
        guard ownsEditableParameters(node) else {return nil}
        if StatefulNormalizationConfiguration.componentIDs.contains(node.type) {return (try? StatefulNormalizationConfiguration(componentID:node.type,values:node.config)).map {[$0.channelCount]}}
        if node.type == "prelu" {return (try? RemainingActivationConfiguration(componentID:node.type,values:node.config)).flatMap {$0.numParameters.map {[$0]}}}
        return (try? LayerNormConfiguration(values:node.config))?.normalizedShape
    }
    @discardableResult func setNormalizationParameters(nodeID:String,values:[String:LearningTensor])->Bool {
        guard historyContextIsActive,let node=graph.nodes.first(where:{$0.id == nodeID}),let shape=parameterShape(node) else {return false}
        let keys:Set<String>=(node.type == "layer_norm" || StatefulNormalizationConfiguration.componentIDs.contains(node.type)) && node.config["bias"]?.boolValue == true ? ["weight","bias"]:["weight"]
        guard Set(values.keys) == keys,values.values.allSatisfy({$0.wellFormed && $0.shape == shape}) else {return false}
        var source=historyState
        if source.parameters == nil {source.parameters=[:]}
        source.parameters?[nodeID]=values
        return install(state,sourceState:source)
    }
    @discardableResult func setGradientTarget(_ nodeID:String?)->Bool {
        guard historyContextIsActive,allowsLayerNorm || statefulNormalizationTask != nil || remainingActivationTask?.isPReLU == true else {return false}
        if let nodeID {guard graph.nodes.contains(where:{$0.id == nodeID && ownsEditableParameters($0)}) else {return false}}
        var source=historyState;source.gradientTarget=nodeID
        return install(state,sourceState:source)
    }
    @discardableResult func initializeNormalizationParameters(nodeID:String)->Bool {
        guard historyContextIsActive,let node=graph.nodes.first(where:{$0.id == nodeID}),let shape=parameterShape(node) else {return false}
        let count=shape.reduce(1,*)
        guard count<=4096 else {statusMessage="Lesson parameter editing supports at most 4096 values per tensor.";return false}
        let initial=node.type == "prelu" ? (try? RemainingActivationConfiguration(componentID:node.type,values:node.config))?.initialSlope ?? 0.25:1
        var values=["weight":LearningTensor(shape:shape,values:Array(repeating:initial,count:count))]
        if (node.type == "layer_norm" || StatefulNormalizationConfiguration.componentIDs.contains(node.type)),node.config["bias"]?.boolValue == true {values["bias"] = .init(shape:shape,values:Array(repeating:0,count:count))}
        var source=historyState
        if source.parameters == nil {source.parameters=[:]}
        source.parameters?[nodeID]=values
        source.pending=source.pending.filter {!$0.key.hasPrefix(pendingKey(nodeID,"parameter."))}
        return install(state,sourceState:source)
    }
    /// Resolve from the live draft: native fields may retain their binding after apply or undo.
    func parameterTextBinding(nodeID:String,name:String,index:Int)->Binding<String> {
        Binding(get:{
            if let pending=self.pendingParameter(nodeID:nodeID,name:name,index:index) {return pending}
            guard let tensor=self.session.draft?.parameters?[nodeID]?[name],
                  tensor.values.indices.contains(index) else {return ""}
            return String(tensor.values[index])
        },set:{text in
            guard self.selectedNodeIDs == [nodeID] else {return}
            _ = self.setPendingParameter(nodeID:nodeID,name:name,index:index,text:text)
        })
    }
    func pendingParameter(nodeID:String,name:String,index:Int)->String? {
        session.draft?.pendingFields[pendingKey(nodeID,"parameter.\(name).\(index)")]
    }
    @discardableResult func setPendingParameter(nodeID:String,name:String,index:Int,text:String)->Bool {
        guard historyContextIsActive,let tensor=session.draft?.parameters?[nodeID]?[name],tensor.values.indices.contains(index) else {return false}
        let key=pendingKey(nodeID,"parameter.\(name).\(index)")
        return session.update {
            if text == String(tensor.values[index]) {$0.pendingFields.removeValue(forKey:key)}
            else {$0.pendingFields[key]=text}
        }
    }
    func discardPendingParameter(nodeID:String,name:String,index:Int) {
        guard historyContextIsActive else {return}
        _ = session.update {$0.pendingFields.removeValue(forKey:pendingKey(nodeID,"parameter.\(name).\(index)"))}
    }
    @discardableResult func applyPendingParameter(nodeID:String,name:String,index:Int)->Bool {
        guard historyContextIsActive,let text=pendingParameter(nodeID:nodeID,name:name,index:index),
              let value=Double(text.trimmingCharacters(in:.whitespacesAndNewlines)),value.isFinite,
              let node=graph.nodes.first(where:{$0.id == nodeID}),let shape=parameterShape(node),
              name == "weight" || (name == "bias" && (node.type == "layer_norm" || StatefulNormalizationConfiguration.componentIDs.contains(node.type)) && node.config["bias"]?.boolValue == true),
              var tensor=session.draft?.parameters?[nodeID]?[name],tensor.values.indices.contains(index),tensor.shape == shape else {
            statusMessage="Enter a finite value for an active parameter with the current trailing shape.";return false
        }
        tensor.values[index]=value
        var source=historyState;source.parameters?[nodeID]?[name]=tensor
        source.pending.removeValue(forKey:pendingKey(nodeID,"parameter.\(name).\(index)"))
        let accepted=install(state,sourceState:source)
        if accepted {statusMessage="Parameter value applied."}
        return accepted
    }
    private func pendingKey(_ nodeID:String,_ field:String)->String {
        // Length-prefix avoids collisions with user-authored identifiers.
        "\(nodeID.utf8.count):\(nodeID):\(field)"
    }
    static func displayValue(_ value:JSONValue)->String {
        if case .array(let values)=value {
            if values.allSatisfy({$0.integerValue != nil}) {return values.compactMap(\.integerValue).map(String.init).joined(separator:", ")}
            return RegistryNumberListParsing.display(value)
        }
        if let boolean=value.boolValue {return boolean ? "true":"false"}
        if let integer=value.integerValue {return String(integer)}
        if let string=value.stringValue {return string}
        return value.doubleValue.map {String($0)} ?? ""
    }
    /// Native text fields can retain a binding across graph updates and focus changes.
    /// Resolve accepted values from the source owner instead of capturing a node snapshot.
    func configurationTextBinding(nodeID:String,field:String)->Binding<String> {
        Binding(get:{
            if let pending=self.pendingText(nodeID:nodeID,field:field) {return pending}
            guard let node=self.graph.nodes.first(where:{$0.id == nodeID}),
                  let definition=self.registry.component(type:node.type)?.fields.first(where:{$0.key == field}) else {return ""}
            return Self.displayValue(node.config[field] ?? definition.defaultValue)
        },set:{_ = self.setPending(nodeID:nodeID,field:field,text:$0)})
    }
    func pendingText(nodeID:String,field:String)->String? {session.draft?.pendingFields[pendingKey(nodeID,field)]}
    @discardableResult func setPending(nodeID:String,field:String,text:String)->Bool {
        if let node=graph.nodes.first(where:{$0.id == nodeID}),isFixedTrainingContext(node) {return false}
        guard historyContextIsActive,let node=graph.nodes.first(where:{$0.id == nodeID}),
              let definition=registry.component(type:node.type)?.fields.first(where:{$0.key == field}) else {return false}
        let unchanged=text == Self.displayValue(node.config[field] ?? definition.defaultValue)
        let accepted=session.update {
            if unchanged {$0.pendingFields.removeValue(forKey:pendingKey(nodeID,field))}
            else {$0.pendingFields[pendingKey(nodeID,field)]=text}
        }
        if !accepted {statusMessage=session.errorMessage ?? "Field text could not be retained."}
        return accepted
    }
    func discardPending(nodeID:String,field:String) {
        guard historyContextIsActive else {return}
        let accepted=session.update {$0.pendingFields.removeValue(forKey:pendingKey(nodeID,field))}
        if accepted {statusMessage="Unapplied change discarded."}
    }
    @discardableResult func setOptionalEnabled(nodeID:String,field:String,enabled:Bool)->Bool {
        guard historyContextIsActive,let node=graph.nodes.first(where:{$0.id == nodeID}),
              let definition=registry.component(type:node.type)?.fields.first(where:{$0.key == field}),definition.allowsNull else {return false}
        guard definition.kind != .numberList || pendingText(nodeID:nodeID,field:field) == nil else {statusMessage="Apply or discard this list before changing its optional setting.";return false}
        var values=node.config
        values[field]=enabled ? (values[field] == .null ? definition.valueWhenEnabled : values[field] ?? definition.valueWhenEnabled) : .null
        guard configure(nodeID:nodeID,values:values) else {return false}
        discardPending(nodeID:nodeID,field:field)
        return true
    }
    @discardableResult func applyPending(nodeID:String,field:String)->Bool {
        guard historyContextIsActive,
              let node=graph.nodes.first(where:{$0.id == nodeID}),
              let definition=registry.component(type:node.type)?.fields.first(where:{$0.key == field}) else {return false}
        // Restoring the committed text removes its pending difference. An explicit
        // Apply/Return still validates that current value and acknowledges repair.
        let text=pendingText(nodeID:nodeID,field:field) ?? Self.displayValue(node.config[field] ?? definition.defaultValue)
        let value:JSONValue
        if definition.kind == .numberList {
            guard let parsed=RegistryNumberListParsing.parse(text,field:definition) else {statusMessage="Enter complete comma-separated finite numbers within the field bounds.";return false}
            value=parsed
        } else if definition.kind == .integerList {
            let trimmed=text.trimmingCharacters(in:.whitespacesAndNewlines)
            // An empty attention-mask list means no additional blocked positions.
            // Other lesson list controls keep their established admission policy.
            let parts = attentionTask != nil && node.type == "attention_mask" && trimmed.isEmpty
                ? [] : text.split(separator:",",omittingEmptySubsequences:false)
            let integers=parts.compactMap {Int64($0.trimmingCharacters(in:.whitespacesAndNewlines))}
            guard integers.count == parts.count else {statusMessage="Enter complete comma-separated integers.";return false}
            value = .array(integers.map(JSONValue.integer))
        } else if definition.kind == .number {
            guard let number=Double(text.trimmingCharacters(in:.whitespacesAndNewlines)),number.isFinite else {statusMessage="Enter a finite number.";return false}
            value = .number(number)
        } else if definition.kind == .choice || definition.kind == .text {
            value = .string(text)
        } else if definition.kind == .boolean {
            guard let boolean=Bool(text.trimmingCharacters(in:.whitespacesAndNewlines)) else {statusMessage="Enter true or false.";return false}
            value = .bool(boolean)
        } else {
            guard let integer=Int64(text.trimmingCharacters(in:.whitespacesAndNewlines)) else {statusMessage="Enter a complete integer.";return false}
            value = .integer(integer)
        }
        guard definition.accepts(value) else {statusMessage="The value is outside this field's bounds.";return false}
        var values=node.config;values[field]=value
        guard configure(nodeID:nodeID,values:values) else {return false}
        discardPending(nodeID:nodeID,field:field)
        statusMessage="Configuration applied."
        return true
    }
    func rename(nodeID:String,to name:String) {
        guard historyContextIsActive,name.count<=256,let index=graph.nodes.firstIndex(where:{$0.id == nodeID}) else {return}
        var proposed=state;proposed.document.nodes[index].name=name;_ = install(proposed)
    }

}
