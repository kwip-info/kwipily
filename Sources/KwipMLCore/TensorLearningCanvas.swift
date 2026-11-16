import Foundation

/// Editor contracts for fixture-bound tensor exercises. Boundary dimensions come from
/// the task, never the production classifier's features/label configuration.
public enum TensorLearningCanvas {
    // Bundled declarations are immutable for this app process. Avoid decoding the
    // full registry again for every node/port while a bag lesson is rendered.
    private static let publishedBagRegistry:ComponentRegistry = {
        guard let published=try? ComponentRegistry.builtIn(),let restricted=try? EmbeddingBagLearningCanvas.registry(from:published) else {return .init(components:[])}
        return restricted
    }()

    public static var registry:ComponentRegistry {
        return .init(components:[
            .init(id:"input",title:"Input",category:"Boundaries",outputs:["out"]),
            TensorLayoutComponent.definition("reshape"),
            TensorLayoutComponent.definition("transpose"),
            TensorLayoutComponent.definition("flatten"),
            .init(id:"output",title:"Model Output",category:"Boundaries",inputs:["in"])
        ])
    }

    public static func registry(operation:String)->ComponentRegistry {
        if DataLoadingLearningTask(rawValue:operation) != nil {
            return .init(components:[BundledDataLoadingComponents.entries["dataset_source"]!.definition,BundledDataLoadingComponents.entries["data_loader"]!.definition,.init(id:"train",title:"Supplied model and source",category:"Training",inputs:["dataset","loader"],outputs:[])])
        }
        if EmbeddingBagLearningTask(rawValue:operation) != nil {
            return publishedBagRegistry
        }
        if let task=BilinearLearningTask(rawValue:operation) {return BilinearLearningCanvas.registry(task:task)}
        var result=registry
        if let task=StatefulNormalizationLearningTask(rawValue:operation) {result.components.insert(BundledNormalizationComponents.definition(task.componentID),at:result.components.count-1)}
        if let task=SpatialPatchLearningTask(rawValue:operation) {result.components.insert(BundledSpatialPatchComponents.definition(task.componentID),at:result.components.count-1)}
        if let task=TensorRoutingLearningTask(rawValue:operation) {
            result.components.insert(BundledTensorRoutingComponents.definition(task.componentID),at:result.components.count-1)
            if task.kind == "split" {result.components.insert(BundledTensorCompositionComponents.definition("concat"),at:result.components.count-1)}
        }
        if let task=RemainingActivationLearningTask(rawValue:operation) {result.components.insert(BundledActivationComponents.definition(task.componentID),at:result.components.count-1)}
        if let task=TensorCompositionLearningTask(rawValue:operation) {result.components.insert(BundledTensorCompositionComponents.definition(task.componentID),at:result.components.count-1)}
        if let task=AdaptivePoolingLearningTask(rawValue:operation) {result.components.insert(BundledPoolingComponents.definition(task.componentID),at:result.components.count-1)}
        if let task=ConfigurableActivationLearningTask(rawValue:operation) {result.components.insert(BundledActivationComponents.definition(task.componentID),at:result.components.count-1)}
        if let task=PoolingLearningTask(rawValue:operation) {result.components.insert(BundledPoolingComponents.definition(task.componentID),at:result.components.count-1)}
        if let task=ConvolutionLearningTask(rawValue:operation) {
            let definition=task.isTransposed
                ? (task.dimensions == 1 ? ConvTranspose1dComponent.definition : task.dimensions == 2 ? ConvTranspose2dComponent.definition : ConvTranspose3dComponent.definition)
                : (task == .one ? Conv1dComponent.definition : task == .two ? Conv2dComponent.definition : Conv3dComponent.definition)
            result.components.insert(definition,at:result.components.count-1)
        }
        if operation == LazyLinearAssessment.operation {result.components.insert(LazyLinearComponent.definition,at:result.components.count-1)}
        if MatrixLearningTask(rawValue:operation) != nil {result.components.insert(MatrixMultiplyComponent.definition,at:result.components.count-1)}
        if TensorLearningTask.softmaxAxis(for:operation) != nil {
            result.components.insert(.init(id:"softmax",title:"Softmax",category:"Activations",inputs:["in"],outputs:["out"],fields:[.init(key:"dim",title:"Normalization axis",defaultValue:-1,min:-8,max:7)]),at:result.components.count-1)
        }
        if LayerNormLearningTask(rawValue:operation) != nil {result.components.insert(LayerNormComponent.definition,at:result.components.count-1)}
        if EmbeddingLearningTask(rawValue:operation) != nil {result.components.insert(EmbeddingComponent.definition,at:result.components.count-1)}
        return result
    }

    /// Structural diagnostics and inferred shapes only; this never asserts task correctness.
    /// Layout lessons show sample [2,3]; lookup lessons show representative [4].
    /// Assessment also runs other index ranks; these labels do not expose private fixtures.
    public static func inspectDataLoading(_ graph:GraphDocument,task:DataLoadingLearningTask)->ValidationResult {
        var result=ValidationResult()
        do {_=try DataLoadingLearningSubmission(task:task,graph:graph)}
        catch {result.issues.append(.init(id:"lesson.dataLoading",message:error.localizedDescription))}
        result.orderedNodeIDs=graph.nodes.filter {$0.type != "train"}.map(\.id)+graph.nodes.filter {$0.type == "train"}.map(\.id)
        return result
    }

    public static func inspect(_ graph:GraphDocument,allowSoftmax:Bool=false,allowLayerNorm:Bool=false,allowEmbedding:Bool=false,matrixTask:MatrixLearningTask?=nil,allowLazyLinear:Bool=false,convolutionTask:ConvolutionLearningTask?=nil,poolingTask:PoolingLearningTask?=nil,activationTask:ConfigurableActivationLearningTask?=nil,adaptivePoolingTask:AdaptivePoolingLearningTask?=nil,compositionTask:TensorCompositionLearningTask?=nil,remainingActivationTask:RemainingActivationLearningTask?=nil,tensorRoutingTask:TensorRoutingLearningTask?=nil,spatialPatchTask:SpatialPatchLearningTask?=nil,statefulNormalizationTask:StatefulNormalizationLearningTask?=nil)->ValidationResult {
        if let task=tensorRoutingTask {return inspectRouting(graph,task:task)}
        var result=ValidationResult()
        func issue(_ message:String,_ nodeID:String?=nil) {
            guard !result.issues.contains(where:{$0.message == message && $0.nodeID == nodeID}) else {return}
            result.issues.append(.init(id:"tensor.lesson.\(result.issues.count)",message:message,nodeID:nodeID))
        }
        func validate(_ complete:Bool)throws {
            if let task=statefulNormalizationTask {_ = try StatefulNormalizationLearningSubmission.validatedGraph(graph,task:task,complete:complete)}
            else if let task=spatialPatchTask {_ = try SpatialPatchLearningSubmission.validatedGraph(graph,task:task,complete:complete)}
            else if let task=remainingActivationTask {_ = try RemainingActivationLearningSubmission.validatedGraph(graph,task:task,complete:complete)}
            else if let task=compositionTask {_ = try TensorCompositionLearningSubmission.validatedGraph(graph,task:task,complete:complete)}
            else if let task=adaptivePoolingTask {_ = try AdaptivePoolingLearningSubmission.validatedGraph(graph,task:task,complete:complete)}
            else if let task=activationTask {_ = try ConfigurableActivationLearningSubmission.validatedGraph(graph,task:task,complete:complete)}
            else if let task=poolingTask {_ = try PoolingLearningSubmission.validatedGraph(graph,task:task,complete:complete)}
            else if let task=convolutionTask {_ = try ConvolutionLearningSubmission.validatedGraph(graph,task:task,complete:complete)}
            else if allowLazyLinear {_ = try LazyLinearLearningSubmission.validatedGraph(graph,complete:complete)}
            else if matrixTask != nil {_ = try MatrixLearningSubmission.validatedGraph(graph,complete:complete)}
            else {_ = try TensorLearningSubmission.validatedGraph(graph,complete:complete,allowSoftmax:allowSoftmax,allowLayerNorm:allowLayerNorm,allowEmbedding:allowEmbedding)}
        }
        do {try validate(false)} catch {issue(error.localizedDescription);return result}
        do {try validate(true)} catch {issue(error.localizedDescription)}
        if allowEmbedding {
            do {_ = try EmbeddingLearningSubmission(graph:graph)} catch {issue(error.localizedDescription)}
        }
        let nodes=Dictionary(uniqueKeysWithValues:graph.nodes.map {($0.id,$0)})
        let incoming=Dictionary(grouping:graph.edges,by:\.target)
        let outgoing=Dictionary(grouping:graph.edges,by:\.source)
        var pending=Dictionary(uniqueKeysWithValues:graph.nodes.map {($0.id,incoming[$0.id]?.count ?? 0)})
        var queue=graph.nodes.filter {pending[$0.id] == 0}.map(\.id)
        var cursor=0
        while cursor<queue.count {
            let id=queue[cursor];cursor+=1
            guard let node=nodes[id] else {continue}
            result.orderedNodeIDs.append(id)
            do {
                let layout:BatchedTensorLayout?
                var dtype:ContractDType = allowEmbedding ? .int64 : .float64
                if node.type == "input" {
                    if let task=compositionTask {
                        let right=node.config["fixture"]?.stringValue == "right"
                        layout=try BatchedTensorLayout(sample:right ? (task.rightSampleShape ?? task.leftSampleShape):task.leftSampleShape)
                    } else if let matrixTask {
                        let left=node.config["fixture"]?.stringValue == "left"
                        layout=try BatchedTensorLayout(sample:left ? (matrixTask == .broadcast ? [2,2,3]:[2,3]):[3,4])
                    } else {layout=try BatchedTensorLayout(sample:statefulNormalizationTask?.sampleShape ?? spatialPatchTask?.sampleShape ?? remainingActivationTask?.sampleShape ?? adaptivePoolingTask?.sampleShape ?? activationTask?.sampleShape ?? poolingTask?.sampleShape ?? convolutionTask?.sampleShape ?? (allowEmbedding ? [4] : [2,3]))}
                }
                else if let task=compositionTask,node.type == task.componentID {
                    let ports=task.isBinary ? ["a","b"]:["in"]
                    let layouts=ports.compactMap {port in incoming[id]?.first(where:{$0.targetPort == port}).flatMap {result.tensorContracts[$0.source]?.layout}}
                    layout=layouts.count == ports.count ? try TensorCompositionConfiguration(componentID:task.componentID,values:node.config).resolved(layouts:layouts):nil
                }
                else if node.type == "matmul",matrixTask != nil {
                    func operand(_ port:String)->TypedTensorContract? {
                        incoming[id]?.first(where:{$0.targetPort == port}).flatMap {result.tensorContracts[$0.source]}
                    }
                    if let a=operand("a"),let b=operand("b") {layout=try a.matrixMultiplied(by:b).layout}
                    else {layout=nil}
                }
                else if let edge=incoming[id]?.first,let input=result.tensorContracts[edge.source]?.layout {
                    dtype=result.tensorContracts[edge.source]!.dtype
                    if node.type == "output" {layout=input}
                    else if node.type == "embedding" {
                        let contract=try result.tensorContracts[edge.source]!.embedding(EmbeddingConfiguration(values:node.config),weightType:.float64)
                        layout=contract.layout;dtype=contract.dtype
                    }
                    else if let task=statefulNormalizationTask,node.type == task.componentID {
                        layout=try StatefulNormalizationConfiguration(componentID:task.componentID,values:node.config).resolved(layout:input)
                    }
                    else if let task=spatialPatchTask,node.type == task.componentID {
                        layout=try SpatialPatchConfiguration(componentID:task.componentID,values:node.config).resolved(layout:input)
                    }
                    else if let task=remainingActivationTask,node.type == task.componentID {
                        layout=try RemainingActivationConfiguration(componentID:task.componentID,values:node.config).resolved(layout:input)
                    }
                    else if let task=adaptivePoolingTask,node.type == task.componentID {
                        layout=task == .averageTwo ? try AvgPool2dConfiguration(values:node.config).resolved(input:input).outputLayout : try AdaptivePoolingConfiguration(componentID:task.componentID,values:node.config).resolved(input:input).outputLayout
                    }
                    else if let task=activationTask,node.type == task.componentID {layout=input}
                    else if let task=poolingTask,node.type == task.componentID {
                        switch task {
                        case .maxOne:layout=try MaxPool1dConfiguration(values:node.config).resolved(input:input).outputLayout
                        case .maxTwo:layout=try MaxPool2dConfiguration(values:node.config).resolved(input:input).outputLayout
                        case .maxThree:layout=try MaxPool3dConfiguration(values:node.config).resolved(input:input).outputLayout
                        case .averageOne:layout=try AvgPool1dConfiguration(values:node.config).resolved(input:input).outputLayout
                        }
                    }
                    else if let task=convolutionTask,node.type == task.componentID {
                        switch task {
                        case .one:layout=try Conv1dConfiguration(values:node.config).resolved(input:input).outputLayout
                        case .two:layout=try Conv2dConfiguration(values:node.config).resolved(input:input).outputLayout
                        case .three:layout=try Conv3dConfiguration(values:node.config).resolved(input:input).outputLayout
                        case .transposedOne:layout=try ConvTranspose1dConfiguration(values:node.config).resolved(input:input).outputLayout
                        case .transposedTwo:layout=try ConvTranspose2dConfiguration(values:node.config).resolved(input:input).outputLayout
                        case .transposedThree:layout=try ConvTranspose3dConfiguration(values:node.config).resolved(input:input).outputLayout
                        }
                    }
                    else if node.type == "lazy_linear",allowLazyLinear {layout=try LazyLinearConfiguration(values:node.config).resolved(input:input).outputLayout}
                    else if node.type == "rms_norm" {layout=try RMSNormConfiguration(values:node.config).outputLayout(for:input)}
                    else if node.type == "layer_norm" {layout=try LayerNormConfiguration(values:node.config).outputLayout(for:input)}
                    else if node.type == "softmax" {layout=try SoftmaxConfiguration(values:node.config).outputLayout(for:input)}
                    else {layout=try TensorTransformConfiguration(operation:node.type,values:node.config).outputLayout(for:input)}
                } else {layout=nil}
                if let layout {result.tensorContracts[id]=TypedTensorContract(layout:layout,dtype:dtype)}
            } catch {issue(error.localizedDescription,id)}
            for edge in outgoing[id] ?? [] {
                pending[edge.target,default:0]-=1
                if pending[edge.target] == 0 {queue.append(edge.target)}
            }
        }
        return result
    }
    /// Routing lessons carry contracts for each named source address, including fixture-owned int64 selectors.
    private static func inspectRouting(_ graph:GraphDocument,task:TensorRoutingLearningTask)->ValidationResult {
        var result=ValidationResult()
        func issue(_ text:String,_ node:String?=nil) {result.issues.append(.init(id:"tensor.routing.\(result.issues.count)",message:text,nodeID:node))}
        do {_ = try TensorRoutingLearningSubmission.validatedGraph(graph,task:task,complete:false)} catch {issue(error.localizedDescription);return result}
        do {_ = try TensorRoutingLearningSubmission.validatedGraph(graph,task:task,complete:true)} catch {issue(error.localizedDescription)}
        let nodes=Dictionary(uniqueKeysWithValues:graph.nodes.map {($0.id,$0)}),incoming=Dictionary(grouping:graph.edges,by:\.target),outgoing=Dictionary(grouping:graph.edges,by:\.source)
        var pending=nodes.mapValues {incoming[$0.id]?.count ?? 0},queue=graph.nodes.filter {pending[$0.id] == 0}.map(\.id),cursor=0
        while cursor<queue.count {
            let id=queue[cursor];cursor+=1;let node=nodes[id]!;result.orderedNodeIDs.append(id)
            do {
                if node.type == "input" {
                    let role=node.config["fixture"]!.stringValue!
                    result.outputTensorContracts[.init(id,"out")]=try .init(sample:task.sampleShapes[role]!,dtype:role == "indices" ? .int64:.float64)
                } else {
                    let ports:[String]
                    if node.type == task.componentID {ports=try TensorRoutingConfiguration(componentID:node.type,values:node.config).configuredInputs}
                    else {ports=node.type == "concat" ? ["a","b"]:["in"]}
                    let inputs=ports.compactMap {port in incoming[id]?.first(where:{$0.targetPort == port}).flatMap {result.outputTensorContracts[.init($0.source,$0.sourcePort)]}}
                    if inputs.count == ports.count {
                        let layouts:[String:BatchedTensorLayout]
                        if node.type == task.componentID {
                            let expected:[ContractDType]=node.type == "gather" ? [.float64,.int64]:Array(repeating:.float64,count:inputs.count)
                            guard inputs.map(\.dtype) == expected else {throw LearningContractError("Use number sources and the fixture-owned whole-number positions at their configured ports.")}
                            layouts=try TensorRoutingConfiguration(componentID:node.type,values:node.config).resolved(layouts:inputs.map(\.layout))
                        } else if node.type == "concat" {
                            guard inputs.allSatisfy({$0.dtype == .float64}) else {throw LearningContractError("Concatenate requires floating number sources.")}
                            layouts=["out":try TensorCompositionConfiguration(componentID:"concat",values:node.config).resolved(layouts:inputs.map(\.layout))]
                        } else if node.type == "output" {layouts=["out":inputs[0].layout]}
                        else {layouts=["out":try TensorTransformConfiguration(operation:node.type,values:node.config).outputLayout(for:inputs[0].layout)]}
                        for (port,layout) in layouts {result.outputTensorContracts[.init(id,port)] = .init(layout:layout,dtype:node.type == "gather" ? .float64:inputs[0].dtype)}
                    }
                }
            } catch {issue(error.localizedDescription,id)}
            for edge in outgoing[id] ?? [] {pending[edge.target,default:0]-=1;if pending[edge.target] == 0 {queue.append(edge.target)}}
        }
        return result
    }

}
