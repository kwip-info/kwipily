import Foundation

/// One universal canvas can execute its data region without a model terminal.
/// Unused model/control nodes are left intact; only data-domain edges execute.
public enum WorkflowDataPreparationPlan {
    public static func compile(_ document:WorkflowDocument,registry:DataFlowOperatorRegistry,allowTrainingTerminals:Bool=false)throws->DataFlowDocument {
        try compilePlan(document,registry:registry,allowTrainingTerminals:allowTrainingTerminals,historicalEvidence:false)
    }
    /// Reconstruct captured preparation for evidence validation only; never authorizes a new run.
    public static func compileHistoricalEvidence(_ document:WorkflowDocument,registry:DataFlowOperatorRegistry,allowTrainingTerminals:Bool=false)throws->DataFlowDocument {
        try compilePlan(document,registry:registry,allowTrainingTerminals:allowTrainingTerminals,historicalEvidence:true)
    }
    private static func compilePlan(_ document:WorkflowDocument,registry:DataFlowOperatorRegistry,allowTrainingTerminals:Bool,historicalEvidence:Bool)throws->DataFlowDocument {
        if !historicalEvidence {try RetiredDataOperatorMigration.requireCurrent(document)}
        try WorkflowCodec.validate(document)
        guard document.drafts.isEmpty,(allowTrainingTerminals || !document.nodes.contains(where:{node in switch node.payload {case .train,.estimator,.saveModel:return true;default:return false}})) else {throw DataFlowContractError("Use the training or estimator runner for a workflow with those terminals; resolve unfinished drafts first.")}
        let nodes=document.nodes.filter {$0.address.domain == .data}
        guard !nodes.isEmpty else {throw DataFlowContractError("Add and connect data components before running preparation.")}
        let dataNodes=try nodes.map {node->DataFlowNode in
            guard case .dataOperator(let id,let configuration)=node.payload,registry.operatorDefinition(id:id) != nil else {throw DataFlowContractError("Unknown data operator at \(node.name).")}
            return DataFlowNode(id:node.address.localID,operatorID:id,config:configuration)
        }
        let edges=document.edges.filter {$0.from.address.domain == .data && $0.to.address.domain == .data}
        let map=Dictionary(uniqueKeysWithValues:nodes.map {($0.address,$0)})
        for node in nodes {
            guard case .dataOperator(let id,let config)=node.payload,let definition=registry.operatorDefinition(id:id)?.configured(config) else {throw DataFlowContractError("Unknown preparation operator.")}
            if id=="kwip.data.transcribeAudio" || (id=="kwip.data.preprocess" && config["mode"] == .string("frozen")) {
                let models=document.edges.filter {$0.to.address==node.address && $0.to.port=="model"}
                guard models.count==1,models[0].kind == .frozenModel,let provider=document.nodes.first(where:{$0.address==models[0].from.address}),case .frozenModel=provider.payload,models[0].from.port=="model" else {throw DataFlowContractError("Connect Load model's Model output to this saved-model consumer before running.")}
            }
            if id=="kwip.data.decide" {
                let models=document.edges.filter {$0.to.address==node.address && $0.to.port=="model"}
                guard models.count==1,models[0].kind == .frozenModel,let provider=document.nodes.first(where:{$0.address==models[0].from.address}),(provider.payload.isDecisionModelProvider),models[0].from.port=="model" else {throw DataFlowContractError("Connect a verified Load model or completed Train model to Decide before running.")}
            }
            if id=="kwip.data.classifyAudio" {
                let models=document.edges.filter {$0.to.address==node.address && $0.to.port=="model"}
                guard models.count==1,models[0].kind == .frozenModel,let provider=document.nodes.first(where:{$0.address==models[0].from.address}),case .frozenModel=provider.payload,models[0].from.port=="model" else {throw DataFlowContractError("Connect Load model's Model output to Classify audio before running.")}
            }
            let incoming=edges.filter {$0.to.address==node.address}
            guard Set(definition.inputs.filter {$0.required && $0.kind != "model"}.map(\.id)).isSubset(of:Set(incoming.map(\.to.port))) && Set(incoming.map(\.to.port)).isSubset(of:Set(definition.inputs.map(\.id))),Set(incoming.map(\.to.port)).count==incoming.count else {throw DataFlowContractError("Connect each required input on \(node.name.isEmpty ? definition.title:node.name).")}
            for edge in incoming {
                guard let source=map[edge.from.address],case .dataOperator(let sourceID,let sourceConfig)=source.payload,let output=registry.operatorDefinition(id:sourceID)?.configured(sourceConfig).outputs.first(where:{$0.id==edge.from.port}),let input=definition.inputs.first(where:{$0.id==edge.to.port}) else {throw DataFlowContractError("A preparation connection references an unavailable input or output.")}
                if sourceID=="kwip.data.preprocess",sourceConfig["mode"] != .string("table"),["validation","test"].contains(edge.from.port),!edges.contains(where:{$0.to.address==source.address && $0.to.port==edge.from.port}) {throw DataFlowContractError("Connect Preprocess's corresponding partition input before using its output.")}
                let tableKinds:Set<String>=["table","dataset"]
                guard output.inspectionOnly != true,(output.kind==input.kind || (tableKinds.contains(output.kind) && tableKinds.contains(input.kind))),(output.kind=="state") == (edge.kind == .fittedState) else {throw DataFlowContractError("Connect compatible table or fitted-state ports.")}
            }
        }
        let graph=DataFlowDocument(nodes:dataNodes,edges:edges.map {.init(from:.init(node:$0.from.address.localID,port:$0.from.port),to:.init(node:$0.to.address.localID,port:$0.to.port))})
        try DataFlowCodec.validate(graph)
        // Population-changing tools must refuse an already partitioned/fitted
        // ancestor in preparation-only flows as well as model-connected flows.
        let populations = nodes.filter { node in
            guard case .dataOperator(let id, let config) = node.payload else { return false }
            return id == "kwip.data.reshapeTable" || (id == "kwip.data.splitText" && config["mode"] == .string("delimited") && config["outputMode"] == .string("rows"))
        }
        let incoming = Dictionary(grouping: edges, by: { $0.to.address })
        for population in populations {
            var selected = Set<WorkflowNodeAddress>(), visiting = Set<WorkflowNodeAddress>(), order: [WorkflowNodeAddress] = []
            func visit(_ address: WorkflowNodeAddress) throws {
                guard !visiting.contains(address) else { throw DataFlowContractError("Dependency cycle prevents preparation.") }
                if selected.contains(address) { return }
                visiting.insert(address)
                for edge in incoming[address] ?? [] { try visit(edge.from.address) }
                visiting.remove(address); selected.insert(address); order.append(address)
            }
            try visit(population.address)
            _ = try WorkflowDataLineage.resolve(.init(nodes: map, incoming: incoming, selected: selected, order: order), dataRegistry: registry)
        }
        return graph
    }
}
