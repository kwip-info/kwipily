import Foundation

public struct WorkflowEstimatorPlan: Equatable {
    public let projectID: String
    public let revision: UInt64
    public let estimator: WorkflowNodeAddress
    public let configuration: WorkflowEstimatorConfiguration
    public let dataflow: DataFlowDocument
    public let sourceAddresses: [WorkflowNodeAddress]
    public let training: DataFlowEndpoint
    public let validation: DataFlowEndpoint
    public var documentModelSource:WorkflowNodeAddress? = nil
    public let model: ModelExecutionSnapshot?
}
public enum WorkflowEstimatorCompiler {
    public static func compile(_ document: WorkflowDocument, estimator: WorkflowNodeAddress,
                               modelRegistry: ComponentRegistry, dataRegistry: DataFlowOperatorRegistry) throws -> WorkflowEstimatorPlan {
        try RetiredTrainingRecipeMigration.requireCurrent(document)
        try RetiredDataOperatorMigration.requireCurrent(document)
        return try compilePlan(document, estimator: estimator, modelRegistry: modelRegistry, dataRegistry: dataRegistry, historicalEvidence: false)
    }
    /// Captured evidence reconstruction only; never authorizes a new estimator run.
    public static func compileHistoricalEvidence(_ document: WorkflowDocument, estimator: WorkflowNodeAddress,
                               modelRegistry: ComponentRegistry, dataRegistry: DataFlowOperatorRegistry) throws -> WorkflowEstimatorPlan {
        try compilePlan(document, estimator: estimator, modelRegistry: modelRegistry, dataRegistry: dataRegistry, historicalEvidence: true)
    }
    private static func compilePlan(_ document: WorkflowDocument, estimator: WorkflowNodeAddress,
                               modelRegistry: ComponentRegistry, dataRegistry: DataFlowOperatorRegistry, historicalEvidence: Bool) throws -> WorkflowEstimatorPlan {
        try WorkflowCodec.validate(document)
        guard document.drafts.isEmpty, let node = document.nodes.first(where: { $0.address == estimator }),
              case .estimator(let raw) = node.payload else { throw DataFlowContractError("Select an estimator with no unapplied Block drafts.") }
        let configuration = try WorkflowEstimatorConfiguration.decode(raw)
        if configuration.backendID == "tensorflow" { try TensorFlowEstimatorConfiguration.validate(configuration.settings) }
        let region = try WorkflowCompiler.dependencies(document, terminal: estimator, modelRegistry: modelRegistry, dataRegistry: dataRegistry, historicalEvidence: historicalEvidence)
        func input(_ port: String) throws -> WorkflowEndpoint {
            guard let edge = region.incoming[estimator]?.first(where: { $0.to.port == port }) else { throw DataFlowContractError("Connect the estimator's \(port) input.") }
            return edge.from
        }
        if configuration.isPrediction {
            let table=try input("table"),source=try input("model")
            guard let provider=document.nodes.first(where:{$0.address==source.address}),case .frozenModel(let values)=provider.payload,try LoadModelConfiguration.decode(values).source == .documentPackage else {throw DataFlowContractError("Connect Load model → Prebuilt / document model package for prediction/evaluation.")}
            let nodes=document.nodes.filter {region.selected.contains($0.address) && $0.address.domain == .data}
            let dataflow=DataFlowDocument(nodes:nodes.compactMap {node in if case .dataOperator(let id,let config)=node.payload {return .init(id:node.address.localID,operatorID:id,config:config)};return nil},edges:document.edges.filter {region.selected.contains($0.to.address) && $0.from.address.domain == .data && $0.to.address.domain == .data}.map {.init(from:.init(node:$0.from.address.localID,port:$0.from.port),to:.init(node:$0.to.address.localID,port:$0.to.port))})
            try DataFlowCodec.validate(dataflow)
            var plan=WorkflowEstimatorPlan(projectID:document.projectID,revision:document.revision,estimator:estimator,configuration:configuration,dataflow:dataflow,sourceAddresses:nodes.filter(DataFlowSourceOperators.isSource).map(\.address),training:.init(node:table.address.localID,port:table.port),validation:.init(node:table.address.localID,port:table.port),model:nil)
            plan.documentModelSource=source.address;return plan
        }
        let training = try input("training"), validation = PrebuiltModelConfiguration.isCluster(raw) ? training:try input("validation")
        let evidence = try WorkflowDataLineage.resolve(region, dataRegistry: dataRegistry)
        if !PrebuiltModelConfiguration.isCluster(raw) {
        guard case .rows(let train) = evidence[training], case .rows(let heldout) = evidence[validation],
              train.role == "training", heldout.role == "validation", train.source == heldout.source,
              train.split != nil, train.split == heldout.split, train.states == heldout.states else {
            throw DataFlowContractError("Estimator inputs require matching prepared training/validation roles, source, split and fitted state.")
        }
        }
        let nodes = document.nodes.filter { region.selected.contains($0.address) && $0.address.domain == .data }
        let dataflow = DataFlowDocument(nodes: nodes.compactMap { node in
            if case .dataOperator(let id, let config) = node.payload { return .init(id: node.address.localID, operatorID: id, config: config) }; return nil
        }, edges: document.edges.filter { region.selected.contains($0.to.address) && $0.from.address.domain == .data && $0.to.address.domain == .data }.map {
            .init(from: .init(node: $0.from.address.localID, port: $0.from.port), to: .init(node: $0.to.address.localID, port: $0.to.port))
        })
        try DataFlowCodec.validate(dataflow)
        let model: ModelExecutionSnapshot?
        if configuration.backendID == "tensorflow" {
            model = try WorkflowCompiler.modelSnapshot(document, definition: input("model").address, modelRegistry: modelRegistry, dataRegistry: dataRegistry)
            if let model { try TensorFlowEstimatorConfiguration.validateGraph(model.graph, registry: modelRegistry) }
        } else { model = nil }
        return .init(projectID: document.projectID, revision: document.revision, estimator: estimator, configuration: configuration,
                     dataflow: dataflow, sourceAddresses: nodes.filter(DataFlowSourceOperators.isSource).map(\.address),
                     training: .init(node: training.address.localID, port: training.port), validation: .init(node: validation.address.localID, port: validation.port), model: model)
    }
}
