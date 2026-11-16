import Foundation

/// Portable authoring examples. Local source permissions/bindings belong to the App.
public enum WorkflowExamples {
    public static let sourceAddress = WorkflowNodeAddress(domain: .data, localID: "observations")
    public static let trainAddress = WorkflowNodeAddress(domain: .control, localID: "train")
    public static let observationsResourceName = "workflow-observations.csv"

    public static func preparedParallelClassifier(projectID: String = UUID().uuidString,
                                                   device: String = "auto") throws -> WorkflowDocument {
        try preparedClassifier(projectID:projectID,device:device,model:nil)
    }
    public static func preparedTensorLayoutClassifier(projectID:String = UUID().uuidString,device:String = "auto")throws->WorkflowDocument {
        try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.tensorLayoutClassifier)
    }
    public static func preparedSoftmaxClassifier(projectID:String = UUID().uuidString,device:String = "auto")throws->WorkflowDocument {
        try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.softmaxClassifier)
    }
    public static func preparedLayerNormClassifier(projectID:String = UUID().uuidString,device:String = "auto",normalizeWholeSample:Bool = false)throws->WorkflowDocument {
        try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.layerNormClassifier(normalizeWholeSample:normalizeWholeSample))
    }
    public static func preparedGELUClassifier(projectID:String = UUID().uuidString,device:String = "auto",approximateTanh:Bool = false)throws->WorkflowDocument {
        try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.geluClassifier(approximateTanh:approximateTanh))
    }
    public static func preparedEmbeddingClassifier(projectID:String = UUID().uuidString,device:String = "auto")throws->WorkflowDocument {
        var document = try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.embeddingClassifier,scaleFeatures:false)
        // Only new examples receive authored lane positions. Never rearrange a saved project.
        let dataLane: [WorkflowNodeAddress] = [sourceAddress, .init(domain:.data,localID:"cast"),
            .init(domain:.data,localID:"split"), .init(domain:.control,localID:"mapping")]
        let modelLane = ["input","lookup","flat","head","output"].map { WorkflowNodeAddress(domain:.model,localID:$0) }
            + [.init(domain:.control,localID:"model"), trainAddress]
        var positions: [WorkflowNodeAddress:(Double,Double)] = [:]
        for (lane, y) in [(dataLane,40.0),(modelLane,210.0)] {
            for (column, address) in lane.enumerated() { positions[address] = (40 + Double(column) * 230, y) }
        }
        positions[.init(domain:.control,localID:"setup")] = (1190,380)
        for index in document.nodes.indices {
            if let position = positions[document.nodes[index].address] {
                document.nodes[index].x = position.0; document.nodes[index].y = position.1
            }
        }
        return document
    }
    static func preparedClassifier(projectID:String,device:String,model:GraphDocument?,scaleFeatures:Bool=true)throws->WorkflowDocument {
        guard UUID(uuidString: projectID) != nil, ["auto", "cpu", "mps"].contains(device) else {
            throw DataFlowContractError("Choose a UUID project identity and Auto, CPU or MPS execution.")
        }
        func address(_ domain: WorkflowDomain, _ id: String) -> WorkflowNodeAddress { .init(domain: domain, localID: id) }
        func data(_ id: String, _ name: String, _ x: Double, _ y: Double,
                  _ op: String, _ config: [String: JSONValue]) -> WorkflowNode {
            .init(address: address(.data, id), name: name, x: x, y: y,
                  payload: .dataOperator(operatorID: "kwip.data." + op, config: config))
        }
        let columns: JSONValue = .array([.string("signal"), .string("context")])
        var nodes: [WorkflowNode] = [
            data("observations", "Signal observations", 80, 80, "sourceCSV", ["separator": .string(",")]),
            data("cast", "Choose numeric columns and class", 340, 80, "columns", CurrentPracticeConfiguration.columns([("signal","float64"),("context","float64"),("label","int64")])),
            data("split", "Reserve validation and final test", 600, 80, "partition", ["strategy":.string("random"),"seed":42,"validationPercent":25,"testPercent":25,"stratifyColumn":.string("label")]),
            data("fit", "Prepare partitions from training statistics", 860, 80, "preprocess", CurrentPracticeConfiguration.scaledPartitions(["signal","context"])),
            data("training", "Inspect prepared training observations", 1120, 80, "view", [:]),
            data("validation", "Inspect prepared validation observations", 1120, 300, "view", [:]),
            .init(address: address(.control, "mapping"), name: "Features and class target", x: 1380, y: 190,
                  payload: .trainingData(config: ["featureColumns": columns, "targetColumn": .string("label")])),
            .init(address: address(.control, "model"), name: "Parallel classifier architecture", x: 1120, y: 530, payload: .modelDefinition),
            .init(address: address(.control, "setup"), name: "Three-epoch experiment", x: 1380, y: 750,
                  payload: .runSetup(config: ["device": .string(device), "epochs": .integer(3),
                    "batchSize": .integer(32), "seed": .integer(42), "learningRate": .number(0.03)])),
            .init(address: trainAddress, name: "Fit and evaluate classifier", x: 1660, y: 410, payload: .train(config: [:]))
        ]
        if model == nil {
            // Fresh examples leave room for the expanded Parallel features container.
            // Saved project positions are never rearranged here.
            for index in nodes.indices {
                switch nodes[index].address.localID {
                case "model":nodes[index].x=1620
                case "setup":nodes[index].x=1880
                case "train":nodes[index].x=2160
                default:break
                }
            }
        }
        let source = model.map { CompositeProjectDocument(name:$0.name,root:CompositeGraph(nodes:$0.nodes,edges:$0.edges,configurationVersion:$0.schemaVersion),definitions:[]) } ?? CompositeExamples.parallelClassifier
        if model != nil {
            for index in nodes.indices where nodes[index].address.domain == .control {
                nodes[index].x += 650
                if nodes[index].address.localID == "model" {nodes[index].name=model!.name + " architecture"}
            }
        }
        let positions: [String: (Double, Double, String)] = [
            "input": (80, 530, "Two scaled features"),
            "classifier": (1100, 530, "Class scores"),
            "output": (1360, 530, "Binary prediction")
        ]
        for node in source.root.nodes {
            let placement = model == nil ? positions[node.id] ?? (80, 530, node.name) : (node.x+50,node.y+400,node.name)
            nodes.append(.init(address: address(.model, node.id), name: placement.2, x: placement.0, y: placement.1,
                payload: .modelPrimitive(type: node.type, configurationVersion: source.root.configurationVersion ?? 1, config: node.config)))
        }
        for instance in source.root.instances {
            nodes.append(.init(address: address(.model, instance.id), name: "Parallel feature paths", x: 340, y: 530,
                payload: .modelBlock(reference: instance.definition, overrides: instance.overrides)))
        }
        var edges: [WorkflowEdge] = []
        func wire(_ fromDomain: WorkflowDomain, _ from: String, _ port: String,
                  _ toDomain: WorkflowDomain, _ to: String, _ input: String, _ kind: WorkflowEdgeKind) {
            edges.append(.init(id: "wire-\(edges.count + 1)",
                from: .init(address: address(fromDomain, from), port: port),
                to: .init(address: address(toDomain, to), port: input), kind: kind))
        }
        wire(.data, "observations", "table", .data, "cast", "table", .table)
        wire(.data, "cast", "table", .data, "split", "table", .table)
        for role in ["training","validation","test"] {wire(.data,"split",role,.data,"fit",role,.table)}
        wire(.data,"fit","training",.data,"training","table",.table)
        wire(.data,"fit","validation",.data,"validation","table",.table)
        wire(.data, "fit", "training", .control, "mapping", "training", .table)
        wire(.data, "fit", "validation", .control, "mapping", "validation", .table)
        for edge in source.root.edges {
            edges.append(.init(id: edge.id, from: .init(address: address(.model, edge.source), port: edge.sourcePort),
                to: .init(address: address(.model, edge.target), port: edge.targetPort), kind: .tensor))
        }
        wire(.model, "output", "definition", .control, "model", "source", .modelDefinition)
        wire(.control, "model", "model", .control, "train", "model", .modelDefinition)
        wire(.control, "mapping", "data", .control, "train", "data", .trainingData)
        wire(.control, "setup", "configuration", .control, "train", "configuration", .runConfiguration)
        if !scaleFeatures {
            let removed=Set(["fit","training","validation"].map {address(.data,$0)})
            nodes.removeAll {removed.contains($0.address)}
            edges.removeAll {removed.contains($0.from.address) || removed.contains($0.to.address)}
            if let cast=nodes.firstIndex(where:{$0.address == address(.data,"cast")}) {
                nodes[cast].name="Exact category indices and class"
                nodes[cast].payload = .dataOperator(operatorID:"kwip.data.columns",config:CurrentPracticeConfiguration.columns([("signal","int64"),("context","int64"),("label","int64")]))
            }
            // Explicit stable identities avoid reusing IDs removed from the original flow.
            for port in ["training","validation"] {
                edges.append(.init(id:"integer-"+port,from:.init(address:address(.data,"split"),port:port),to:.init(address:address(.control,"mapping"),port:port),kind:.table))
            }
        }
        let document = WorkflowDocument(projectID: projectID, name: model == nil ? "Prepare and train a parallel classifier" : "Prepare and train " + model!.name.lowercased(),
            nodes: nodes, edges: edges, definitions: source.definitions)
        try WorkflowCodec.validate(document)
        return document
    }
}
