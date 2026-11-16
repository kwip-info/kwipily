// Frozen authored fixture copied from WorkflowExamples.swift at
// 7bb7f8a8e4ac75a40df09efbc2792a77bcc962af. The unchanged original capture hashes
// in LossWorkflowArchiveTests detect any transitive model/projection drift.
import Foundation
import KwipMLCore

/// Portable authoring examples. Local source permissions/bindings belong to the App.
enum HistoricalLossArchiveSource {
    public static let sourceAddress = WorkflowNodeAddress(domain: .data, localID: "observations")
    public static let trainAddress = WorkflowNodeAddress(domain: .control, localID: "train")
    public static let observationsResourceName = "workflow-observations.csv"

    public static func preparedParallelClassifier(projectID: String = UUID().uuidString,
                                                   device: String = "auto") throws -> WorkflowDocument {
        try preparedClassifier(projectID:projectID,device:device,model:nil)
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
            data("cast", "Numeric features and class", 340, 80, "cast", ["columns": .object([
                "signal": .string("float64"), "context": .string("float64"), "label": .string("int64")])]),
            data("split", "Hold out one quarter", 600, 80, "split", ["seed": .integer(42),
                "validationPercent": .integer(25), "stratifyColumn": .string("label")]),
            data("fit", "Learn training statistics", 860, 80, "standardScalerFit", ["columns": columns]),
            data("training", "Scale training observations", 1120, 80, "transform", [:]),
            data("validation", "Apply to held-out observations", 1120, 300, "transform", [:]),
            .init(address: address(.control, "mapping"), name: "Features and class target", x: 1380, y: 190,
                  payload: .trainingData(config: ["featureColumns": columns, "targetColumn": .string("label")])),
            .init(address: address(.control, "model"), name: "Parallel classifier architecture", x: 1120, y: 530, payload: .modelDefinition),
            .init(address: address(.control, "setup"), name: "Three-epoch experiment", x: 1380, y: 750,
                  payload: .runSetup(config: ["device": .string(device), "epochs": .integer(3),
                    "batchSize": .integer(32), "seed": .integer(42), "learningRate": .number(0.03)])),
            .init(address: trainAddress, name: "Fit and evaluate classifier", x: 1660, y: 410, payload: .train(config: [:]))
        ]
        let source = model.map { CompositeProjectDocument(name:$0.name,root:CompositeGraph(nodes:$0.nodes,edges:$0.edges,configurationVersion:$0.schemaVersion),definitions:[]) } ?? CompositeExamples.parallelClassifier
        if model != nil {
            for index in nodes.indices where nodes[index].address.domain == .control {
                nodes[index].x += 650
                if nodes[index].address.localID == "model" {nodes[index].name=model!.name + " architecture"}
            }
        }
        let positions: [String: (Double, Double, String)] = [
            "input": (80, 530, "Two scaled features"),
            "classifier": (600, 530, "Class scores"),
            "output": (860, 530, "Binary prediction")
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
        wire(.data, "split", "training", .data, "fit", "training", .dataset)
        wire(.data, "split", "training", .data, "training", "table", .dataset)
        wire(.data, "split", "validation", .data, "validation", "table", .dataset)
        wire(.data, "fit", "state", .data, "training", "state", .fittedState)
        wire(.data, "fit", "state", .data, "validation", "state", .fittedState)
        wire(.data, "training", "table", .control, "mapping", "training", .table)
        wire(.data, "validation", "table", .control, "mapping", "validation", .table)
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
                nodes[cast].payload = .dataOperator(operatorID:"kwip.data.cast",config:["columns":.object(["signal":.string("int64"),"context":.string("int64"),"label":.string("int64")])])
            }
            // Explicit stable identities avoid reusing IDs removed from the original flow.
            for port in ["training","validation"] {
                edges.append(.init(id:"integer-"+port,from:.init(address:address(.data,"split"),port:port),to:.init(address:address(.control,"mapping"),port:port),kind:.dataset))
            }
        }
        let document = WorkflowDocument(projectID: projectID, name: model == nil ? "Prepare and train a parallel classifier" : "Prepare and train " + model!.name.lowercased(),
            nodes: nodes, edges: edges, definitions: source.definitions)
        try WorkflowCodec.validate(document)
        return document
    }
}
