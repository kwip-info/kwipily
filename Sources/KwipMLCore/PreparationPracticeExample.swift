import Foundation

/// A small, public synthetic exercise, not a model-quality benchmark.
/// The reserved future/test branch is not evaluated by this training example.
public enum PreparationPracticeExample {
    public static let observationsResource = "preparation-practice-observations"
    public static let featureColumns = ["amount", "onehot:[\"region\",\"north\"]", "onehot:[\"region\",\"south\"]"]
    public static func observationsURL() throws -> URL {
        guard let url = Bundle.module.url(forResource: observationsResource, withExtension: "csv") else {
            throw DataFlowContractError("The preparation practice observations are missing from this build.")
        }
        return url
    }
    public static func document(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument {
        let model = GraphDocument(name: "Prepared category classifier", nodes: [
            .init(id: "input", type: "input", name: "Amount and two category indicators", x: 30, y: 180, config: ["features": 3]),
            .init(id: "head", type: "linear", name: "Two class scores", x: 260, y: 180, config: ["in_features": 3, "out_features": 2, "bias": true]),
            .init(id: "output", type: "output", name: "Class prediction", x: 490, y: 180)
        ], edges: [.init(id: "input-head", source: "input", target: "head"), .init(id: "head-output", source: "head", target: "output")])
        var document = try WorkflowCodec.migrateToVersion2(WorkflowExamples.preparedClassifier(projectID: projectID, device: device, model: model))
        document.name = "Practice missing values and categories"
        func address(_ id: String) -> WorkflowNodeAddress { .init(domain: .data, localID: id) }
        let removed = Set(["fit", "training", "validation"].map(address))
        document.nodes.removeAll { removed.contains($0.address) }
        document.edges.removeAll { removed.contains($0.from.address) || removed.contains($0.to.address) }
        for index in document.nodes.indices {
            let node = document.nodes[index]
            if node.address == WorkflowExamples.sourceAddress { document.nodes[index].name = "Synthetic dated observations" }
            if node.address == address("cast") {
                document.nodes[index].name = "Preserve missing amount; type the class"
                document.nodes[index].payload = .dataOperator(operatorID: "kwip.data.columns", config: CurrentPracticeConfiguration.columns([("amount","float64"),("label","int64")],missingPolicy:"preserve"))
            }
            if node.address == address("split") {
                document.nodes[index].name = "Eight training rows; four held out; four reserved"
                document.nodes[index].payload = .dataOperator(operatorID: "kwip.data.partition", config: ["strategy":.string("time"),
                    "timeColumn": .string("observed_at"), "timeEncoding": .string("iso8601Offset"),
                    "trainingEnd": .string("2026-01-08T00:00:00Z"), "validationEnd": .string("2026-01-12T00:00:00Z"),
                    "embargoMilliseconds": 0, "entityColumns": .array([]), "entityPolicy": .string("allowAcrossTime")])
            }
            if case .trainingData = node.payload {
                document.nodes[index].name = "Amount plus two category indicators"
                document.nodes[index].payload = .trainingData(config: ["featureColumns": .array(featureColumns.map(JSONValue.string)), "targetColumn": .string("label")])
            }
            if case .runSetup(var config) = node.payload {
                config["epochs"] = 1; config["batchSize"] = 8; config["preparationCachePolicy"] = .string("materialize")
                document.nodes[index].name = "One-epoch preparation smoke test"
                document.nodes[index].payload = .runSetup(config: config)
            }
        }
        func data(_ id: String, _ title: String, _ x: Double, _ y: Double, _ op: String, _ config: [String: JSONValue]) -> WorkflowNode {
            .init(address: address(id), name: title, x: x, y: y, payload: .dataOperator(operatorID: "kwip.data." + op, config: config))
        }
        var amount=PreprocessConfiguration.Column(column:"amount");amount.missing="mean"
        var region=PreprocessConfiguration.Column(column:"region");region.encoding="oneHot";region.vocabularyOrder="lexicographic";region.unknownPolicy="ignore";region.maxCategories=2
        document.nodes.append(data("prepare","Learn replacements and categories from training",860,40,"preprocess",PreprocessConfiguration(columns:[amount,region]).values))
        for (index,role) in ["training","validation"].enumerated() {
            document.nodes.append(data(role,"Inspect prepared "+role,1200,Double(120+index*220),"view",[:]))
            document.edges.append(.init(id:"practice-prepare-"+role,from:.init(address:address("prepare"),port:role),to:.init(address:address(role),port:"table"),kind:.table))
            document.edges.append(.init(id:"practice-"+role+"-mapping",from:.init(address:address("prepare"),port:role),to:.init(address:.init(domain:.control,localID:"mapping"),port:role),kind:.table))
        }
        for role in ["training","validation","test"] {
            document.edges.append(.init(id:"practice-partition-"+role,from:.init(address:address("split"),port:role),to:.init(address:address("prepare"),port:role),kind:.table))
        }
        try WorkflowCodec.validate(document)
        return document
    }
}
