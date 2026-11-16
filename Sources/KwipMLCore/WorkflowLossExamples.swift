import Foundation

extension WorkflowExamples {
    /// Real typed source/target bindings; source permissions are supplied by the App.
    public static func preparedLossObjective(task: LossLearningTask, projectID: String = UUID().uuidString,
                                             device: String = "auto") throws -> WorkflowDocument {
        guard UUID(uuidString: projectID) != nil, ["auto", "cpu", "mps"].contains(device) else {
            throw GraphConfigurationError("Choose a UUID project identity and Auto, CPU or MPS execution.")
        }
        func address(_ id: String, _ domain: WorkflowDomain = .control) -> WorkflowNodeAddress { .init(domain: domain, localID: id) }
        let inputs = task == .triplet ? ["anchor_x","anchor_y","match_x","match_y","nonmatch_x","nonmatch_y"] : ["signal","context"]
        func role(_ name: String, _ columns: [String], _ shape: [Int], _ dtype: ContractDType = .float32) -> LossTargetRoleMapping {
            .init(role: name, columns: columns, sampleShape: shape, dtype: dtype)
        }
        let roles: [LossTargetRoleMapping]
        switch task {
        case .crossEntropy, .nll: roles = [role("target", ["class"], [], .int64)]
        case .bce: roles = [role("target", ["desired_a","desired_b"], [2]), role("weight", ["importance_a","importance_b"], [2])]
        case .mse: roles = [role("target", ["desired"], [1])]
        case .l1: roles = [role("target", ["desired_a","desired_b"], [2])]
        case .kl: roles = [role("target", ["log_share_a","log_share_b","log_share_c"], [3])]
        case .ctc: roles = [role("target", ["label_a","label_b"], [2], .int64), role("input_lengths", ["input_length"], [], .int64), role("target_lengths", ["target_length"], [], .int64)]
        case .triplet: roles = []
        }
        let mapping = LossTargetMapping(componentID: task.componentID, inputColumns: inputs, roles: roles,
            tripletSourceGroups: task == .triplet ? [Array(inputs[0..<2]), Array(inputs[2..<4]), Array(inputs[4..<6])] : [])
        try mapping.validate()
        var casts = Dictionary(uniqueKeysWithValues: inputs.map { ($0, JSONValue.string("float64")) })
        for item in roles { for column in item.columns { casts[column] = .string(item.dtype == .float32 ? "float64" : item.dtype.rawValue) } }
        let mappingValue = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(mapping))
        var nodes: [WorkflowNode] = [
            .init(address: sourceAddress, name: "Example observations", x: 40, y: 40, payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: ["separator": .string(",")])),
            .init(address: address("cast", .data), name: "Explicit column types", x: 280, y: 40, payload: .dataOperator(operatorID: "kwip.data.columns", config: CurrentPracticeConfiguration.columns(casts.keys.sorted().map {($0,casts[$0]!.stringValue!)}))),
            .init(address: address("split", .data), name: "Reserve validation and final test", x: 520, y: 40, payload: .dataOperator(operatorID: "kwip.data.partition", config: ["strategy":.string("random"),"seed":42,"validationPercent":25,"testPercent":25])),
            .init(address: address("mapping"), name: "Input and desired-outcome columns", x: 780, y: 40, payload: .trainingData(config: ["lossTargetMapping": mappingValue])),
            .init(address: address("model"), name: "Prediction architecture", x: 1000, y: 260, payload: .modelDefinition),
            .init(address: address("setup"), name: "Three-epoch experiment", x: 780, y: 480, payload: .runSetup(config: ["device": .string(device), "epochs": 3, "batchSize": 32, "seed": 42, "learningRate": 0.03])),
            .init(address: trainAddress, name: "Fit and inspect held-out outcomes", x: 1240, y: 180, payload: .train(config: [:]))
        ]
        var edges: [WorkflowEdge] = []
        func wire(_ from: WorkflowNodeAddress, _ output: String, _ to: WorkflowNodeAddress, _ input: String, _ kind: WorkflowEdgeKind) {
            edges.append(.init(id: "loss-wire-\(edges.count + 1)", from: .init(address: from, port: output), to: .init(address: to, port: input), kind: kind))
        }
        wire(sourceAddress, "table", address("cast", .data), "table", .table)
        wire(address("cast", .data), "table", address("split", .data), "table", .table)
        for partition in ["training","validation"] { wire(address("split", .data), partition, address("mapping"), partition, .table) }
        wire(address("mapping"), "data", trainAddress, "data", .trainingData)
        wire(address("model"), "model", trainAddress, "model", .modelDefinition)
        wire(address("setup"), "configuration", trainAddress, "configuration", .runConfiguration)
        var previous: WorkflowNodeAddress?
        var column = 0
        func model(_ id: String, _ title: String, _ kind: String, _ config: [String: JSONValue]) {
            let current = address(id, .model)
            nodes.append(.init(address: current, name: title, x: 40 + Double(column) * 220, y: 260,
                payload: .modelPrimitive(type: kind, configurationVersion: 2, config: config)))
            if let prior = previous { wire(prior, "out", current, "in", .tensor) }
            previous = current; column += 1
        }
        model("input", "Ordered input values", "input", ["features": .integer(Int64(inputs.count))])
        if task == .triplet { model("shape", "Anchor, match and nonmatch", "reshape", ["shape": .array([3,2])]) }
        let width = task == .ctc ? 9 : (task == .crossEntropy || task == .nll || task == .kl ? 3 : 2)
        model("head", task == .triplet ? "Shared comparison coordinates" : "Produced values", "linear", ["in_features": 2, "out_features": .integer(Int64(width)), "bias": true])
        if task == .ctc { model("shape", "Three positions and three classes", "reshape", ["shape": .array([3,3])]) }
        if task.usesLogSoftmax { model("logs", "Prepare logarithmic probabilities", "log_softmax", ["dim": -1]) }
        model("output", "Model prediction", "output", [:])
        wire(address("output", .model), "definition", address("model"), "source", .modelDefinition)
        var document = WorkflowDocument(projectID: projectID, name: "Prepare and train · " + task.title, nodes: nodes, edges: edges)
        // The already-authored lossTargetMapping has its own schema8 prerequisite.
        // Establish it before validated Train migration retains the source2 model metadata.
        document.schemaVersion = 8
        try CurrentPracticeConfiguration.setTrain("objective",value:try LossConfiguration(componentID:task.componentID,values:task.configuration).recipeValue,in:&document)
        try WorkflowCodec.validate(document)
        return document
    }
    public static func preparedCrossEntropyClassifier(device: String = "auto") throws -> WorkflowDocument { try preparedLossObjective(task: .crossEntropy, device: device) }
    public static func preparedBCEMultilabel(device: String = "auto") throws -> WorkflowDocument { try preparedLossObjective(task: .bce, device: device) }
    public static func preparedMSERegression(device: String = "auto") throws -> WorkflowDocument { try preparedLossObjective(task: .mse, device: device) }
    public static func preparedL1Regression(device: String = "auto") throws -> WorkflowDocument { try preparedLossObjective(task: .l1, device: device) }
    public static func preparedNLLClassifier(device: String = "auto") throws -> WorkflowDocument { try preparedLossObjective(task: .nll, device: device) }
    public static func preparedKLDistribution(device: String = "auto") throws -> WorkflowDocument { try preparedLossObjective(task: .kl, device: device) }
    public static func preparedCTCSequence(device: String = "auto") throws -> WorkflowDocument { try preparedLossObjective(task: .ctc, device: device) }
    public static func preparedTripletDistance(device: String = "auto") throws -> WorkflowDocument { try preparedLossObjective(task: .triplet, device: device) }
}
