import Foundation

public struct ExecutionSettings: Codable, Equatable {
    public var device: String
    public var epochs: Int
    public var datasetCSV: String?
    public var featureColumns: [String]
    public var targetColumn: String?
    public var batchSize: Int
    public var sampleLimit: Int
    public var seed: Int
    public var splitSeed: Int
    public var validationPercent: Int
    public var learningRate: Double
    public var preparedTraining: PreparedTrainingSelection?
    public var compositeSource: CompositeSourceReference?
    public var trainingControl:TrainingControlPolicy?
    public var gradientCapture: GradientCaptureConfiguration?
    public init(device: String = "auto", epochs: Int = 100, datasetCSV: String? = nil, featureColumns: [String] = [], targetColumn: String? = nil, batchSize: Int = 1024, sampleLimit: Int = 0, seed: Int = 42, splitSeed: Int = 42, validationPercent: Int = 20, learningRate: Double = 0.03, preparedTraining: PreparedTrainingSelection? = nil, compositeSource: CompositeSourceReference? = nil, gradientCapture: GradientCaptureConfiguration? = nil,trainingControl:TrainingControlPolicy? = nil) {
        self.trainingControl=trainingControl
        self.device = device; self.epochs = epochs; self.datasetCSV = datasetCSV; self.featureColumns = featureColumns; self.targetColumn = targetColumn
        self.batchSize = batchSize; self.sampleLimit = sampleLimit; self.seed = seed; self.splitSeed = splitSeed; self.validationPercent = validationPercent; self.learningRate = learningRate; self.preparedTraining = preparedTraining; self.compositeSource = compositeSource; self.gradientCapture = gradientCapture
    }
    private enum CodingKeys: String, CodingKey {
        case trainingControl, device, epochs, datasetCSV, featureColumns, targetColumn, batchSize, sampleLimit, seed, splitSeed, validationPercent, learningRate, preparedTraining, compositeSource, gradientCapture
    }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.init(device: try values.decodeIfPresent(String.self, forKey: .device) ?? "auto",
                  epochs: try values.decodeIfPresent(Int.self, forKey: .epochs) ?? 100,
                  datasetCSV: try values.decodeIfPresent(String.self, forKey: .datasetCSV),
                  featureColumns: try values.decodeIfPresent([String].self, forKey: .featureColumns) ?? [],
                  targetColumn: try values.decodeIfPresent(String.self, forKey: .targetColumn),
                  batchSize: try values.decodeIfPresent(Int.self, forKey: .batchSize) ?? 1024,
                  sampleLimit: try values.decodeIfPresent(Int.self, forKey: .sampleLimit) ?? 0,
                  seed: try values.decodeIfPresent(Int.self, forKey: .seed) ?? 42,
                  splitSeed: try values.decodeIfPresent(Int.self, forKey: .splitSeed) ?? 42,
                  validationPercent: try values.decodeIfPresent(Int.self, forKey: .validationPercent) ?? 20,
                  learningRate: try values.decodeIfPresent(Double.self, forKey: .learningRate) ?? 0.03,
                  preparedTraining: try values.decodeIfPresent(PreparedTrainingSelection.self, forKey: .preparedTraining),
                  compositeSource: try values.decodeIfPresent(CompositeSourceReference.self, forKey: .compositeSource),
                  gradientCapture: try values.decodeIfPresent(GradientCaptureConfiguration.self, forKey: .gradientCapture),trainingControl:try values.decodeIfPresent(TrainingControlPolicy.self,forKey:.trainingControl))
    }
}

public struct GraphDocument: Codable, Equatable {
    public var namedInputs:[String:String]? = nil
    public var compositionPlan: CompositionPlan?
    public var schemaVersion: Int
    public var name: String
    public var nodes: [GraphNode]
    public var edges: [GraphEdge]
    public var execution: ExecutionSettings?
    private enum CodingKeys: String, CodingKey { case schemaVersion, name, nodes, edges, execution, compositionPlan, namedInputs }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        namedInputs = try c.decodeIfPresent([String:String].self,forKey:.namedInputs)
        schemaVersion = try c.decode(Int.self, forKey: .schemaVersion)
        name = try c.decode(String.self, forKey: .name)
        nodes = try c.decode([GraphNode].self, forKey: .nodes)
        edges = try c.decode([GraphEdge].self, forKey: .edges)
        execution = try c.decodeIfPresent(ExecutionSettings.self, forKey: .execution)
        compositionPlan = c.contains(.compositionPlan) ? try c.decode(CompositionPlan.self, forKey: .compositionPlan) : nil
    }
    public init(schemaVersion: Int = 2, name: String = "Untitled flow", nodes: [GraphNode] = [], edges: [GraphEdge] = [], execution: ExecutionSettings? = nil, compositionPlan: CompositionPlan? = nil) {
        self.compositionPlan = compositionPlan
        self.schemaVersion = schemaVersion; self.name = name; self.nodes = nodes; self.edges = edges; self.execution = execution
    }
}
public struct GraphNode: Codable, Identifiable, Equatable {
    public var parameterSettings:[String:JSONValue]? = nil
    public var id: String
    public var type: String
    public var name: String
    public var x: Double
    public var y: Double
    public var config: [String: JSONValue]
    public init(id: String = UUID().uuidString, type: String = "input", name: String = "", x: Double = 0, y: Double = 0, config: [String: JSONValue] = [:], parameterSettings:[String:JSONValue]? = nil) {
        self.parameterSettings=parameterSettings
        self.id = id; self.type = type; self.name = name; self.x = x; self.y = y; self.config = config
        if type == "linear" && self.config["bias"] == nil { self.config["bias"] = .bool(true) }
    }
}
public struct GraphEdge: Codable, Identifiable, Equatable {
    public var id: String
    public var source: String
    public var sourcePort: String
    public var target: String
    public var targetPort: String
    public init(id: String = UUID().uuidString, source: String = "", sourcePort: String = "out", target: String = "", targetPort: String = "in") {
        self.id = id; self.source = source; self.sourcePort = sourcePort; self.target = target; self.targetPort = targetPort
    }
}
public struct ComponentFieldChoice: Codable, Equatable {
    public var value: String
    public var title: String
    public init(value: String, title: String) { self.value = value; self.title = title }
}

public struct ComponentField: Codable, Equatable {
    public var key:String
    public var title:String
    public var kind:ComponentFieldKind
    public var defaultValue:JSONValue
    public var min:Double?
    public var max:Double?
    public var minItems:Int?
    public var maxItems:Int?
    public var allowsNull:Bool
    /// Optional concrete starting value when a nullable field is enabled.
    public var enabledValue:JSONValue?
    public var choices:[ComponentFieldChoice]?
    public init(key:String="",title:String="",kind:ComponentFieldKind = .integer,defaultValue:JSONValue=1,min:Double?=1,max:Double?=512,minItems:Int?=nil,maxItems:Int?=nil,allowsNull:Bool=false,choices:[ComponentFieldChoice]?=nil,enabledValue:JSONValue?=nil) {
        self.key=key;self.title=title;self.kind=kind;self.defaultValue=defaultValue;self.min=min;self.max=max;self.minItems=minItems;self.maxItems=maxItems;self.allowsNull=allowsNull;self.choices=choices;self.enabledValue=enabledValue
    }
    private enum CodingKeys:String,CodingKey {case key,title,kind,defaultValue,min,max,minItems,maxItems,allowsNull,choices,enabledValue}
    public func encode(to encoder:Encoder)throws {
        var c=encoder.container(keyedBy:CodingKeys.self)
        try c.encode(key,forKey:.key);try c.encode(title,forKey:.title)
        try c.encode(kind,forKey:.kind);try c.encode(defaultValue,forKey:.defaultValue)
        try c.encodeIfPresent(min,forKey:.min);try c.encodeIfPresent(max,forKey:.max)
        try c.encodeIfPresent(minItems,forKey:.minItems);try c.encodeIfPresent(maxItems,forKey:.maxItems)
        if allowsNull {try c.encode(true,forKey:.allowsNull)}
        try c.encodeIfPresent(choices,forKey:.choices)
        try c.encodeIfPresent(enabledValue,forKey:.enabledValue)
    }

    public init(from decoder:Decoder)throws {
        let c=try decoder.container(keyedBy:CodingKeys.self)
        key=try c.decode(String.self,forKey:.key);title=try c.decode(String.self,forKey:.title)
        kind=try c.decodeIfPresent(ComponentFieldKind.self,forKey:.kind) ?? .integer
        defaultValue=try c.decode(JSONValue.self,forKey:.defaultValue)
        min=try c.decodeIfPresent(Double.self,forKey:.min);max=try c.decodeIfPresent(Double.self,forKey:.max)
        minItems=try c.decodeIfPresent(Int.self,forKey:.minItems);maxItems=try c.decodeIfPresent(Int.self,forKey:.maxItems)
        allowsNull=try c.decodeIfPresent(Bool.self,forKey:.allowsNull) ?? false
        choices=try c.decodeIfPresent([ComponentFieldChoice].self,forKey:.choices)
        enabledValue=try c.decodeIfPresent(JSONValue.self,forKey:.enabledValue)
    }
}
public struct ComponentDefinition: Codable, Identifiable, Equatable {
    public var id: String
    public var title: String
    public var category: String
    public var inputs: [String]
    public var outputs: [String]
    public var fields: [ComponentField]
    public var configurationAlternatives:[String:[ComponentField]]? = nil
    public func fields(configuration:[String:JSONValue])->[ComponentField] {
        configurationAlternatives?.values.first(where:{Set($0.map(\.key))==Set(configuration.keys)}) ?? fields
    }
    public init(id: String = "", title: String = "", category: String = "", inputs: [String] = [], outputs: [String] = [], fields: [ComponentField] = []) {
        self.id = id; self.title = title; self.category = category; self.inputs = inputs; self.outputs = outputs; self.fields = fields
    }
}
public struct ComponentRegistry: Codable, Equatable {
    public var schemaVersion: Int
    public var components: [ComponentDefinition]
    public init(schemaVersion: Int = 2, components: [ComponentDefinition] = []) { self.schemaVersion = schemaVersion; self.components = components }
    public static func builtIn() throws -> ComponentRegistry {
        guard let url = Bundle.module.url(forResource: "components", withExtension: "json") else { throw CocoaError(.fileNoSuchFile) }
        return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
    public func component(type: String) -> ComponentDefinition? { components.first { $0.id == type } }
}
public struct ValidationIssue: Identifiable, Equatable {
    public var id: String
    public var message: String
    public var nodeID: String?
    public init(id: String = UUID().uuidString, message: String = "", nodeID: String? = nil) { self.id = id; self.message = message; self.nodeID = nodeID }
}
public struct ValidationResult: Equatable {
    public var issues: [ValidationIssue]
    public var widths: [String: Int]
    public var outputTensorContracts: [GraphPortAddress: TypedTensorContract] = [:]
    public var outputBagContracts: [GraphPortAddress: BagPortContract] = [:]
    /// Legacy default-output view; named outputs retain independent types and shapes.
    public var tensorContracts: [String: TypedTensorContract] {
        get { Dictionary(uniqueKeysWithValues: outputTensorContracts.compactMap { address, value in
            address.portID == "out" ? (address.nodeID, value) : nil
        }) }
        set {
            outputTensorContracts = outputTensorContracts.filter { $0.key.portID != "out" }
            for (node, value) in newValue { outputTensorContracts[.init(node, "out")] = value }
        }
    }
    /// Compatibility projection. Shape and type have one authoritative owner.
    public var tensorLayouts: [String: BatchedTensorLayout] {tensorContracts.mapValues(\.layout)}
    public var orderedNodeIDs: [String]
    public var isValid: Bool { issues.isEmpty }
    public init(issues: [ValidationIssue] = [], widths: [String: Int] = [:], orderedNodeIDs: [String] = []) { self.issues = issues; self.widths = widths; self.orderedNodeIDs = orderedNodeIDs }
}
public enum ExampleFlows {
    public static var parallelClassifier: GraphDocument {
        GraphDocument(name: "Parallel classifier", nodes: [
            GraphNode(id: "input", type: "input", name: "Measurements", x: 30, y: 205, config: ["features": 2]),
            GraphNode(id: "branch-a", type: "linear", name: "Branch A", x: 215, y: 110, config: ["in_features": 2, "out_features": 8]),
            GraphNode(id: "activation-a", type: "relu", name: "Activation A", x: 400, y: 110),
            GraphNode(id: "branch-b", type: "linear", name: "Branch B", x: 215, y: 300, config: ["in_features": 2, "out_features": 8]),
            GraphNode(id: "activation-b", type: "tanh", name: "Activation B", x: 400, y: 300),
            GraphNode(id: "merge", type: "concat", name: "Combine", x: 585, y: 205),
            GraphNode(id: "classifier", type: "linear", name: "Classifier", x: 770, y: 205, config: ["in_features": 16, "out_features": 2]),
            GraphNode(id: "output", type: "output", name: "Prediction", x: 955, y: 205)
        ], edges: [
            GraphEdge(id: "e1", source: "input", target: "branch-a"), GraphEdge(id: "e2", source: "input", target: "branch-b"),
            GraphEdge(id: "e3", source: "branch-a", target: "activation-a"), GraphEdge(id: "e4", source: "branch-b", target: "activation-b"),
            GraphEdge(id: "e5", source: "activation-a", target: "merge", targetPort: "a"), GraphEdge(id: "e6", source: "activation-b", target: "merge", targetPort: "b"),
            GraphEdge(id: "e7", source: "merge", target: "classifier"), GraphEdge(id: "e8", source: "classifier", target: "output")
        ])
    }
}
