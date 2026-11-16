import Foundation

public enum SklearnEstimatorConfiguration {
    public static let defaultSettings: [String: JSONValue] = ["C": .integer(1), "maxIterations": .integer(200), "tolerance": .number(0.0001), "fitIntercept": .bool(true)]
    public static func validate(_ settings: [String: JSONValue]) throws {
        guard Set(settings.keys) == Set(defaultSettings.keys),
              let c = settings["C"]?.doubleValue, c.isFinite, (1e-8...1e8).contains(c),
              let tolerance = settings["tolerance"]?.doubleValue, tolerance.isFinite, (1e-12...1).contains(tolerance),
              case .integer(let iterations) = settings["maxIterations"], (1...10000).contains(iterations),
              case .bool = settings["fitIntercept"] else {
            throw DataFlowContractError("Logistic regression requires bounded C, tolerance, integer iterations and Boolean intercept.")
        }
    }
}

public struct WorkflowEstimatorConfiguration: Equatable {
    public var algorithm:String? = nil
    public var task: String? = nil
    public var reviewThreshold:Double? = nil
    public var operation:String = "fit"
    public var isPrebuilt:Bool {task == "prebuilt"}
    public var hasTableOutputs:Bool {isPrebuilt || isDocumentClassification}
    public var isPrediction:Bool {hasTableOutputs && operation != "fit"}
    public var isDocumentClassification:Bool {task == DocumentClassificationConfiguration.task}
    public let backendID: String
    public let featureColumns: [String]
    public let targetColumn: String
    public let settings: [String: JSONValue]
    public static func decode(_ values: [String: JSONValue]) throws -> Self {
        if PrebuiltModelConfiguration.isPrebuilt(values) {return try PrebuiltModelConfiguration.decode(values)}
        let documentTask=DocumentClassificationConfiguration.isDocument(values)
        guard Set(values.keys) == Set(["backendID", "featureColumns", "targetColumn", "settings"]+(documentTask ? ["task"]+(values["operation"] != nil ? ["operation"]:[])+(values["reviewThreshold"] != nil ? ["reviewThreshold"]:[]):[])),
              case .string(let backend) = values["backendID"], ["sklearn", "tensorflow"].contains(backend),
              case .array(let columns) = values["featureColumns"], case .string(let target) = values["targetColumn"],
              case .object(let settings) = values["settings"] else { throw DataFlowContractError("Invalid backend estimator configuration.") }
        let operation=values["operation"]?.stringValue ?? "fit"
        guard !documentTask || ["fit","predict","evaluate"].contains(operation) else {throw DataFlowContractError("Choose Fit, Predict or Evaluate for document classification.")}
        let threshold=values["reviewThreshold"]?.doubleValue
        if let raw=values["reviewThreshold"] {guard let n=raw.doubleValue,n.isFinite,n>0,n<=1,documentTask else {throw DataFlowContractError("Review threshold must be a probability above 0 and at most 1.")}}
        let names = columns.compactMap(\.stringValue)
        guard names.count == columns.count, names.count == (documentTask ? 1:2) else { throw DataFlowContractError(documentTask ? "Select one document text column.":"Select exactly two ordered feature columns.") }
        try TrainingColumnMapping.validate(features: names, target: target)
        if documentTask {guard backend == "sklearn" else {throw DataFlowContractError("Document classification uses the reviewed scikit-learn adapter.")};try DocumentClassificationConfiguration.validateEstimator(settings)}
        else if backend == "sklearn" { try SklearnEstimatorConfiguration.validate(settings) }
        // The concrete TensorFlow adapter validates its settings before compilation/launch.
        var result=Self(backendID: backend, featureColumns: names, targetColumn: target, settings: settings)
        result.reviewThreshold=threshold
        result.operation=operation
        result.task=documentTask ? DocumentClassificationConfiguration.task:nil
        return result
    }
    public static var sklearnDefaults: [String: JSONValue] {
        ["backendID": .string("sklearn"), "featureColumns": .array([.string(""), .string("")]), "targetColumn": .string(""), "settings": .object(SklearnEstimatorConfiguration.defaultSettings)]
    }
}
