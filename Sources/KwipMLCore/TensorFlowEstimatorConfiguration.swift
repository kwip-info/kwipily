import Foundation

/// Reviewed foundation subset; additional adapters must explicitly extend this boundary.
public enum TensorFlowEstimatorConfiguration {
    public static let supportedComponents:Set<String>=["input","linear","relu","tanh","add","concat","output"]
    public static let defaultSettings:[String:JSONValue]=["seed":.integer(42),"epochs":.integer(30),"batchSize":.integer(16),"learningRate":.number(0.03),"device":.string("cpu")]
    public static let fields:[WorkflowRunField]=[
        .init(key:"device",title:"Device",kind:.choice(["cpu"]),defaultValue:.string("cpu")),
        .init(key:"epochs",title:"Epochs",kind:.integer(1...1000),defaultValue:.integer(30)),
        .init(key:"batchSize",title:"Batch size",kind:.integer(1...100000),defaultValue:.integer(16)),
        .init(key:"seed",title:"Seed",kind:.integer(0...4294967295),defaultValue:.integer(42)),
        .init(key:"learningRate",title:"Learning rate",kind:.positiveDecimal(maximum:1),defaultValue:.number(0.03))]
    public static func validate(_ settings:[String:JSONValue])throws {
        guard Set(settings.keys)==Set(defaultSettings.keys),fields.allSatisfy({field in settings[field.key].map(field.accepts) ?? false}) else {
            throw DataFlowContractError("TensorFlow requires explicit CPU, seed, epochs (1–1000), batch size (1–100000) and learning rate (0–1 exclusive of zero).")
        }
    }
    public static func validateGraph(_ graph:GraphDocument,registry:ComponentRegistry)throws {
        guard graph.schemaVersion==2,graph.nodes.allSatisfy({supportedComponents.contains($0.type)}) else {
            throw DataFlowContractError("This TensorFlow foundation supports Input, Linear, ReLU, Tanh, Add, Concatenate and Output in schema 2.")
        }
        let result=GraphValidator.validate(graph,registry:registry)
        guard result.isValid else {throw DataFlowContractError(result.issues.map(\.message).joined(separator:" "))}
    }
}
