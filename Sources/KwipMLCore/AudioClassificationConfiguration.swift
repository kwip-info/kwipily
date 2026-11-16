import Foundation
public enum AudioClassificationConfiguration {
    public static let operatorID="kwip.data.classifyAudio"
    public static func validate(_ config:[String:JSONValue],definition:DataFlowOperatorDefinition)throws {
        try definition.validateConfiguration(config)
        guard let input=config["inputColumn"]?.stringValue,let prefix=config["outputPrefix"]?.stringValue,!input.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,!prefix.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,let window=config["windowSeconds"]?.doubleValue,let hop=config["hopSeconds"]?.doubleValue,hop<=window else {throw DataFlowContractError("Choose nonblank waveform and output names, and a hop no longer than its window.")}
    }
}
