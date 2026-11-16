import Foundation
public enum SpeechTranscriptionConfiguration {
    public static let operatorID="kwip.data.transcribeAudio"
    public static func validate(_ config:[String:JSONValue],definition:DataFlowOperatorDefinition)throws {
        try definition.validateConfiguration(config)
        guard let input=config["inputColumn"]?.stringValue,let output=config["outputColumn"]?.stringValue,!input.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,!output.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,input != output else {throw DataFlowContractError("Choose distinct nonblank waveform and transcript columns.")}
    }
}
