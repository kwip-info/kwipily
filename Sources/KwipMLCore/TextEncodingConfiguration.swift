import Foundation

public enum TextEncodingConfiguration {
    public static let operatorID="kwip.data.textEncode"
    public static func validate(_ values:[String:JSONValue],definition:DataFlowOperatorDefinition)throws {
        try definition.validateConfiguration(values)
        if values["mode"] == .string("documentFeatures") {
            guard values["vocabularySource"] == .string("training"),values["keepSourceText"] == true,case .object(let settings)=values["featureSettings"] else {throw DataFlowContractError("Document features fit on Training and retain source text with their processor.")}
            try DocumentClassificationConfiguration.validateFeatures(settings)
        } else if values["mode"] != nil && values["mode"] != .string("tokens") {throw DataFlowContractError("Choose token IDs or document features.")}
        let supplied=values["vocabularySource"] == .string("supplied")
        guard case .array(let vocabulary)=values["vocabulary"],supplied ? !vocabulary.isEmpty:vocabulary.isEmpty else {throw DataFlowContractError("Fit on Training or supply a nonempty vocabulary, never both.")}
        let reserved=(values["unknownPolicy"] == .string("unk") ? 1:0)+["pad","bos","eos"].filter {values[$0] == .bool(true)}.count
        guard let limit=values["vocabularyLimit"]?.integerValue,limit>=reserved,!supplied || vocabulary.count+reserved<=limit else {throw DataFlowContractError("Vocabulary limit includes its reserved special tokens.")}
        for key in ["textColumn","outputColumn"] {guard let name=values[key]?.stringValue,!name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,!name.contains("\0") else {throw DataFlowContractError("Choose nonblank text and token column names.")}}
    }
}
