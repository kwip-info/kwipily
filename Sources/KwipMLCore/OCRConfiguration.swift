import Foundation

public enum OCRConfiguration {
    public static func defaults()throws->[String:JSONValue] {
        guard let url=Bundle.module.url(forResource:"ocr-options",withExtension:"json") else {throw DataFlowContractError("OCR defaults are unavailable.")}
        return try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:url))
    }
    public static let operatorID="kwip.data.extractText"
    public static func validate(_ config:[String:JSONValue],definition:DataFlowOperatorDefinition)throws {
        try definition.validateConfiguration(config)
        guard config["pack"] == .string(OCRPackCatalog.id),config["device"] == .string("cpu"),
              let source=config["inputColumn"]?.stringValue,!source.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,
              let output=config["outputColumn"]?.stringValue,!output.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,source != output,
              config["allowlist"] == .string("") || config["blocklist"] == .string("") else {throw DataFlowContractError("Choose verified English OCR, distinct source/text columns, and only one character filter.")}
    }
}
