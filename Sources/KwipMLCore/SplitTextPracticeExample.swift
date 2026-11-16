import Foundation

public enum SplitTextPracticeExample {
    public static let exampleID = "split-text"
    public static let lessonID = "splitText"
    public static let observationsResource = "split-text-observations"
    public static let splitAddress = WorkflowNodeAddress(domain: .data, localID: "split")
    public static let viewAddress = WorkflowNodeAddress(domain: .data, localID: "view")
    public static let configuration: [String: JSONValue] = [
        "mode": .string("delimited"), "source": .string("address"), "missingSource": .string("preserve"), "retainSource": .bool(false),
        "delimiter": .string("|"), "quoted": .bool(true), "quote": .string("\""), "escaping": .string("double"), "escape": .string("\\"),
        "outputMode": .string("columns"), "fields": .array(["street", "city", "postal_code"].map { .object(["name": .string($0), "type": .string("string"), "start": .integer(1), "width": .integer(1)]) }),
        "extraFields": .string("error"), "shortFields": .string("missing"), "emptyTokens": .string("keep"), "ordinal": .bool(false), "ordinalName": .string("token_ordinal"),
        "measurement": .string("codepoints"), "shortRecord": .string("missing"), "trim": .bool(false), "conversionFailure": .string("error")]
    public static func document(projectID: String = UUID().uuidString) throws -> WorkflowDocument {
        let document = WorkflowDocument(projectID: projectID, name: "Separate address fields with Split Text", nodes: [
            .init(address: WorkflowExamples.sourceAddress, name: "Three synthetic addresses", x: 80, y: 120, payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: CSVParsingConfiguration.newDefaults)),
            .init(address: splitAddress, name: "Extract street, city and postal code", x: 420, y: 120, payload: .dataOperator(operatorID: "kwip.data.splitText", config: configuration)),
            .init(address: viewAddress, name: "Inspect separated fields", x: 760, y: 120, payload: .dataOperator(operatorID: "kwip.data.view", config: [:]))
        ], edges: [
            .init(id: "source-to-split", from: .init(address: WorkflowExamples.sourceAddress, port: "table"), to: .init(address: splitAddress, port: "table"), kind: .table),
            .init(id: "split-to-view", from: .init(address: splitAddress, port: "table"), to: .init(address: viewAddress, port: "table"), kind: .table)])
        try WorkflowCodec.validate(document)
        return document
    }
}
