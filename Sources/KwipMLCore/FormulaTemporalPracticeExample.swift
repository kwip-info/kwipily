import Foundation

public enum FormulaTemporalPracticeExample {
    public static let exampleID = "formula-temporal"
    public static let observationsResource = "formula-temporal-observations"
    public static let address = WorkflowNodeAddress(domain: .data, localID: "formula-temporal")
    public static func observationsURL() throws -> URL {
        guard let url = Bundle.module.url(forResource: observationsResource, withExtension: "csv") else { throw DataFlowContractError("Date/time practice observations are missing.") }
        return url
    }
    private static func operation(_ id: String, _ expression: String, _ name: String, _ dtype: String) -> JSONValue {
        .object(["id": .string(id), "expression": .string(expression), "outputs": .array([.object(["name": .string(name), "dtype": .string(dtype), "write": .string("new")])]), "nullPolicy": .string("preserve")])
    }
    public static var configuration: [String: JSONValue] {
        ["operations": .array([
            operation("earliest", "temporal_parse([appointment], \"datetime\", \"iso\", \"America/New_York\", \"UTC\", \"stop\", \"earliest\", \"missing\", \"v1\")", "earliest", "datetime"),
            operation("latest", "temporal_parse([appointment], \"datetime\", \"iso\", \"America/New_York\", \"UTC\", \"stop\", \"latest\", \"missing\", \"v1\")", "latest", "datetime"),
            operation("elapsed", "temporal_difference([latest], [earliest], \"seconds\", \"integer\")", "elapsed_seconds", "int64"),
            operation("display", "temporal_format([earliest], \"iso\", \"UTC\")", "utc_text", "string")
        ])]
    }
    public static func document(projectID: String = UUID().uuidString) throws -> WorkflowDocument {
        let view = WorkflowNodeAddress(domain: .data, localID: "view")
        let document = WorkflowDocument(projectID: projectID, name: "Compare local appointments with Formula", nodes: [
            .init(address: WorkflowExamples.sourceAddress, name: "Synthetic local appointments", x: 80, y: 200, payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: CSVParsingConfiguration.newDefaults)),
            .init(address: address, name: "Resolve repeated local times", x: 420, y: 200, payload: .dataOperator(operatorID: "kwip.data.formula", config: configuration)),
            .init(address: view, name: "Inspect instants and elapsed seconds", x: 760, y: 200, payload: .dataOperator(operatorID: "kwip.data.view", config: [:]))
        ], edges: [
            .init(from: .init(address: WorkflowExamples.sourceAddress, port: "table"), to: .init(address: address, port: "table"), kind: .table),
            .init(from: .init(address: address, port: "table"), to: .init(address: view, port: "table"), kind: .table)
        ])
        try WorkflowCodec.validate(document); return document
    }
}
