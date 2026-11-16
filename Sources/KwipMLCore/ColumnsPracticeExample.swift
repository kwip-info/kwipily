import Foundation

/// Public six-row practice. The authored grid deliberately changes all four column choices.
public enum ColumnsPracticeExample {
    public static let exampleID = "columns-grid"
    public static let lessonID = "columns"
    public static let observationsResource = "columns-observations"
    public static let columnsAddress = WorkflowNodeAddress(domain: .data, localID: "columns")
    public static let viewAddress = WorkflowNodeAddress(domain: .data, localID: "view")
    public static let configuration: [String: JSONValue] = [
        "rows": .array([
            .object(["source": .string("score_text"), "included": .bool(true), "outputName": .string("score"), "targetType": .string("float64")]),
            .object(["source": .string("person_id"), "included": .bool(true), "outputName": .string("person_id"), "targetType": .string("keep")]),
            .object(["source": .string("active_text"), "included": .bool(true), "outputName": .string("active"), "targetType": .string("bool")]),
            .object(["source": .string("notes"), "included": .bool(false), "outputName": .string("notes"), "targetType": .string("keep")])
        ]),
        "castFailure": .string("error"), "trimText": .bool(false),
        "newColumns": .string("drop"), "missingPolicy": .string("preserve")
    ]
    public static func document(projectID: String = UUID().uuidString) throws -> WorkflowDocument {
        let document = WorkflowDocument(projectID: projectID, name: "Choose names, order and types with Columns", nodes: [
            .init(address: WorkflowExamples.sourceAddress, name: "Six synthetic observations", x: 80, y: 120,
                  payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: CSVParsingConfiguration.newDefaults)),
            .init(address: columnsAddress, name: "Choose model-ready columns", x: 420, y: 120,
                  payload: .dataOperator(operatorID: "kwip.data.columns", config: configuration)),
            .init(address: viewAddress, name: "Explore the changed table", x: 760, y: 120,
                  payload: .dataOperator(operatorID: "kwip.data.view", config: [:]))
        ], edges: [
            .init(id: "source-to-columns", from: .init(address: WorkflowExamples.sourceAddress, port: "table"), to: .init(address: columnsAddress, port: "table"), kind: .table),
            .init(id: "columns-to-view", from: .init(address: columnsAddress, port: "table"), to: .init(address: viewAddress, port: "table"), kind: .table)
        ])
        try WorkflowCodec.validate(document)
        return document
    }
}
