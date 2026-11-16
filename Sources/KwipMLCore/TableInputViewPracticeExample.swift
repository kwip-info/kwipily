import Foundation

/// Two editable canvas tools with public bundled observations and no private source paths.
public enum TableInputViewPracticeExample {
    public static let exampleID = "table-view"
    public static let viewAddress = WorkflowNodeAddress(domain: .data, localID: "view")
    public static func document(projectID: String = UUID().uuidString) throws -> WorkflowDocument {
        let document = WorkflowDocument(projectID: projectID, name: "Read and explore a table", nodes: [
            .init(address: WorkflowExamples.sourceAddress, name: "Synthetic signal observations", x: 80, y: 120,
                  payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: CSVParsingConfiguration.newDefaults)),
            .init(address: viewAddress, name: "Explore the complete captured table", x: 420, y: 120,
                  payload: .dataOperator(operatorID: "kwip.data.view", config: [:]))
        ], edges: [
            .init(id: "table-to-view", from: .init(address: WorkflowExamples.sourceAddress, port: "table"),
                  to: .init(address: viewAddress, port: "table"), kind: .table)
        ])
        try WorkflowCodec.validate(document)
        return document
    }
}
