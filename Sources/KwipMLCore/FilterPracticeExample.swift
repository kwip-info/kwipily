import Foundation

/// Public rows make both branches, missing-value handling and text matching inspectable.
public enum FilterPracticeExample {
    public static let exampleID = "filter-rows"
    public static let lessonID = "filter"
    public static let observationsResource = "filter-observations"
    public static let filterAddress = WorkflowNodeAddress(domain: .data, localID: "filter")
    public static let matchingAddress = WorkflowNodeAddress(domain: .data, localID: "matching-view")
    public static let nonmatchingAddress = WorkflowNodeAddress(domain: .data, localID: "nonmatching-view")
    public static let configuration: [String: JSONValue] = [
        "mode": .string("simple"),
        "conditions": .array([.object(["column": .string("amount"), "comparison": .string("ge"), "value": .integer(10), "caseSensitive": .bool(true)])]),
        "match": .string("all"), "expression": .string("[amount] >= 10"),
        "column": .string("status"), "pattern": .string("^ready$"),
        "matchMode": .string("full"), "flags": .array([.string("ignoreCase")]),
        "nullPolicy": .string("nonmatching")
    ]
    public static var sourceConfiguration: [String: JSONValue] {
        var result = CSVParsingConfiguration.newDefaults
        result["emptyField"] = .string("null")
        result["schemaOverrides"] = .object(["amount": .string("Float64")])
        return result
    }
    public static func document(projectID: String = UUID().uuidString) throws -> WorkflowDocument {
        let document = WorkflowDocument(projectID: projectID, name: "Choose rows and inspect both Filter branches", nodes: [
            .init(address: WorkflowExamples.sourceAddress, name: "Eight synthetic observations", x: 80, y: 180,
                  payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: sourceConfiguration)),
            .init(address: filterAddress, name: "Amounts of at least ten", x: 420, y: 180,
                  payload: .dataOperator(operatorID: "kwip.data.filterRows", config: configuration)),
            .init(address: matchingAddress, name: "Explore matching rows", x: 760, y: 80,
                  payload: .dataOperator(operatorID: "kwip.data.view", config: [:])),
            .init(address: nonmatchingAddress, name: "Explore nonmatching rows", x: 760, y: 330,
                  payload: .dataOperator(operatorID: "kwip.data.view", config: [:]))
        ], edges: [
            .init(id: "source-to-filter", from: .init(address: WorkflowExamples.sourceAddress, port: "table"), to: .init(address: filterAddress, port: "table"), kind: .table),
            .init(id: "filter-matching", from: .init(address: filterAddress, port: "matching"), to: .init(address: matchingAddress, port: "table"), kind: .table),
            .init(id: "filter-nonmatching", from: .init(address: filterAddress, port: "nonmatching"), to: .init(address: nonmatchingAddress, port: "table"), kind: .table)
        ])
        try WorkflowCodec.validate(document)
        return document
    }
}
