import Foundation

/// Public synthetic collection. Its private on-device directory is bound by the example loader.
public enum FileInputViewPracticeExample {
    public static let exampleID = "file-view"
    public static let lessonID = "file-input"
    public static let sourceAddress = WorkflowExamples.sourceAddress
    public static let viewAddress = WorkflowNodeAddress(domain: .data, localID: "view")
    public static var resourceNames: [String] {
        (1...10).map { String(format: "file-practice-%02d-notes.txt", $0) }
        + ["file-practice-11-checklist.docx", "file-practice-12-colors.png"]
    }
    public static func document(projectID: String = UUID().uuidString) throws -> WorkflowDocument {
        var config = FileInputConfiguration.newDefaults
        config["sourceMode"] = .string("directory")
        let document = WorkflowDocument(projectID: projectID, name: "Read files and explore their contents", nodes: [
            .init(address: sourceAddress, name: "Twelve synthetic files", x: 80, y: 120,
                  payload: .dataOperator(operatorID: "kwip.data.sourceFiles", config: config)),
            .init(address: viewAddress, name: "Explore captured file contents", x: 420, y: 120,
                  payload: .dataOperator(operatorID: "kwip.data.view", config: [:]))
        ], edges: [
            .init(id: "files-to-view", from: .init(address: sourceAddress, port: "table"), to: .init(address: viewAddress, port: "table"), kind: .table)
        ])
        try WorkflowCodec.validate(document)
        return document
    }
}
