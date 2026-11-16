import Foundation

/// Authored JSON is the single source for these unscored public exercises.
public enum DataEngineeringPractice {
    public struct Lesson: Codable, Identifiable {
        public let id: String
        public let title: String
        public let objective: String
        public let terms: String
        public let steps: [String]
        public let inspect: String
        public let operators: [String]
        public let exampleIDs: [String]
        public let operatorID: String
        public let graph: String
        public let source: String
        public let sourceNode: String
        public let reading: String
        public let mode: String
        public let isAssessed: Bool
        public let contentVersion: Int
        public let authoredAt: String
        public let disclosure: String
        public let prediction: String
        public let limits: String
    }
    public struct Guidance: Codable, Identifiable {
        public let id: String
        public let title: String
        public let objective: String
        public let terms: String
        public let steps: [String]
        public let inspect: String
    }
    public struct Catalog: Codable {
        public let schemaVersion: Int
        public let family: String
        public let lessons: [Lesson]
        public let sharedGuidance: [Guidance]
    }
    private static let loaded: Result<Catalog, Error> = Result { try loadCatalog() }
    private static func loadCatalog() throws -> Catalog {
        let url = try bundledURL("data02-lessons.json")
        let data = try Data(contentsOf: url)
        guard data.count <= 500_000 else { throw DataFlowContractError("Data practice catalog exceeds its reading budget.") }
        let catalog = try JSONDecoder().decode(Catalog.self, from: data)
        let operators = Set(["filterRows", "formula", "outputDelimited"].map { "kwip.data." + $0 })
        guard catalog.schemaVersion == 1, catalog.family == "data-engineering", catalog.lessons.count == 6,
              Set(catalog.lessons.map(\.id)).count == 6, Set(catalog.lessons.map(\.operatorID)) == operators,
              catalog.sharedGuidance.count == 3, Set(catalog.sharedGuidance.map(\.id)).count == 3 else {
            throw DataFlowContractError("Data practice catalog does not match the reviewed six-exercise scope.")
        }
        for lesson in catalog.lessons {
            guard lesson.id.hasPrefix("data02-"), !lesson.isAssessed, lesson.mode == "guidedPractice",
                  lesson.exampleIDs == [lesson.id], lesson.contentVersion > 0, !lesson.authoredAt.isEmpty,
                  !lesson.title.isEmpty, !lesson.objective.isEmpty, !lesson.terms.isEmpty, !lesson.steps.isEmpty,
                  !lesson.disclosure.isEmpty, !lesson.prediction.isEmpty, !lesson.limits.isEmpty,
                  [lesson.graph, lesson.source, lesson.reading].allSatisfy(safeResourceName) else {
                throw DataFlowContractError("A bundled data exercise has invalid teaching or resource metadata.")
            }
        }
        return catalog
    }
    public static func catalog() throws -> Catalog { try loaded.get() }
    public static func lesson(_ id: String) throws -> Lesson {
        guard let lesson = try catalog().lessons.first(where: { $0.id == id }) else {
            throw DataFlowContractError("That data practice is not included in this build.")
        }
        return lesson
    }
    private static func safeResourceName(_ name: String) -> Bool {
        name.hasPrefix("data02-") && !name.contains("/") && !name.contains("\\") && !name.contains("\0") && name.utf8.count <= 200
    }
    private static func bundledURL(_ name: String) throws -> URL {
        guard safeResourceName(name), let url = Bundle.module.url(forResource: name, withExtension: nil) else {
            throw DataFlowContractError("The bundled data practice resource is missing. Reinstall the complete application.")
        }
        return url
    }
    public static func resourceURL(_ name: String) throws -> URL {
        let catalog = try catalog()
        let known = Set(catalog.lessons.flatMap { [$0.graph, $0.source, $0.reading] }).union(["data02-lessons.json", "data02-shared-guidance.md"])
        guard known.contains(name) else { throw DataFlowContractError("Unknown data practice resource.") }
        return try bundledURL(name)
    }
    public static func document(_ id: String, projectID: String = UUID().uuidString) throws -> WorkflowDocument {
        let lesson = try lesson(id)
        let graph = try DataFlowCodec.decodeGraph(Data(contentsOf: resourceURL(lesson.graph)))
        guard graph.nodes.contains(where: { $0.id == lesson.sourceNode && $0.operatorID == "kwip.data.sourceCSV" }),
              graph.nodes.filter({ $0.operatorID == lesson.operatorID }).count == 1 else {
            throw DataFlowContractError("Data practice graph does not match its teaching and source declarations.")
        }
        let nodes = graph.nodes.enumerated().map { index, node in
            WorkflowNode(address: .init(domain: .data, localID: node.id),
                name: node.id == lesson.sourceNode ? "Eight sample orders" : node.operatorID == "kwip.data.columns" ? "Read quantities and prices as numbers" : lesson.title,
                x: Double(40 + index * 245), y: 100, payload: .dataOperator(operatorID: node.operatorID, config: node.config))
        }
        let edges = graph.edges.enumerated().map { index, edge in
            WorkflowEdge(id: "data02-connection-" + String(index),
                from: .init(address: .init(domain: .data, localID: edge.from.node), port: edge.from.port),
                to: .init(address: .init(domain: .data, localID: edge.to.node), port: edge.to.port), kind: .table)
        }
        let document = WorkflowDocument(projectID: projectID, name: lesson.title, nodes: nodes, edges: edges)
        try WorkflowCodec.validate(document)
        return document
    }
    public static func isExample(_ id: String) -> Bool { (try? lesson(id)) != nil }
}
