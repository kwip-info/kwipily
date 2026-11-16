import XCTest
@testable import KwipMLCore

final class ProjectLibraryTests: XCTestCase {
    var directory: URL!
    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: directory) }
    func testRoundTripInvalidDraftAndStableIdentity() throws {
        let library = ProjectLibrary(directory: directory)
        var graph = ExampleFlows.parallelClassifier
        graph.name = "My research"
        graph.nodes[0].name = "Personal measurements"
        graph.edges.removeLast()
        let first = try library.save(graph)
        XCTAssertEqual(try library.load(id: first.id), graph)
        graph.name = "Renamed research"
        let second = try library.save(graph, id:first.id)
        XCTAssertEqual(second.id,first.id)
        XCTAssertEqual(second.name,graph.name)
        XCTAssertEqual(try library.list().count,1)
        XCTAssertEqual(try library.load(id:first.id),graph)
    }
    func testHistoryOrderingAndIndependentProjects() throws {
        let library = ProjectLibrary(directory:directory)
        var a = ExampleFlows.parallelClassifier; a.name = "A"
        var b = a; b.name = "B"
        let first = try library.save(a)
        let second = try library.save(b)
        XCTAssertEqual(try library.list().map(\.id),[second.id,first.id])
        a.name = "A revised"
        _ = try library.save(a,id:first.id)
        XCTAssertEqual(try library.list().map(\.id),[first.id,second.id])
        XCTAssertEqual(try ProjectLibrary(directory:directory).load(id:second.id),b)
    }
    func testImportCreatesIndependentIdentityAndPreservesLabels() throws {
        let library = ProjectLibrary(directory:directory)
        var graph = ExampleFlows.parallelClassifier
        graph.nodes[1].name = "My branch"
        let original = try library.save(graph)
        let imported = try library.importDocument(from:library.documentURL(id:original.id))
        XCTAssertNotEqual(imported.id,original.id)
        XCTAssertEqual(try library.load(id:imported.id),graph)
    }
    func testRejectTraversalUnsupportedSchemaAndNonFiniteNumbers() throws {
        let library = ProjectLibrary(directory:directory)
        XCTAssertThrowsError(try library.documentURL(id:"../escape"))
        XCTAssertThrowsError(try library.save(ExampleFlows.parallelClassifier,id:"../escape"))
        var graph = ExampleFlows.parallelClassifier
        graph.schemaVersion = 9
        XCTAssertThrowsError(try library.save(graph))
        graph = ExampleFlows.parallelClassifier
        graph.nodes[0].x = .infinity
        XCTAssertThrowsError(try library.save(graph))
        graph = ExampleFlows.parallelClassifier
        graph.nodes[1].config["in_features"] = .number(.nan)
        XCTAssertThrowsError(try library.save(graph))
    }
    func testCorruptProjectDoesNotHideHealthyProjects() throws {
        let library = ProjectLibrary(directory:directory)
        let healthy = try library.save(ExampleFlows.parallelClassifier)
        let badID = UUID().uuidString
        try Data("{broken".utf8).write(to:library.documentURL(id:badID))
        XCTAssertThrowsError(try library.load(id:badID))
        XCTAssertEqual(try library.list().map(\.id),[healthy.id])
        var unsupported = ExampleFlows.parallelClassifier; unsupported.schemaVersion = 99
        try JSONEncoder().encode(unsupported).write(to:library.documentURL(id:badID))
        XCTAssertThrowsError(try library.load(id:badID))
    }
    func testResourceBoundsAndSymlinkProtection() throws {
        let library = ProjectLibrary(directory:directory)
        var graph = ExampleFlows.parallelClassifier
        graph.nodes += (0..<129).map { GraphNode(id:"extra-\($0)") }
        XCTAssertThrowsError(try library.save(graph))
        _ = try library.save(ExampleFlows.parallelClassifier)
        let outside = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:outside) }
        try JSONEncoder().encode(ExampleFlows.parallelClassifier).write(to:outside)
        let id = UUID().uuidString
        try FileManager.default.createSymbolicLink(at:library.documentURL(id:id),withDestinationURL:outside)
        XCTAssertThrowsError(try library.load(id:id))
        XCTAssertThrowsError(try library.save(ExampleFlows.parallelClassifier,id:id))
    }
}


extension ProjectLibraryTests {
    func testExecutionSettingsSurviveProjectReopen() throws {
        let library = ProjectLibrary(directory:directory)
        var graph = ExampleFlows.parallelClassifier
        graph.execution = ExecutionSettings(device:"cpu",epochs:250,datasetCSV:"/tmp/measures.csv",featureColumns:["length","width"],targetColumn:"class")
        let record = try library.save(graph)
        XCTAssertEqual(try library.load(id:record.id).execution,graph.execution)
    }
    func testLegacyDocumentWithoutExecutionDecodes() throws {
        let graph = ExampleFlows.parallelClassifier
        let data = try JSONEncoder().encode(graph)
        XCTAssertNil(try JSONDecoder().decode(GraphDocument.self,from:data).execution)
    }
}
