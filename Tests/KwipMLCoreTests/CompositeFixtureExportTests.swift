import XCTest
@testable import KwipMLCore

/// Opt-in fixture exporter for the real Python parity suite, not a UI acceptance test.
final class CompositeFixtureExportTests:XCTestCase {
    func testExportReferenceForBackendParity() throws {
        guard let path=ProcessInfo.processInfo.environment["KWIPML_COMPOSITE_FIXTURE_DIR"] else {throw XCTSkip("Set KWIPML_COMPOSITE_FIXTURE_DIR to export backend parity fixtures.")}
        let directory=URL(fileURLWithPath:path)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let project=CompositeExamples.parallelClassifier
        let result=try CompositeExpander.expand(project,registry:.builtIn())
        let encoder=JSONEncoder();encoder.outputFormatting=[.prettyPrinted,.sortedKeys]
        try encoder.encode(project).write(to:directory.appendingPathComponent("project.json"),options:.atomic)
        try encoder.encode(result.graph).write(to:directory.appendingPathComponent("flat.json"),options:.atomic)
        try encoder.encode(ExampleFlows.parallelClassifier).write(to:directory.appendingPathComponent("reference.json"),options:.atomic)
        try encoder.encode(result.nodeOrigins).write(to:directory.appendingPathComponent("origins.json"),options:.atomic)
        try encoder.encode(result.consumedDefinitions).write(to:directory.appendingPathComponent("pins.json"),options:.atomic)
        XCTAssertEqual(result.graph.nodes.count,8)
    }
}
