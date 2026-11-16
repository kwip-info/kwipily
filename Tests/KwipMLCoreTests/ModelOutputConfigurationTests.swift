import XCTest
@testable import KwipMLCore

final class ModelOutputConfigurationTests: XCTestCase {
    func testLegacyAndExactSampleAssertions() throws {
        let legacy = try ModelOutputConfiguration([:])
        XCTAssertFalse(legacy.checkExpectedOutput)
        var exact = legacy; exact.checkExpectedOutput = true; exact.expectedDType = "float32"; exact.expectedSampleRank = "exact"; exact.expectedSampleDimensions = [nil, 3]
        try exact.check(sample: [2, 3], dtype: "float32")
        XCTAssertThrowsError(try exact.check(sample: [2, 4], dtype: "float32"))
        XCTAssertThrowsError(try exact.check(sample: [3], dtype: "float32"))
        XCTAssertThrowsError(try exact.check(sample: [2, 3], dtype: "int64"))
        XCTAssertEqual(try ModelOutputConfiguration(exact.values), exact)
        exact.expectedSampleRank = "any"
        try exact.check(sample: [10, 20, 30], dtype: "float32")
    }
    func testRefusesMalformedExpectations() {
        for config: [String: JSONValue] in [["checkExpectedOutput": .integer(1)], ["expectedDType": .string("float16")], ["expectedSampleDimensions": .array([.integer(0)])], ["expectedSampleDimensions": .array([.bool(true)])], ["unknown": .null]] {
            XCTAssertThrowsError(try ModelOutputConfiguration(config))
        }
    }
}

extension ModelOutputConfigurationTests {
    func testPracticeKeepsStableOutputIdentityAndDefinitionConnection() throws {
        let document = try ModelOutputPracticeExample.document(device: "cpu")
        let output = try XCTUnwrap(document.nodes.first { $0.address == ModelOutputPracticeExample.address })
        guard case .modelPrimitive(let type, _, let config) = output.payload else { return XCTFail("Expected model boundary") }
        XCTAssertEqual(type, "output")
        XCTAssertTrue(try ModelOutputConfiguration(config).checkExpectedOutput)
        XCTAssertTrue(document.edges.contains { $0.from.address == output.address && $0.from.port == "definition" && $0.to.port == "source" })
        XCTAssertTrue(document.edges.contains { $0.to.address == output.address && $0.to.port == "in" })
    }
}

extension ModelOutputConfigurationTests {
    func testPreflightFailureRetainsModelOutputAddressThroughCompositeExpansion() throws {
        var document = try ModelOutputPracticeExample.document(device: "cpu")
        let index = try XCTUnwrap(document.nodes.firstIndex { $0.address == ModelOutputPracticeExample.address })
        var config = try ModelOutputConfiguration([:]); config.checkExpectedOutput = true
        config.expectedSampleRank = "exact"; config.expectedSampleDimensions = [3]
        document.nodes[index].payload = .modelPrimitive(type: "output", configurationVersion: 2, config: config.values)
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry = try ComponentRegistry.builtIn()
        let data = try DataFlowOperatorRegistry.load(from: root.appendingPathComponent("runner/dataflow-operators.json"))
        XCTAssertThrowsError(try WorkflowCompiler.modelSnapshot(document, definition: .init(domain: .control, localID: "model"), modelRegistry: registry, dataRegistry: data)) { error in
            guard let failure = error as? WorkflowCompilerError else { return XCTFail("Expected structured compiler failure") }
            XCTAssertEqual(failure.code, .invalidModel)
            XCTAssertEqual(failure.node, ModelOutputPracticeExample.address)
            XCTAssertTrue(failure.modelIssues.contains { $0.address == ModelOutputPracticeExample.address && $0.message.contains("expected 3") && $0.message.contains("observed 2") })
        }
    }
}
