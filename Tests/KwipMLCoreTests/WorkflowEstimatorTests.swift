import XCTest
@testable import KwipMLCore

final class WorkflowEstimatorTests: XCTestCase {
    func testEstimatorPreservesPreparedPartitionContractAndRoundTrips() throws {
        let model = try ComponentRegistry.builtIn(), data = try DataFlowOperatorRegistry.load(from: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))
        var document = try WorkflowCodec.migrateToVersion5(WorkflowExamples.preparedSeparableConv2dClassifier())
        let original = try WorkflowCompiler.compile(document, train: document.nodes.first { if case .train = $0.payload { return true }; return false }!.address, modelRegistry: model, dataRegistry: data, contractVersion: 4)
        let address = WorkflowNodeAddress(domain: .control, localID: "estimator")
        let settings: [String: JSONValue] = ["backendID": .string("sklearn"), "featureColumns": .array(original.featureColumns.map(JSONValue.string)), "targetColumn": .string(try XCTUnwrap(original.targetColumn)), "settings": .object(SklearnEstimatorConfiguration.defaultSettings)]
        document.nodes.append(.init(address: address, name: "My classifier", x: 100, y: 100, payload: .estimator(config: settings)))
        for (port,endpoint) in [("training",original.training),("validation",original.validation)] {
            document.edges.append(.init(from: .init(address: .init(domain: .data, localID: endpoint.node), port: endpoint.port), to: .init(address: address, port: port), kind: .table))
        }
        let plan = try WorkflowEstimatorCompiler.compile(document, estimator: address, modelRegistry: model, dataRegistry: data)
        XCTAssertEqual(plan.training, original.training)
        XCTAssertEqual(plan.validation, original.validation)
        XCTAssertNil(plan.model)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)), document)
        document.edges.removeAll { $0.to.address == address && $0.to.port == "validation" }
        XCTAssertThrowsError(try WorkflowEstimatorCompiler.compile(document, estimator: address, modelRegistry: model, dataRegistry: data))
    }
    func testBundledSklearnLessonIsVersionedAndExplicitAboutAssessment() throws {
        let lesson = try SklearnFoundationLesson.bundled()
        XCTAssertEqual(lesson.id, "sklearn.logistic-foundation")
        XCTAssertFalse(lesson.steps.isEmpty)
        XCTAssertTrue(lesson.assessment.contains("no scored mastery"))
    }
    func testStrictEstimatorSettingsRejectBooleanNumberAndUnknownSettings() throws {
        try SklearnEstimatorConfiguration.validate(SklearnEstimatorConfiguration.defaultSettings)
        var settings = SklearnEstimatorConfiguration.defaultSettings
        settings["maxIterations"] = .bool(true)
        XCTAssertThrowsError(try SklearnEstimatorConfiguration.validate(settings))
        settings = SklearnEstimatorConfiguration.defaultSettings
        settings["seed"] = .integer(42)
        XCTAssertThrowsError(try SklearnEstimatorConfiguration.validate(settings))
    }
}
