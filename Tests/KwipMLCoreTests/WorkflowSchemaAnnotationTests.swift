import XCTest
@testable import KwipMLCore

final class WorkflowSchemaAnnotationTests: XCTestCase {
    func testFormulaDiscoveryAnnotationsDoNotPreventExecutionValidation() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry = try DataFlowOperatorRegistry.load(from: root.appendingPathComponent("runner/dataflow-operators.json"))
        let definition = try XCTUnwrap(registry.operatorDefinition(id: "kwip.data.formula"))
        let valid: JSONValue = .object(["operations": .array([.object(["id": .string("one"), "expression": .string("1"), "outputs": .array([.object(["name": .string("one"), "dtype": .string("int64"), "write": .string("new")])]), "nullPolicy": .string("error")])])])
        XCTAssertNoThrow(try WorkflowSchema.check(valid, schema: definition.configSchema))
        XCTAssertThrowsError(try WorkflowSchema.check(.object([:]), schema: definition.configSchema))
        XCTAssertThrowsError(try WorkflowSchema.check(.string("wrong"), schema: definition.configSchema))
    }

    func testUnknownAssertionsAndMalformedAnnotationsRemainRejected() throws {
        for (key, value): (String, JSONValue) in [("pattern", .string("^x$")), ("x-arbitrary", .object([:])), ("description", .integer(1)), ("examples", .string("wrong")), ("x-functionSignatures", .array([])), ("x-regexOptions", .string("wrong")), ("x-functionSignatures", .object(["abs": .integer(1)])), ("x-regexOptions", .object(["flags": .bool(true)])), ("x-regexOptions", .object(["patternCharactersMaximum": .integer(-1)])), ("x-regexOptions", .object(["patternCharactersMaximum": .string("512")])), ("x-regexOptions", .object(["unknown": .integer(1)]))] {
            XCTAssertThrowsError(try WorkflowSchema.check(.string("x"), schema: .object(["type": .string("string"), key: value])), key)
        }
    }

    func testFormulaPartitionPrebuiltCompilerPreservesChronologicalRoles() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry = try DataFlowOperatorRegistry.load(from: root.appendingPathComponent("runner/dataflow-operators.json"))
        let source = WorkflowNodeAddress(domain: .data, localID: "source")
        let formula = WorkflowNodeAddress(domain: .data, localID: "calendar")
        let split = WorkflowNodeAddress(domain: .data, localID: "split")
        let estimator = WorkflowNodeAddress(domain: .control, localID: "fit")
        let formulaConfig: [String: JSONValue] = ["operations": .array([.object([
            "id": .string("calendar-order"), "expression": .string("[original_epoch_milliseconds]"),
            "outputs": .array([.object(["name": .string("calendar_order"), "dtype": .string("int64"), "write": .string("new")])]),
            "nullPolicy": .string("error")])])]
        let splitConfig: [String: JSONValue] = ["strategy": .string("time"), "timeColumn": .string("calendar_order"),
            "timeEncoding": .string("epochMilliseconds"), "trainingEnd": .integer(100), "validationEnd": .integer(200),
            "embargoMilliseconds": .integer(0), "entityColumns": .array([]), "entityPolicy": .string("allowAcrossTime")]
        let fitConfig: [String: JSONValue] = ["task": .string("prebuilt"), "backendID": .string("sklearn"),
            "algorithm": .string("randomForestRegressor"), "operation": .string("fit"),
            "featureColumns": .array([.string("observed_value")]), "targetColumn": .string("target"),
            "settings": .object(["trees": .integer(100), "maxDepth": .integer(8), "minSamplesLeaf": .integer(1), "seed": .integer(42)])]
        func edge(_ from: WorkflowNodeAddress, _ output: String, _ to: WorkflowNodeAddress, _ input: String) -> WorkflowEdge {
            .init(from: .init(address: from, port: output), to: .init(address: to, port: input), kind: .table)
        }
        var document = WorkflowDocument(projectID: "6F984148-CB8B-4DD4-9BB4-53FC9FB32743", nodes: [
            .init(address: source, payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: try XCTUnwrap(registry.operatorDefinition(id: "kwip.data.sourceCSV")).defaultConfig)),
            .init(address: formula, payload: .dataOperator(operatorID: "kwip.data.formula", config: formulaConfig)),
            .init(address: split, payload: .dataOperator(operatorID: "kwip.data.partition", config: splitConfig)),
            .init(address: estimator, payload: .estimator(config: fitConfig))
        ], edges: [edge(source, "table", formula, "table"), edge(formula, "table", split, "table"),
                   edge(split, "training", estimator, "training"), edge(split, "validation", estimator, "validation")])
        document.schemaVersion = 17
        let plan = try WorkflowEstimatorCompiler.compile(document, estimator: estimator, modelRegistry: .builtIn(), dataRegistry: registry)
        XCTAssertEqual(plan.training, .init(node: "split", port: "training"))
        XCTAssertEqual(plan.validation, .init(node: "split", port: "validation"))
        XCTAssertEqual(plan.sourceAddresses, [source])
        XCTAssertEqual(plan.dataflow.nodes.count, 3)
        XCTAssertEqual(plan.configuration.settings["trees"], .integer(100))
    }

    func testPrivateCase07ActualPlanWhenExplicitFixtureProvided() throws {
        guard let path = ProcessInfo.processInfo.environment["KWIPML_CASE07_DIAGNOSTIC_PROJECT"] else { throw XCTSkip("Private case07 document is an explicitly supplied diagnostic fixture.") }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let document = try WorkflowCodec.decode(Data(contentsOf: URL(fileURLWithPath: path)))
        XCTAssertEqual(document.revision, 3)
        let plan = try WorkflowEstimatorCompiler.compile(document, estimator: .init(domain: .control, localID: "fit"), modelRegistry: .builtIn(), dataRegistry: .load(from: root.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertEqual(plan.training, .init(node: "split", port: "training"))
        XCTAssertEqual(plan.validation, .init(node: "split", port: "validation"))
        XCTAssertEqual(plan.configuration.backendID, "sklearn")
    }
}
