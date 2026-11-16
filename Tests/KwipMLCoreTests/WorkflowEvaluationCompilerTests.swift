import XCTest
@testable import KwipMLCore

final class WorkflowEvaluationCompilerTests:XCTestCase {
    private let frozen=WorkflowNodeAddress(domain:.control,localID:"frozen")
    private let evaluate=WorkflowNodeAddress(domain:.control,localID:"evaluate")
    private func address(_ id:String)->WorkflowNodeAddress {.init(domain:.data,localID:id)}
    private func fixture()throws->WorkflowDocument {
        var document=try WorkflowExamples.preparedGroupSplitClassifier()
        document.nodes.removeAll {$0.address.domain != .data}
        let retained=Set(document.nodes.map(\.address))
        document.edges.removeAll {!retained.contains($0.from.address) || !retained.contains($0.to.address)}
        document.nodes += [.init(address:frozen,payload:.frozenModel(config:[:])),.init(address:evaluate,payload:.evaluateTest(config:["device":.string("cpu"),"batchSize":.integer(7),"previewLimit":.integer(3)]))]
        document.edges += [.init(from:.init(address:frozen,port:"model"),to:.init(address:evaluate,port:"model"),kind:.frozenModel),.init(from:.init(address:address("split"),port:"test"),to:.init(address:evaluate,port:"test"),kind:.table)]
        return document
    }
    private func compile(_ document:WorkflowDocument)throws->WorkflowEvaluationPlan {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowEvaluationCompiler.compile(document,evaluate:evaluate,modelRegistry:.builtIn(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
    }
    func testExplicitValidationMatchesItsOwnRoleAndRefusesTestCrosswire()throws {
        var document=try fixture()
        let index=try XCTUnwrap(document.nodes.firstIndex {$0.address==evaluate})
        document.nodes[index].payload = .evaluateTest(config:["settingsVersion":2,"partitionRole":.string("validation")])
        XCTAssertThrowsError(try compile(document))
        document.edges.removeAll {$0.to.address==evaluate && $0.to.port=="test"}
        document.edges.append(.init(from:.init(address:address("fit"),port:"validation"),to:.init(address:evaluate,port:"test"),kind:.table))
        let plan=try compile(document)
        XCTAssertEqual(plan.settings.partitionRole,"validation")
        XCTAssertEqual(plan.testEndpoint,.init(node:"fit",port:"validation"))
        XCTAssertEqual(plan.canonicalTestEndpoint,.init(node:"split",port:"validation"))
        document.nodes[index].payload = .evaluateTest(config:[:])
        XCTAssertThrowsError(try compile(document),"Absent role remains Test; never infer Validation from the edge")
    }
    func testSettingsPreserveLegacyBytesAndRejectUnversionedOrUnknownRoles()throws {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        XCTAssertEqual(String(decoding:try encoder.encode(WorkflowEvaluationSettings()),as:UTF8.self),"{\"batchSize\":256,\"device\":\"auto\",\"previewLimit\":8}")
        let validation=try WorkflowEvaluationSettings.decode(["settingsVersion":2,"partitionRole":.string("validation")])
        XCTAssertEqual(try JSONDecoder().decode(WorkflowEvaluationSettings.self,from:encoder.encode(validation)),validation)
        for bad:[String:JSONValue] in [["partitionRole":.string("validation")],["settingsVersion":2],["settingsVersion":1,"partitionRole":.string("validation")],["settingsVersion":2,"partitionRole":.string("training")],["settingsVersion":2,"partitionRole":.string("test")],["settingsVersion":true,"partitionRole":.string("validation")]] {
            XCTAssertThrowsError(try WorkflowEvaluationSettings.decode(bad))
        }
    }
    func testIndependentEvaluationNeedsNoTrainArchitectureOrSourceFiles()throws {
        let document=try fixture(),plan=try compile(document)
        XCTAssertEqual(plan.projectID,document.projectID);XCTAssertEqual(plan.revision,document.revision)
        XCTAssertEqual(plan.evaluate,evaluate);XCTAssertEqual(plan.frozenModel,frozen)
        XCTAssertEqual(plan.testEndpoint,.init(node:"split",port:"test"))
        XCTAssertEqual(plan.testLineage.role,"test")
        XCTAssertEqual(plan.testLineage.source,WorkflowExamples.sourceAddress)
        XCTAssertEqual(plan.testLineage.split,address("split"))
        XCTAssertEqual(plan.testLineage.states,[])
        XCTAssertEqual(plan.canonicalTestEndpoint,.init(node:"split",port:"test"))
        XCTAssertEqual(plan.membershipValidation,.exactArchivedProtectedPopulation)
        XCTAssertEqual(plan.settings.device,"cpu");XCTAssertEqual(plan.settings.batchSize,7);XCTAssertEqual(plan.settings.previewLimit,3)
    }
    func testTransformedTestDeclaresTrainingFittedChainWithoutExecutingFit()throws {
        var document=try fixture()
        let fit=try XCTUnwrap(document.nodes.first {if case .dataOperator("kwip.data.preprocess",_)=$0.payload{return true};return false})
        document.edges.removeAll {$0.to.address==evaluate && $0.to.port=="test"}
        document.edges.append(.init(from:.init(address:fit.address,port:"test"),to:.init(address:evaluate,port:"test"),kind:.table))
        let plan=try compile(document)
        XCTAssertEqual(plan.testLineage.states,[fit.address])
        XCTAssertEqual(plan.canonicalTestEndpoint,.init(node:"split",port:"test"))
        XCTAssertEqual(plan.testEndpoint,.init(node:"fit",port:"test"))
    }

    func testPopulationChangingDescendantRequiresArchivedExactMembershipAuthority()throws {
        var filter=try XCTUnwrap(DataFlowOperatorRegistry.load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json")).operatorDefinition(id:"kwip.data.filterRows")).defaultConfig
        filter["conditions"] = .array([.object(["column":.string("signal"),"comparison":.string("gt"),"value":.number(0),"caseSensitive":true])])
        for (operatorID,config) in [("kwip.data.sampleRows",["count":JSONValue.integer(5),"seed":.integer(42)]),("kwip.data.filterRows",filter)] {
            var document=try fixture();let subset=address("subset")
            document.nodes.append(.init(address:subset,payload:.dataOperator(operatorID:operatorID,config:config)))
            document.edges.removeAll {$0.to.address==evaluate && $0.to.port=="test"}
            let output=operatorID == "kwip.data.filterRows" ? "matching":"table"
            document.edges += [.init(from:.init(address:address("split"),port:"test"),to:.init(address:subset,port:"table"),kind:.table),.init(from:.init(address:subset,port:output),to:.init(address:evaluate,port:"test"),kind:.table)]
            let plan=try compile(document)
            XCTAssertEqual(plan.testEndpoint,.init(node:"subset",port:output))
            XCTAssertEqual(plan.canonicalTestEndpoint,.init(node:"split",port:"test"))
            XCTAssertEqual(plan.membershipValidation,.exactArchivedProtectedPopulation,"Compilation is declaration only; subset execution still must reject")
        }
    }
    func testWrongRolesEditableModelAndMissingFrozenModelReject()throws {
        for port in ["training","validation"] {
            var document=try fixture()
            let index=try XCTUnwrap(document.edges.firstIndex {$0.to.address==evaluate && $0.to.port=="test"})
            document.edges[index].from.port=port
            XCTAssertThrowsError(try compile(document))
        }
        var document=try fixture()
        document.nodes[try XCTUnwrap(document.nodes.firstIndex {$0.address==frozen})].payload = .modelDefinition
        XCTAssertThrowsError(try compile(document))
        document=try fixture();document.edges.removeAll {$0.to.address==evaluate && $0.to.port=="model"}
        XCTAssertThrowsError(try compile(document))
    }
    func testInvalidStateRoleAndDependencyCycleReject()throws {
        var document=try fixture()
        let fit=try XCTUnwrap(document.nodes.first {if case .dataOperator("kwip.data.preprocess",_)=$0.payload{return true};return false})
        document.edges.removeAll {($0.to.address==evaluate && $0.to.port=="test") || ($0.to.address==fit.address && $0.to.port=="training")}
        document.edges += [.init(from:.init(address:address("split"),port:"test"),to:.init(address:fit.address,port:"training"),kind:.table),.init(from:.init(address:fit.address,port:"test"),to:.init(address:evaluate,port:"test"),kind:.table)]
        XCTAssertThrowsError(try compile(document)) {error in
            XCTAssertTrue(error.localizedDescription.contains("Training"),error.localizedDescription)
        }
        document.edges.removeAll {$0.to.address==fit.address && $0.to.port=="training"}
        document.edges.append(.init(from:.init(address:fit.address,port:"test"),to:.init(address:fit.address,port:"training"),kind:.table))
        XCTAssertThrowsError(try compile(document))
    }

    private func compileTrain(_ document:WorkflowDocument)throws->WorkflowExecutionPlan {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:13)
    }
    func testSelectedTrainIgnoresDisconnectedIncompleteEvaluationRegion()throws {
        var document=try WorkflowExamples.preparedGroupSplitClassifier(device:"cpu")
        let expected=try compileTrain(document)
        document.nodes += [.init(address:frozen,payload:.frozenModel(config:[:])),.init(address:evaluate,payload:.evaluateTest(config:[:]))]
        // The evaluation region deliberately has neither required input connected.
        // Selection scopes dependency validity; it must not disable existing Train.
        XCTAssertEqual(try compileTrain(document),expected)
        XCTAssertThrowsError(try compile(document))
    }
    func testThreeWayJoinAndAggregateRetainTrainingSourcesAndEvaluationOrigins()throws {
        for joined in [true,false] {
            var document:WorkflowDocument
            if joined {document=try WorkflowExamples.preparedJoinedClassifier(device:"cpu")}
            else {document=try WorkflowExamples.preparedAggregatedClassifier(device:"cpu")}
            let split=try XCTUnwrap(document.nodes.firstIndex {$0.address==address("split")})
            document.nodes[split].payload = .dataOperator(operatorID:"kwip.data.partition",config:["strategy":.string("entity"),
                "groupBy":.array([.string(joined ? "signal_observation_id":"subject_id")]),
                "seed":.integer(42),"validationPercent":.integer(20),"testPercent":.integer(20)])
            // Evaluation nodes require schema 5. Capture both selected-Train plans
            // under the same document contract so this checks dependency isolation.
            document.schemaVersion=max(5,document.schemaVersion)
            let original=try compileTrain(document)
            document.nodes += [.init(address:frozen,payload:.frozenModel(config:[:])),.init(address:evaluate,payload:.evaluateTest(config:[:]))]
            document.edges += [.init(from:.init(address:frozen,port:"model"),to:.init(address:evaluate,port:"model"),kind:.frozenModel),.init(from:.init(address:address("split"),port:"test"),to:.init(address:evaluate,port:"test"),kind:.table)]
            let training=try compileTrain(document),evaluation=try compile(document)
            XCTAssertEqual(training,original,"Evaluation declarations do not change selected training compilation")
            let sources:Set<WorkflowNodeAddress>=joined ? [WorkflowExamples.sourceAddress,WorkflowExamples.joinContextAddress]:[WorkflowExamples.sourceAddress]
            XCTAssertEqual(Set(training.sourceAddresses),sources)
            XCTAssertEqual(evaluation.testLineage.source,address(joined ? "join":"aggregate"))
            XCTAssertEqual(evaluation.canonicalTestEndpoint,.init(node:"split",port:"test"))
            XCTAssertEqual(evaluation.membershipValidation,.exactArchivedProtectedPopulation)
        }
    }

}
