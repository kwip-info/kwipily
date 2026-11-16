import XCTest
import KwipMLCore
@testable import KwipMLApp

final class GeneralizationPracticeExampleTests: XCTestCase {
    func testPublicCasesChangeOnlyHeldOutLabelsAndRetainReservedFutureRows() throws {
        func rows(_ control: Bool) throws -> [[String]] {
            let text = try String(contentsOf: GeneralizationPracticeExample.observationsURL(control: control), encoding: .utf8)
            return text.split(separator: "\n").map { $0.split(separator: ",").map(String.init) }
        }
        let shifted = try rows(false), healthy = try rows(true)
        XCTAssertEqual(shifted.first, ["observed_at", "signal", "context", "label"])
        XCTAssertEqual(shifted.count, 81)
        XCTAssertEqual(healthy.count, 81)
        XCTAssertEqual(Array(shifted[1...48]), Array(healthy[1...48]))
        XCTAssertEqual(Array(shifted[65...80]), Array(healthy[65...80]))
        for index in 49...64 {
            XCTAssertEqual(Array(shifted[index].prefix(3)), Array(healthy[index].prefix(3)))
            XCTAssertEqual(try XCTUnwrap(Int(shifted[index][3])) + XCTUnwrap(Int(healthy[index][3])), 1)
        }
    }
    func testChronologicalTrainingOnlyPreparationAndIdenticalRunBudget() throws {
        for control in [false, true] {
            let document = try GeneralizationPracticeExample.document(control: control)
            try WorkflowCodec.validate(document)
            XCTAssertEqual(document.name, control ? "Consistent-label comparison" : "Deliberate held-out label shift")
            func dataConfig(_ id: String) throws -> [String: JSONValue] {
                let node = try XCTUnwrap(document.nodes.first { $0.address == .init(domain: .data, localID: id) })
                guard case .dataOperator(_, let config) = node.payload else { throw NSError(domain: "test", code: 1) }
                return config
            }
            let split = try dataConfig("split")
            XCTAssertEqual(split["trainingEnd"], .string("2026-01-01T00:47:00Z"))
            XCTAssertEqual(split["validationEnd"], .string("2026-01-01T01:03:00Z"))
            XCTAssertTrue(document.edges.contains { $0.from.address == .init(domain: .data, localID: "split") && $0.from.port == "training" && $0.to.address == .init(domain: .data, localID: "fit") })
            XCTAssertEqual(split["strategy"],.string("time"))
            XCTAssertEqual(split["timeColumn"],.string("observed_at"));XCTAssertEqual(split["timeEncoding"],.string("iso8601Offset"))
            XCTAssertEqual(split["embargoMilliseconds"],0);XCTAssertEqual(split["entityPolicy"],.string("allowAcrossTime"))
            XCTAssertEqual(split["entityColumns"],.array([]))
            XCTAssertFalse(document.nodes.contains {if case .dataOperator(let id,_)=$0.payload {return RetiredDataOperatorMigration.retired.contains(id)};return false})
            let fit=try PreprocessConfiguration(values:dataConfig("fit"))
            XCTAssertEqual(fit.mode,"partitions");XCTAssertEqual(fit.columns.map(\.column),["signal","context"])
            XCTAssertTrue(fit.columns.allSatisfy {$0.missing=="error" && $0.scaling=="standard" && $0.center && $0.unitVariance})
            for role in ["training","validation","test"] {
                XCTAssertTrue(document.edges.contains {$0.from.address == .init(domain:.data,localID:"split") && $0.from.port==role && $0.to.address == .init(domain:.data,localID:"fit") && $0.to.port==role && $0.kind == .table})
            }
            XCTAssertFalse(document.edges.contains {$0.from.port=="test" && $0.to.address.domain == .control},"Reserved future observations are never selected for training or evaluation.")
            let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
            let plan=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema:document.schemaVersion))
            XCTAssertEqual(plan.training,.init(node:"fit",port:"training"));XCTAssertEqual(plan.validation,.init(node:"fit",port:"validation"))
            XCTAssertEqual(plan.sourceAddresses,[WorkflowExamples.sourceAddress]);XCTAssertEqual(plan.featureColumns,["signal","context"])
            // Pure evaluation compilation verifies the retained protected lineage;
            // these test-only declarations are never installed or executed.
            var protected=document;protected.schemaVersion=max(5,protected.schemaVersion)
            let frozen=WorkflowNodeAddress(domain:.control,localID:"lineage-frozen"),evaluate=WorkflowNodeAddress(domain:.control,localID:"lineage-evaluate")
            protected.nodes += [.init(address:frozen,payload:.frozenModel(config:[:])),.init(address:evaluate,payload:.evaluateTest(config:[:]))]
            protected.edges += [.init(from:.init(address:frozen,port:"model"),to:.init(address:evaluate,port:"model"),kind:.frozenModel),.init(from:.init(address:.init(domain:.data,localID:"fit"),port:"test"),to:.init(address:evaluate,port:"test"),kind:.table)]
            let evaluation=try WorkflowEvaluationCompiler.compile(protected,evaluate:evaluate,modelRegistry:.builtIn(),dataRegistry:registry)
            XCTAssertEqual(evaluation.testLineage.role,"test");XCTAssertEqual(evaluation.testLineage.source,WorkflowExamples.sourceAddress)
            XCTAssertEqual(evaluation.testLineage.split,.init(domain:.data,localID:"split"));XCTAssertEqual(evaluation.testLineage.states,[.init(domain:.data,localID:"fit")])
            XCTAssertEqual(evaluation.canonicalTestEndpoint,.init(node:"split",port:"test"));XCTAssertEqual(evaluation.membershipValidation,.exactArchivedProtectedPopulation)
            for node in document.nodes {
                if case .runSetup(let config) = node.payload {
                    XCTAssertEqual(config["epochs"], 20); XCTAssertEqual(config["batchSize"], 48)
                    XCTAssertEqual(config["seed"], 42); XCTAssertEqual(config["learningRate"], .number(0.03))
                    XCTAssertEqual(config["device"], .string("cpu"))
                }
                if case .trainingData(let config) = node.payload {
                    XCTAssertEqual(config["featureColumns"], .array([.string("signal"), .string("context")]))
                }
            }
        }
    }
    @MainActor func testLauncherCreatesDistinctDurableProjectsWithoutPortablePaths() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        XCTAssertTrue(workspace.loadGeneralizationPracticeExample(), workspace.statusMessage)
        let shifted = try XCTUnwrap(workspace.workflowSession)
        let firstID = shifted.owner.document.projectID
        XCTAssertTrue(workspace.statusMessage.contains("label shift"))
        XCTAssertTrue(workspace.loadGeneralizationPracticeExample(control: true), workspace.statusMessage)
        let healthy = try XCTUnwrap(workspace.workflowSession)
        XCTAssertNotEqual(firstID, healthy.owner.document.projectID)
        XCTAssertEqual(healthy.owner.document.name, "Consistent-label comparison")
        let bytes = try WorkflowCodec.encode(healthy.owner.document)
        XCTAssertFalse(String(decoding: bytes, as: UTF8.self).contains(root.path))
        let sources = root.appendingPathComponent("example-sources").appendingPathComponent(healthy.owner.document.projectID)
        let copied=sources.appendingPathComponent("generalization-consistent.csv")
        XCTAssertTrue(FileManager.default.fileExists(atPath:copied.path))
        XCTAssertEqual(try Data(contentsOf:copied),try Data(contentsOf:GeneralizationPracticeExample.observationsURL(control:true)))
        let document=healthy.owner.document,projectID=document.projectID
        XCTAssertEqual(try workspace.projectLibrary.loadSource(id:projectID),.workflow(document))
        XCTAssertEqual(healthy.bindings.path(for:WorkflowExamples.sourceAddress),copied.path)
        XCTAssertTrue(workspace.beginWorkflowProject(id:projectID),workspace.statusMessage)
        let reopened=try XCTUnwrap(workspace.workflowSession)
        XCTAssertEqual(reopened.owner.document,document);XCTAssertEqual(reopened.bindings.path(for:WorkflowExamples.sourceAddress),copied.path)
        let portable=String(decoding:try WorkflowCodec.encode(reopened.owner.document),as:UTF8.self)
        XCTAssertFalse(portable.contains(root.path));XCTAssertFalse(portable.contains("sourceBindings"))
    }
    @MainActor func testWindowRoutingOpensSavedProjectWithoutReplacingApplicationWorkspace() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        var opened: ProjectRecord?
        workspace.openProjectInWindow = { record, section in
            opened = record
            XCTAssertEqual(section, "Build")
            XCTAssertTrue(FileManager.default.fileExists(atPath: record.path))
        }
        XCTAssertTrue(workspace.loadGeneralizationPracticeExample(control: true))
        XCTAssertEqual(opened?.name, "Consistent-label comparison")
        XCTAssertNil(workspace.workflowSession, "Application-level Learn must route to a document window.")
    }

}
