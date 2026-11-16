import XCTest
import KwipMLCore
@testable import KwipMLApp

final class GeluLessonStoreTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func graph(mode:Bool)->GraphDocument {
        GraphDocument(nodes:[
            GraphNode(id:"input",type:"input",config:["features":2]),
            GraphNode(id:"linear",type:"linear",config:["in_features":2,"out_features":2,"bias":false]),
            GraphNode(id:"gelu",type:"gelu",config:["approximate_tanh":.bool(mode)]),GraphNode(id:"output",type:"output")
        ],edges:[GraphEdge(source:"input",target:"linear"),GraphEdge(source:"linear",target:"gelu"),GraphEdge(source:"gelu",target:"output")])
    }
    @MainActor private func construct(_ lesson:AffineLearningStore,_ workspace:WorkspaceStore,mode:Bool) {
        workspace.edit {$0=graph(mode:mode)}
        workspace.selectNode("linear");lesson.resetParameters(nodeID:"linear")
        for (index,value) in [1.0,0,0,1].enumerated() {
            XCTAssertTrue(lesson.setWeight(nodeID:"linear",row:index/2,column:index%2,value:value))
        }
    }
    @MainActor func testLessonRequiresVerifiedSpecificRuntimeSupportAndReinstallClearsReadiness()throws {
        guard FileManager.default.fileExists(atPath:repository.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development runtime unavailable")}
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:directory)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:directory,detectCapabilities:false)
        let lesson=AffineLearningStore(workspace:workspace)
        for (identity,mode) in [(ModelConstructionLessonIdentity.geluExact,false),(.geluTanh,true)] {
            lesson.start(identity:identity,mode:.guided);construct(lesson,workspace,mode:mode)
            try installFixtureExecutionContract(workspace,version:"2.14.1")
            XCTAssertTrue(workspace.validation.isValid)
            XCTAssertFalse(lesson.canSubmit,"A framework runtime alone cannot enable the GELU lesson.")
            let count=lesson.attempts.count
            lesson.submit();XCTAssertFalse(workspace.isAssessingLearning);XCTAssertEqual(lesson.attempts.count,count)
            XCTAssertTrue(lesson.statusMessage.localizedCaseInsensitiveContains("GELU"),lesson.statusMessage)
            let descriptor=Data(#"{"version":{"major":1,"minor":0,"patch":0},"adapterID":"kwip.runner.graph-v2.gelu","graphSchemaVersions":[2],"approximateTanhModes":[false,true]}"#.utf8)
            let base=try BuiltInCapabilityBridge.executionContract(registry:workspace.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu])
            try workspace.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:JSONEncoder().encode(base),geluDescriptor:descriptor)
            XCTAssertTrue(lesson.canSubmit,lesson.parameterIssues.joined(separator:"; "))
            try installFixtureExecutionContract(workspace,version:"2.14.1")
            XCTAssertFalse(lesson.canSubmit,"New base-only detection must clear optional GELU readiness.")
        }
        lesson.start(identity:.composition,mode:.guided)
        var legacy=graph(mode:false);legacy.nodes.removeAll{$0.id=="gelu"}
        legacy.edges=[GraphEdge(source:"input",target:"linear"),GraphEdge(source:"linear",target:"output")]
        workspace.edit {$0=legacy};workspace.selectNode("linear");lesson.resetParameters(nodeID:"linear")
        XCTAssertTrue(lesson.canSubmit,"An older verified worker must still support the existing affine lesson.")
    }
    @MainActor func testActualBothModesDistinguishConstructionErrorsNumericFailureAndPrivateReceipts()async throws {
        guard FileManager.default.fileExists(atPath:repository.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development worker unavailable")}
        for (identity,mode,objective) in [(ModelConstructionLessonIdentity.geluExact,false,"activations.gelu-exact"),(.geluTanh,true,"activations.gelu-tanh")] {
            let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer{try? FileManager.default.removeItem(at:directory)}
            let workspace=WorkspaceStore(rootOverride:repository,storageOverride:directory,detectCapabilities:false)
            await verifyRuntimeForIntegration(workspace)
            XCTAssertNotNil(workspace.geluRuntimeCapability)
            let lesson=AffineLearningStore(workspace:workspace)
            lesson.start(identity:identity,mode:.guided)
            XCTAssertEqual(lesson.lessonTask?.identity,identity)
            XCTAssertEqual(Set(lesson.lessonTask?.allowedNodeTypes ?? []),["input","linear","gelu","add","concat","output"])
            XCTAssertEqual(workspace.graph.nodes.map(\.type),["input","output"])
            XCTAssertTrue(workspace.graph.edges.isEmpty,"The task must not load a solved GELU flow.")
            construct(lesson,workspace,mode:mode)
            lesson.setNotes("Identity weights isolate the activation and its selected approximation mode.")
            lesson.submit();try await finish(workspace)
            XCTAssertEqual(lesson.latestGrade?.status,.passed,lesson.statusMessage)
            XCTAssertEqual(lesson.formativeProjection?.fixtures.count,4)
            XCTAssertFalse(lesson.hasCurrentCompletion)
            let validAttempt=try XCTUnwrap(lesson.latestAttempt)

            workspace.edit { graph in graph.nodes[graph.nodes.firstIndex{$0.id=="gelu"}!].config["approximate_tanh"] = .bool(!mode) }
            XCTAssertFalse(lesson.canSubmit);XCTAssertFalse(lesson.parameterIssues.isEmpty)
            XCTAssertFalse(lesson.latestResultIsCurrent);XCTAssertNil(lesson.formativeProjection)
            let count=lesson.attempts.count
            lesson.submit();XCTAssertFalse(workspace.isAssessingLearning)
            XCTAssertEqual(lesson.attempts.count,count);XCTAssertEqual(lesson.latestAttempt?.id,validAttempt.id,"Construction rejection cannot masquerade as a new numerical grade.")
            workspace.edit { graph in graph.nodes[graph.nodes.firstIndex{$0.id=="gelu"}!].config["approximate_tanh"] = .bool(mode) }
            workspace.selectNode("linear")
            XCTAssertTrue(lesson.setWeight(nodeID:"linear",row:0,column:0,value:0))
            XCTAssertTrue(lesson.canSubmit,lesson.parameterIssues.joined(separator:"; "))
            lesson.submit();try await finish(workspace)
            XCTAssertEqual(lesson.latestGrade?.status,.failed,lesson.statusMessage)
            XCTAssertEqual(lesson.latestAttempt?.objectiveID,objective)

            XCTAssertTrue(lesson.setWeight(nodeID:"linear",row:0,column:0,value:1))
            lesson.setMode(.assessment);lesson.submit();try await finish(workspace)
            XCTAssertEqual(lesson.latestGrade?.status,.passed,lesson.statusMessage)
            XCTAssertTrue(lesson.hasCurrentCompletion);XCTAssertTrue(lesson.resultSaved)
            let assessed=try XCTUnwrap(lesson.latestAttempt)
            let receipt=try JSONDecoder().decode(AffineLearningReceipt.self,from:Data(contentsOf:directory.appendingPathComponent("learning/attempts/\(assessed.id).json")))
            XCTAssertEqual(receipt.lessonIdentity,identity);XCTAssertEqual(receipt.submission.schemaVersion,2)
            XCTAssertEqual(receipt.runtime.capabilities["gelu"],ContractVersion())
            XCTAssertEqual(receipt.runtime.capabilities["linear"],ContractVersion(1,1,0))
            lesson.setMode(.practice);XCTAssertNil(lesson.formativeProjection)
            XCTAssertTrue(lesson.save())
            let reopened=AffineLearningStore(workspace:workspace);reopened.reopenCurrentProject()
            XCTAssertEqual(reopened.lessonTask?.identity,identity)
            XCTAssertEqual(reopened.latestAttempt?.id,assessed.id);XCTAssertTrue(reopened.latestResultIsCurrent)
            XCTAssertTrue(reopened.hasCurrentCompletion);XCTAssertNil(reopened.formativeProjection,"Assessment fixture privacy survives reopening and mode change.")
            reopened.submit();try await finish(workspace)
            XCTAssertEqual(reopened.latestGrade?.status,.passed,reopened.statusMessage)
            XCTAssertEqual(reopened.formativeProjection?.fixtures.count,4)
            reopened.start(identity:mode ? .geluExact:.geluTanh,mode:.guided)
            XCTAssertFalse(reopened.hasCurrentCompletion,"Exact and tanh lessons have separate progress identities.")
        }
    }
    @MainActor private func finish(_ workspace:WorkspaceStore)async throws {
        let deadline=Date().addingTimeInterval(25)
        while workspace.isAssessingLearning && Date()<deadline {try await Task.sleep(nanoseconds:25_000_000)}
        XCTAssertFalse(workspace.isAssessingLearning)
    }
}
