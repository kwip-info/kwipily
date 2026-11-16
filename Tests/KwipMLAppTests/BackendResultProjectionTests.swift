import XCTest
import KwipMLCore
@testable import KwipMLApp

final class BackendResultProjectionTests:XCTestCase {
    private func projection(operation:String="fit",phase:WorkflowExecutionPhase = .completed,accepted:Bool=true,construction:Bool=true,bindings:Bool=true)->BackendResultProjection {
        .init(runID:UUID(),capturedRevision:3,phase:phase,operation:operation,backendID:"sklearn",currentConstruction:construction,currentBindings:bindings,accepted:accepted,tablePorts:["predictions","profiles","/private/path"],manifest:[
            "evaluation":.object(["count":649,"mae":12.5,"rmse":15.1,"classOrder":.array([.string("private-label")]),"privatePath":.string("/secret"),"infinity":.number(.infinity)]),
            "artifacts":.array([.object(["path":.string("/private/path")])]),"provenance":.object(["source":.string("/private/source")]),"configuration":.object(["featureColumns":.array([.string("private-feature")])])])
    }
    func testMetadataNeverExposesValuesAndConsentRecheckedPerResponse() throws {
        let p=projection()
        let metadata=p.response(includeValues:false)
        XCTAssertEqual(metadata["available"],true)
        XCTAssertNil(metadata["evaluation"])
        let encoded=String(decoding:try JSONEncoder().encode(metadata),as:UTF8.self)
        for secret in ["649","12.5","private","featureColumns","artifacts","provenance"] {XCTAssertFalse(encoded.contains(secret),secret)}
        let values=p.response(includeValues:true)
        XCTAssertEqual(values["evaluation"],.object(["count":649,"mae":12.5,"rmse":15.1]))
        XCTAssertNil(p.response(includeValues:false)["evaluation"],"Revoking values cannot reuse an enriched cached dictionary")
    }
    func testOperationAndFreshnessAreNotTrainingSuccessClaims() {
        for operation in ["fit","predict","evaluate"] {
            let stale=projection(operation:operation,construction:false,bindings:false).response(includeValues:true)
            XCTAssertEqual(stale["operation"],.string(operation))
            XCTAssertEqual(stale["currentConstruction"],false);XCTAssertEqual(stale["currentBindings"],false)
            XCTAssertNil(stale["model"]);XCTAssertNil(stale["exports"])
        }
        // DBSCAN fit has profiles but no predictive model. Metadata never implies model creation.
        let dbscan=projection().response(includeValues:false)
        XCTAssertEqual(dbscan["ports"],.array([.string("predictions"),.string("profiles")]))
        XCTAssertNil(dbscan["model"])
    }
    func testNonfiniteRecognizedMetricsAndUnknownIdentifiersFailClosed() {
        XCTAssertEqual(BackendResultProjection.numericMetrics(["evaluation":.object([
            "mae":.number(.nan),"rmse":.number(.infinity),"rSquared":.number(-.infinity),"count":7])]),["count":7])
        let unknown=BackendResultProjection(runID:UUID(),capturedRevision:1,phase:.completed,
            operation:"future-operation",backendID:"/private/backend",currentConstruction:true,currentBindings:true,
            accepted:true,tablePorts:[],manifest:[:]).response(includeValues:false)
        XCTAssertEqual(unknown["operation"],.string("unknown"))
        XCTAssertEqual(unknown["backendID"],.string("unknown"))
    }

    func testIncompleteAndUnacceptedSessionsNeverExposeManifestValues() {
        for phase in [WorkflowExecutionPhase.idle,.preparing,.training,.failed,.cancelled] {
            let result=projection(phase:phase).response(includeValues:true)
            XCTAssertEqual(result["available"],false);XCTAssertNil(result["evaluation"])
            XCTAssertEqual(result["ports"],.array([]))
        }
        XCTAssertNil(projection(accepted:false).response(includeValues:true)["evaluation"])
    }
}

extension MCPWorkflowGatewayTests {
    func testBackendMetadataUsesCurrentAuthorityAndValuesPermission() async throws {
        let model=try ComponentRegistry.builtIn()
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let document=try WorkflowExamples.preparedSklearnClassifier()
        let f=try fixture(document:document)
        let host=try XCTUnwrap(f.windows.openControllers.first?.session.workspace.workflowSession)
        let workspace=try XCTUnwrap(f.windows.openControllers.first?.session.workspace)
        let runtimeRoot=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:runtimeRoot)}
        for path in [".venv/bin/python","runner/main.py","Sources/KwipMLCore/Resources/components.json"] {
            let file=runtimeRoot.appendingPathComponent(path)
            try FileManager.default.createDirectory(at:file.deletingLastPathComponent(),withIntermediateDirectories:true)
            try Data().write(to:file)
        }
        try FileManager.default.setAttributes([.posixPermissions:0o700],ofItemAtPath:runtimeRoot.appendingPathComponent(".venv/bin/python").path)
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:runtimeRoot,allowDevelopment:true)
        let plan=try WorkflowEstimatorCompiler.compile(host.owner.document,estimator:WorkflowExamples.sklearnEstimatorAddress,modelRegistry:model,dataRegistry:registry)
        let policy=try workspace.executionPolicy(for:host.owner.projectID)
        func session()->BackendFitSession {.init(workspace:workspace,owner:host.owner,bindings:host.bindings,history:host.backendHistory,plan:plan,runtime:runtime,policy:policy,launcher:{_,_,_,_,_ in XCTFail("No worker allowed");throw CancellationError()})}
        let first=session();host.observeBackendFit(first)
        let address:JSONValue = .object(["domain":.string("control"),"localID":.string(plan.estimator.localID)])
        let metadata=try await call(f,.resultsRead,["address":address])
        XCTAssertEqual(metadata["resultKind"],.string("backend"));XCTAssertEqual(metadata["phase"],.string("idle"))
        XCTAssertNil(metadata["evaluation"])
        XCTAssertNil(host.backendResultSummary(for:.init(domain:.control,localID:"forged"),includeValues:true))
        let second=session();host.observeBackendFit(second)
        XCTAssertEqual(host.backendResultSummary(for:plan.estimator,includeValues:false)?["runID"],.string(second.runID.uuidString))
        var settings=f.controller.settings;settings.projects[f.owner.projectID]?.readCapturedValues=false;try f.controller.save(settings)
        do {_=try await call(f,.resultsRead,["address":address,"includeValues":true]);XCTFail("Revoked values access accepted")} catch {}
        let stillMetadata=try await call(f,.resultsRead,["address":address]);XCTAssertEqual(stillMetadata["resultKind"],.string("backend"))
        host.retire();XCTAssertNil(host.backendResultSummary(for:plan.estimator,includeValues:false))
    }
}
