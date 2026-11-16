import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExecutionBoundaryTests: XCTestCase {
    private func address(_ domain:WorkflowDomain,_ id:String)->WorkflowNodeAddress {.init(domain:domain,localID:id)}
    private func edge(_ a:WorkflowDomain,_ id:String,_ port:String,_ b:WorkflowDomain,_ target:String,_ input:String,_ kind:WorkflowEdgeKind)->WorkflowEdge {
        .init(id:"\(a)-\(id)-\(port)-\(b)-\(target)-\(input)",from:.init(address:address(a,id),port:port),to:.init(address:address(b,target),port:input),kind:kind)
    }
    private func fixture(block:Bool=false)->WorkflowDocument {CurrentExecutionWorkflowTestFixture.make(block:block)}

    private func copiedRuntime(from original:URL,into root:URL) throws -> RuntimeLocation {
        let manager = FileManager.default
        try manager.createDirectory(at:root,withIntermediateDirectories:true)
        try manager.createSymbolicLink(at:root.appendingPathComponent(".venv"),withDestinationURL:original.appendingPathComponent(".venv"))
        for name in ["runner","scripts","Sources/KwipMLCore/Resources"] {
            try manager.createDirectory(at:root.appendingPathComponent(name),withIntermediateDirectories:true)
        }
        let source = original.appendingPathComponent("runner")
        let listing = try XCTUnwrap(manager.enumerator(at:source,includingPropertiesForKeys:[.isDirectoryKey,.isSymbolicLinkKey]))
        while let url = listing.nextObject() as? URL {
            let name = url.lastPathComponent
            if ["tests","__pycache__",".pytest_cache"].contains(name) {listing.skipDescendants();continue}
            if name.hasSuffix(".pyc") {continue}
            let info = try url.resourceValues(forKeys:[.isDirectoryKey,.isSymbolicLinkKey])
            XCTAssertNotEqual(info.isSymbolicLink,true,"Fixture source must be regular copied worker content")
            let relative = String(url.path.dropFirst(source.path.count+1))
            let destination = root.appendingPathComponent("runner").appendingPathComponent(relative)
            if info.isDirectory == true {try manager.createDirectory(at:destination,withIntermediateDirectories:true)}
            else {try manager.copyItem(at:url,to:destination)}
        }
        try manager.copyItem(at:original.appendingPathComponent("scripts/runtime_launch.py"),to:root.appendingPathComponent("scripts/runtime_launch.py"))
        try manager.copyItem(at:original.appendingPathComponent("Sources/KwipMLCore/Resources/components.json"),to:root.appendingPathComponent("Sources/KwipMLCore/Resources/components.json"))
        return try RuntimeLocation.resolve(resources:nil,developmentRoot:root,allowDevelopment:true)
    }
    @MainActor private func checkBoundary(cancel:Bool) async throws {
        guard let path = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Set KWIPML_TEST_DEVELOPMENT_ROOT for actual preparation phase-boundary checks.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let runtime = try copiedRuntime(from:URL(fileURLWithPath:path),into:root.appendingPathComponent("isolated-runtime"))
        let library = ProjectLibrary(directory:root.appendingPathComponent("projects")), document = fixture(block:true)
        _ = try library.saveSource(.workflow(document),id:document.projectID)
        let model = try JSONDecoder().decode(ComponentRegistry.self,from:Data(contentsOf:runtime.registryURL))
        let data = try DataFlowOperatorRegistry.load(from:runtime.workingDirectoryURL.appendingPathComponent("runner/dataflow-operators.json"))
        let owner = try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:model,dataRegistry:data)
        let request = try WorkflowExecutionRequest.capture(from:owner,train:address(.control,"train"),runtime:runtime)
        let source = root.appendingPathComponent("source.csv")
        let rows = (0..<64).map {"\($0),\(($0 * 7) % 13),\($0 % 2)"}
        try Data((["a,b,label"]+rows).joined(separator:"\n").utf8).write(to:source)
        let reached = expectation(description:"Actual preparation exits before phase transition")
        var forward:(()->Void)?, launches = 0, releases = 0, outcomes:[WorkflowExecutionOutcome] = []
        var receivedCompletion = false
        let launcher:WorkflowWorkerLauncher = { executable,arguments,directory,line,exit in
            launches += 1
            XCTAssertEqual(launches,1,"Training must never launch in this boundary test")
            XCTAssertTrue(arguments.contains("--execute-dataflow"))
            return try WorkflowExecutionSession.launchWorker(executable,arguments,directory,{ event in
                if event.contains("dataflowCompleted") {receivedCompletion = true}
                line(event)
            },{code,stderr in
                XCTAssertEqual(code,0,stderr)
                forward = {exit(code,stderr)}
                reached.fulfill()
            })
        }
        let run = WorkflowExecutionSession(request:request,owner:owner,
            sources:[address(.data,"input"):try WorkflowCapturedSource.capture(source)],directory:root.appendingPathComponent("10000000-0000-0000-0000-000000000001"),
            reservation:.init(acquire:{UUID()},release:{_ in releases += 1}),launch:launcher,
            onTerminal:{outcomes.append($0)})
        XCTAssertTrue(run.start())
        await fulfillment(of:[reached],timeout:45)
        guard let forward else {run.cancel();return XCTFail("Preparation never reached its captured exit boundary")}
        XCTAssertTrue(receivedCompletion)
        XCTAssertTrue(FileManager.default.fileExists(atPath:root.appendingPathComponent("10000000-0000-0000-0000-000000000001/preparation/artifacts/manifest.json").path))
        XCTAssertEqual(run.phase,.preparing);XCTAssertTrue(owner.isBusy)
        XCTAssertEqual(releases,0);XCTAssertTrue(outcomes.isEmpty)
        if cancel {
            run.cancel()
            XCTAssertTrue(owner.isBusy);XCTAssertEqual(releases,0);XCTAssertTrue(outcomes.isEmpty)
        } else {
            let engine = runtime.workingDirectoryURL.appendingPathComponent("runner/engine.py")
            var bytes = try Data(contentsOf:engine);bytes.append(Data("\n# Changed only in isolated phase-boundary test fixture.\n".utf8))
            try bytes.write(to:engine)
            XCTAssertFalse(try request.runtimeEvidence.matchesCurrent(runtime))
        }
        forward()
        XCTAssertEqual(run.phase,cancel ? .cancelled : .failed)
        XCTAssertEqual(launches,1);XCTAssertEqual(releases,1);XCTAssertFalse(owner.isBusy)
        XCTAssertEqual(outcomes.count,1)
        let outcome = try XCTUnwrap(outcomes.first)
        XCTAssertNil(outcome.modelManifestSHA256);XCTAssertEqual(outcome.request.document,document)
        if !cancel {XCTAssertTrue(outcome.error?.contains("Worker source changed") == true)}
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("10000000-0000-0000-0000-000000000001/training").path))
        forward();XCTAssertEqual(outcomes.count,1);XCTAssertEqual(releases,1)
    }
    @MainActor func testActualPreparationCancellationBeforeForwardedExitCannotLaunchTraining() async throws {
        try await checkBoundary(cancel:true)
    }
    @MainActor func testActualPreparationRejectsChangedIsolatedRuntimeBeforeTraining() async throws {
        try await checkBoundary(cancel:false)
    }
}
