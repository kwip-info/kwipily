import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelRequestProvenanceTests:XCTestCase {
    @MainActor private func setup(block:Bool=true)throws->(URL,WorkspaceStore,ModelAuthoringStore) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let id=UUID().uuidString
        if block {
            var source=CompositeExamples.parallelClassifier
            source.execution=ExecutionSettings(compositeSource:CompositeSourceReference(path:"/unused/legacy/source.json",sha256:String(repeating:"0",count:64),expandedSemanticSHA256:String(repeating:"0",count:64)))
            _ = try workspace.projectLibrary.saveDocument(.blockModel(ModelAuthoringDocument(projectID:id,source:source)),id:id)
        } else {_ = try workspace.projectLibrary.save(ExampleFlows.parallelClassifier,id:id)}
        let owner=try ModelAuthoringStore(library:workspace.projectLibrary,projectID:id)
        workspace.activeProjectID=id;workspace.modelAuthoring=owner
        return (root,workspace,owner)
    }
    @MainActor func testCapturedParentStagesExactProvenanceAndActualWorkerArchivesIt()throws {
        let (root,workspace,owner)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let request=try ModelExecutionRequest.capture(from:owner,registry:workspace.registry)
        let directory=root.appendingPathComponent("request");try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let args=try workspace.compositeTrainingArguments(request:request,requestDirectory:directory)
        XCTAssertEqual(args.count,4);XCTAssertEqual(args[0],"--composite-source");XCTAssertEqual(args[2],"--composite-source-sha256")
        let bytes=try Data(contentsOf:URL(fileURLWithPath:args[1]))
        XCTAssertEqual(CompositeArchitectureIdentity.hash(bytes),args[3])
        let source=try JSONDecoder().decode(CompositeSourceSnapshot.self,from:bytes)
        XCTAssertEqual(source,request.snapshot.provenance);XCTAssertEqual(source.expandedGraph,request.graph)
        XCTAssertEqual(try CompositeArchitectureIdentity.digest(source.expandedGraph),request.snapshot.semanticSHA256)
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python=repository.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.fileExists(atPath:python.path) else {throw XCTSkip("Local Python runtime unavailable")}
        let graph=directory.appendingPathComponent("graph.json"),output=root.appendingPathComponent("artifacts")
        try JSONEncoder().encode(request.graph).write(to:graph)
        let process=Process(),pipe=Pipe();process.executableURL=python;process.currentDirectoryURL=repository
        process.arguments=[repository.appendingPathComponent("runner/main.py").path,"--graph",graph.path,"--output",output.path,"--device","cpu","--epochs","1"]+args
        process.standardOutput=pipe;process.standardError=pipe;try process.run()
        let messages=pipe.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus,0,String(decoding:messages,as:UTF8.self))
        XCTAssertEqual(try Data(contentsOf:output.appendingPathComponent("composite/source.json")),bytes)
        let manifest=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:output.appendingPathComponent("manifest.json"))) as? [String:Any])
        let provenance=try XCTUnwrap(manifest["provenance"] as? [String:Any])
        XCTAssertEqual((provenance["composite"] as? [String:Any])?["sha256"] as? String,args[3])
    }
    @MainActor func testStaleOwnerAndPendingDraftCannotStage()throws {
        let (root,workspace,owner)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let request=try ModelExecutionRequest.capture(from:owner,registry:workspace.registry)
        XCTAssertTrue(owner.editMetadata({name,_ in name="Changed"},lease:owner.lease()))
        XCTAssertThrowsError(try workspace.compositeTrainingArguments(request:request,requestDirectory:root))
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("composite-source.json").path))
        XCTAssertTrue(owner.beginDraft(path:["features"],lease:owner.lease()))
        XCTAssertThrowsError(try workspace.compositeTrainingArguments(requestDirectory:root))
    }
    @MainActor func testFlatOwnerPreservesLegacySidecarAndHashRejection()throws {
        let (root,workspace,owner)=try setup(block:false);defer{try? FileManager.default.removeItem(at:root)}
        XCTAssertTrue(try workspace.compositeTrainingArguments(requestDirectory:root).isEmpty)
        let source=CompositeExamples.parallelClassifier,expanded=try CompositeExpander.expand(source,registry:workspace.registry)
        let bytes=try JSONEncoder().encode(CompositeSourceSnapshot(project:source,expansion:expanded))
        let original=root.appendingPathComponent("legacy-source.json");try bytes.write(to:original)
        var flat=expanded.graph
        flat.execution=ExecutionSettings(compositeSource:CompositeSourceReference(path:original.path,sha256:CompositeArchitectureIdentity.hash(bytes),expandedSemanticSHA256:try CompositeArchitectureIdentity.digest(flat)))
        _ = try workspace.projectLibrary.save(flat,id:owner.projectID)
        let reopened=try ModelAuthoringStore(library:workspace.projectLibrary,projectID:owner.projectID);workspace.modelAuthoring=reopened
        let request=try ModelExecutionRequest.capture(from:reopened,registry:workspace.registry)
        XCTAssertNil(request.snapshot.provenance)
        let args=try workspace.compositeTrainingArguments(request:request,requestDirectory:root)
        XCTAssertEqual(try Data(contentsOf:URL(fileURLWithPath:args[1])),bytes)
        try Data("{}".utf8).write(to:original)
        XCTAssertThrowsError(try workspace.compositeTrainingArguments(request:request,requestDirectory:root))
    }
}
