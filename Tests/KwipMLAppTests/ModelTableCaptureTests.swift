import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelTableCaptureTests:XCTestCase {
    private func makeRoot()throws->URL {
        let root=try canonicalBackendTemporaryDirectory()
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        return root
    }
    func testCompleteOwnedTableAboveInlineLimitCopiesExactBytesWithFixedRoleName()throws {
        let root=try makeRoot();defer {try? FileManager.default.removeItem(at:root)}
        let original=root.appendingPathComponent("prepared-source.json"),validation=root.appendingPathComponent("held-out.json"),target=root.appendingPathComponent("input")
        try FileManager.default.createDirectory(at:target,withIntermediateDirectories:false)
        let bytes=Data(repeating:32,count:8_000_001);try bytes.write(to:original);try Data("{}".utf8).write(to:validation)
        let inputs:[ModelTableCapture.Input]=[.init(role:.training,url:original,sha256:WorkflowArchiveCodec.hash(bytes)),.init(role:.validation,url:validation,sha256:WorkflowArchiveCodec.hash(Data("{}".utf8)))]
        let files=try ModelTableCapture.inventory(inputs)
        XCTAssertEqual(files.first(where:{$0.path=="input/training.table.json"})?.bytes,8_000_001)
        try ModelTableCapture.copy(inputs,files:files,to:target)
        XCTAssertEqual(try Data(contentsOf:target.appendingPathComponent("training.table.json")),bytes)
        XCTAssertFalse(FileManager.default.fileExists(atPath:target.appendingPathComponent(original.lastPathComponent).path))
        XCTAssertThrowsError(try ModelTableCapture.copy(inputs,files:files,to:target),"Never overwrite a captured table")
    }
    func testMissingRoleWrongPinSymlinkAndChangedCaptureAreRefused()throws {
        let root=try makeRoot();defer {try? FileManager.default.removeItem(at:root)}
        let table=root.appendingPathComponent("table.json"),validation=root.appendingPathComponent("validation.json"),target=root.appendingPathComponent("input")
        try Data("{}".utf8).write(to:table);try Data("[]".utf8).write(to:validation);try FileManager.default.createDirectory(at:target,withIntermediateDirectories:false)
        let pin=WorkflowArchiveCodec.hash(Data("{}".utf8))
        let train=ModelTableCapture.Input(role:.training,url:table,sha256:pin),valid=ModelTableCapture.Input(role:.validation,url:validation,sha256:WorkflowArchiveCodec.hash(Data("[]".utf8)))
        XCTAssertThrowsError(try ModelTableCapture.inventory([train]))
        XCTAssertThrowsError(try ModelTableCapture.inventory([train,train]))
        XCTAssertThrowsError(try ModelTableCapture.inventory([.init(role:.training,url:table,sha256:String(repeating:"a",count:64)),valid]))
        let link=root.appendingPathComponent("link.json");try FileManager.default.createSymbolicLink(at:link,withDestinationURL:table)
        XCTAssertThrowsError(try ModelTableCapture.inventory([.init(role:.training,url:link,sha256:pin),valid]))
        let files=try ModelTableCapture.inventory([train,valid]);try Data("changed".utf8).write(to:table)
        XCTAssertThrowsError(try ModelTableCapture.copy([train,valid],files:files,to:target))
    }
    func testOverBudgetAndCancelledInventoryDoNotPublish()async throws {
        let root=try makeRoot();defer {try? FileManager.default.removeItem(at:root)}
        let path=root.appendingPathComponent("oversized.json");FileManager.default.createFile(atPath:path.path,contents:nil)
        let fd=try FileHandle(forWritingTo:path);try fd.truncate(atOffset:UInt64(TableTransferLimits.artifactBytes+1));try fd.close()
        let input=ModelTableCapture.Input(role:.training,url:path,sha256:String(repeating:"a",count:64))
        let valid=ModelTableCapture.Input(role:.validation,url:path,sha256:input.sha256)
        XCTAssertThrowsError(try ModelTableCapture.inventory([input,valid]))
        let task=Task {withUnsafeCurrentTask {$0?.cancel()};return try ModelTableCapture.inventory([input,valid])}
        do {_=try await task.value;XCTFail("Cancelled capture was admitted")} catch is CancellationError {} catch {XCTFail("Expected cancellation, got \(error)")}
    }
    func testVersionedTableContractPreservesLegacyAndRequiresBothTrainingRoles()throws {
        let root=try makeRoot();defer {try? FileManager.default.removeItem(at:root)}
        let project=UUID().uuidString,operation=UUID().uuidString,identity=String(repeating:"a",count:64)
        let directory=root.appendingPathComponent("workflow-operations/\(project)/\(operation)")
        try FileManager.default.createDirectory(at:directory.appendingPathComponent("input"),withIntermediateDirectories:true)
        let tables=[WorkflowArchiveFile(path:"input/training.table.json",bytes:8_000_001,sha256:identity),.init(path:"input/validation.table.json",bytes:2,sha256:identity)]
        let source=["config.json","tokenizer.json","model.safetensors"].map{WorkflowArchiveFile(path:"source/"+$0,bytes:2,sha256:identity)}
        let request=WorkflowToolRequest(operationID:operation,projectID:project,requestIdentity:identity,toolMode:.modelAsset,arguments:["--model-asset",directory.appendingPathComponent("input/model-request.json").path],inputFiles:tables+source+[.init(path:"input/model-request.json",bytes:100,sha256:identity)],executionBudget:try .resolve(global:.factory,projectMaximumRunSeconds:60),remainingSeconds:30)
        func inner(version:Int,action:String="train")throws {
            let value:[String:JSONValue]=["schemaVersion":.integer(Int64(version)),"projectID":.string(project),"operationID":.string(operation),"requestIdentity":.string(identity),"action":.string(action),"settings":.object([:]),"sourceIdentity":.string(identity)]
            try DiagnosticFileIO.writeAtomic(WorkflowArchiveCodec.encode(value),root:directory,parts:["input","model-request.json"])
        }
        try ModelAssetToolContract.validate(arguments:request.arguments,files:request.inputFiles,directory:directory)
        try inner(version:2);try ModelAssetToolContract.verifyCaptureVersion(request,directory:directory)
        try inner(version:1);XCTAssertThrowsError(try ModelAssetToolContract.verifyCaptureVersion(request,directory:directory))
        try inner(version:2,action:"inspect");XCTAssertThrowsError(try ModelAssetToolContract.verifyCaptureVersion(request,directory:directory))
        try inner(version:3);XCTAssertNoThrow(try ModelAssetToolContract.verifyCaptureVersion(request,directory:directory))
        try inner(version:4);XCTAssertThrowsError(try ModelAssetToolContract.verifyCaptureVersion(request,directory:directory))
        let segment=WorkflowArchiveFile(path:"input/table-pack-"+String(repeating:"a",count:32)+".kwipdata",bytes:100,sha256:identity)
        let segmented=WorkflowToolRequest(operationID:operation,projectID:project,requestIdentity:identity,toolMode:.modelAsset,arguments:request.arguments,inputFiles:request.inputFiles+[segment],executionBudget:request.executionBudget,remainingSeconds:30)
        XCTAssertNoThrow(try ModelAssetToolContract.validate(arguments:segmented.arguments,files:segmented.inputFiles,directory:directory))
        XCTAssertNoThrow(try ModelAssetToolContract.verifyCaptureVersion(segmented,directory:directory))
        try inner(version:3);XCTAssertThrowsError(try ModelAssetToolContract.verifyCaptureVersion(segmented,directory:directory))
        try inner(version:4,action:"inspect");XCTAssertThrowsError(try ModelAssetToolContract.verifyCaptureVersion(segmented,directory:directory))
    }

    func testPinnedSegmentsCopyOnceAndRecheckEverySourceBeforePublication()throws {
        let root=try makeRoot();defer {try? FileManager.default.removeItem(at:root)}
        let target=root.appendingPathComponent("input"),a=root.appendingPathComponent("a.json"),b=root.appendingPathComponent("b.json")
        try FileManager.default.createDirectory(at:target,withIntermediateDirectories:false)
        let bytes=Data("pinned segment fixture".utf8),table=Data("{}".utf8)
        let name="table-pack-"+String(repeating:"a",count:32)+".kwipdata",pack=root.appendingPathComponent("table-pack-"+String(repeating:"a",count:32)+".kwipdata")
        try bytes.write(to:pack);try table.write(to:a);try table.write(to:b)
        let segment=WorkflowArchiveFile(path:name,bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))
        let inputs:[ModelTableCapture.Input]=[.init(role:.training,url:a,sha256:WorkflowArchiveCodec.hash(table),retainedSegments:[segment]),.init(role:.validation,url:b,sha256:WorkflowArchiveCodec.hash(table),retainedSegments:[segment])]
        let files=try ModelTableCapture.inventory(inputs)
        XCTAssertEqual(files.filter {$0.path=="input/"+name}.count,1)
        try ModelTableCapture.copy(inputs,files:files,to:target)
        XCTAssertEqual(try Data(contentsOf:target.appendingPathComponent(name)),bytes)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:target.path).count,3)
        try Data("tamper".utf8).write(to:pack)
        XCTAssertThrowsError(try ModelTableCapture.inventory(inputs))
        let other=root.appendingPathComponent("other");try FileManager.default.createDirectory(at:other,withIntermediateDirectories:false)
        XCTAssertThrowsError(try ModelTableCapture.copy(inputs,files:files,to:other))
        try FileManager.default.removeItem(at:pack)
        XCTAssertThrowsError(try ModelTableCapture.inventory(inputs))
    }

}
