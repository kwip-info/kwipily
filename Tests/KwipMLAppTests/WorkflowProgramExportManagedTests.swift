import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowProgramExportManagedTests:XCTestCase {
    // These bytes exercise native transport admission, not executable-program numerics.
    private func fixture()throws->(URL,WorkflowToolRequest) {
        guard let pointer=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(pointer)}
        let workspace=URL(fileURLWithPath:String(cString:pointer)).appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:workspace)}
        let project=UUID().uuidString,operation=UUID().uuidString,pin=String(repeating:"a",count:64)
        let directory=workspace.appendingPathComponent("workflow-operations/\(project)/\(operation)")
        try FileManager.default.createDirectory(at:directory.appendingPathComponent("source"),withIntermediateDirectories:true)
        try FileManager.default.createDirectory(at:directory.appendingPathComponent("input"),withIntermediateDirectories:true)
        let budget=try ResolvedExecutionBudget.resolve(global:.factory,projectMaximumRunSeconds:60)
        var sourceFiles:[WorkflowArchiveFile]=[]
        for name in ["model.py","checkpoint.pt","inference.py","input-contract.json"] {
            let bytes=name == "input-contract.json" ? Data("{\"schemaVersion\":1,\"input\":{\"dtype\":\"float32\",\"shape\":[\"N\",2]}}".utf8):Data("native admission fixture".utf8)
            try bytes.write(to:directory.appendingPathComponent("source/"+name))
            sourceFiles.append(.init(path:name,bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes)))
        }
        let manifest:JSONValue = .object(["schemaVersion":1,"kind":.string("kwipmlTrainedModel"),"runID":.string(UUID().uuidString),"requestIdentity":.string(pin),"modelManifestSHA256":.string(pin),"files":.array(sourceFiles.map{.object(["path":.string($0.path),"bytes":.integer(Int64($0.bytes)),"sha256":.string($0.sha256)])})])
        let manifestBytes=try WorkflowArchiveCodec.encode(manifest)
        try manifestBytes.write(to:directory.appendingPathComponent("source/export-manifest.json"))
        let inner=WorkflowProgramExportRequest(schemaVersion:2,projectID:project,exportID:operation,requestIdentity:pin,sourceDirectory:directory.appendingPathComponent("source").path,sourceManifestSHA256:WorkflowArchiveCodec.hash(manifestBytes),configuration:ProgramExportConfiguration.defaults,executionBudget:budget)
        let bytes=try inner.encoded();try bytes.write(to:directory.appendingPathComponent("input/program-export.json"))
        let files=sourceFiles.map{WorkflowArchiveFile(path:"source/"+$0.path,bytes:$0.bytes,sha256:$0.sha256)} + [WorkflowArchiveFile(path:"source/export-manifest.json",bytes:manifestBytes.count,sha256:WorkflowArchiveCodec.hash(manifestBytes)),WorkflowArchiveFile(path:"input/program-export.json",bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))]
        return (directory,.init(operationID:operation,projectID:project,requestIdentity:pin,toolMode:.programExport,arguments:["--export-program",directory.appendingPathComponent("input/program-export.json").path,"--program-request-sha256",WorkflowArchiveCodec.hash(bytes)],inputFiles:files,executionBudget:budget,remainingSeconds:55))
    }
    func testManagedRequestPinsCompleteTrainedSourceAndRoundTrips()throws {
        let (directory,request)=try fixture()
        XCTAssertEqual(try WorkflowToolRequest.decode(request.encoded()),request)
        try request.validate(directory:directory,verifyInputs:true)
    }
    func testManagedAdmissionRechecksSourceBytesAndInnerOperationIdentity()throws {
        let (directory,request)=try fixture()
        try request.validate(directory:directory,verifyInputs:true)
        try Data("modified".utf8).write(to:directory.appendingPathComponent("source/checkpoint.pt"))
        XCTAssertThrowsError(try request.validate(directory:directory,verifyInputs:true))
        let (_,fresh)=try fixture()
        let foreign=WorkflowToolRequest(operationID:fresh.operationID,projectID:fresh.projectID,requestIdentity:String(repeating:"b",count:64),toolMode:.programExport,arguments:fresh.arguments,inputFiles:fresh.inputFiles,executionBudget:fresh.executionBudget,remainingSeconds:55)
        XCTAssertThrowsError(try foreign.validate(directory:URL(fileURLWithPath:fresh.arguments[1]).deletingLastPathComponent().deletingLastPathComponent(),verifyInputs:true))
    }
    func testManagedInventoryRejectsUnknownDuplicateAndOversizedSource()throws {
        let (_,request)=try fixture()
        try request.validate()
        for extra in [WorkflowArchiveFile(path:"source/private.csv",bytes:1,sha256:String(repeating:"a",count:64)),request.inputFiles[0],WorkflowArchiveFile(path:"source/checkpoint.pt",bytes:100_000_001,sha256:String(repeating:"a",count:64))] {
            let wrong=WorkflowToolRequest(operationID:request.operationID,projectID:request.projectID,requestIdentity:request.requestIdentity,toolMode:.programExport,arguments:request.arguments,inputFiles:request.inputFiles+[extra],executionBudget:request.executionBudget,remainingSeconds:55)
            XCTAssertThrowsError(try wrong.validate())
        }
    }
    @MainActor func testManagedTransportReviewsPinnedOutputOnlyAfterOwnedTerminal()async throws {
        try await exerciseTransport(cancel:false)
    }
    @MainActor func testManagedCancellationKeepsOwnerUntilActualWorkerExit()async throws {
        try await exerciseTransport(cancel:true)
    }
    @MainActor private func exerciseTransport(cancel:Bool)async throws {
        let (original,request)=try fixture()
        let sourceDirectory=original.appendingPathComponent("source")
        let sourceBytes=try Data(contentsOf:sourceDirectory.appendingPathComponent("export-manifest.json"))
        let header=try XCTUnwrap(JSONSerialization.jsonObject(with:sourceBytes) as? [String:Any])
        let source=WorkflowTrainedExportPackage(reference:.init(projectID:request.projectID,directoryID:UUID()),runID:try XCTUnwrap(header["runID"] as? String),requestIdentity:request.requestIdentity,modelManifestSHA256:request.requestIdentity,directory:sourceDirectory,files:request.inputFiles.filter{$0.path.hasPrefix("source/") && $0.path != "source/export-manifest.json"}.map{.init(path:String($0.path.dropFirst(7)),bytes:$0.bytes,sha256:$0.sha256)},manifestBytes:sourceBytes,inputContract:try TrainedModelInputContract(graph:ExampleFlows.parallelClassifier,registry:.builtIn(),columns:["x","y"],target:"label"))
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:repository,allowDevelopment:true)
        let workspace=original.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let window=try ExecutionBudgetWindow(policy:request.executionBudget,startedAtUptime:ProcessInfo.processInfo.systemUptime)
        let held=expectation(description:"owned worker awaiting terminal")
        var pendingExit:((Int32,String)->Void)?,owned:URL?,returned=false,cancelCalls=0
        let transport=WorkflowProgramExportManagedTransport(expectedTorchVersion:"2.14.1",launcher:{_,args,_,line,exit in
            let index=try XCTUnwrap(args.firstIndex(of:"--kwip-operation-supervise"))
            let directory=URL(fileURLWithPath:args[index+1]);owned=directory
            let outer=try WorkflowToolRequest.decode(Data(contentsOf:directory.appendingPathComponent("request.json")))
            let ready:[String:Any]=["event":"operationReady","operationID":outer.operationID,"toolMode":"programExport","requestSHA256":args[index+2],"ownershipSHA256":args[index+3]]
            line(String(decoding:try JSONSerialization.data(withJSONObject:ready),as:UTF8.self))
            if !cancel {
                let (packageRoot,_,_,_,_)=try WorkflowProgramExportTests.fixture()
                defer {try? FileManager.default.removeItem(at:packageRoot)}
                let target=directory.appendingPathComponent("package")
                try FileManager.default.copyItem(at:packageRoot,to:target)
                let manifestPath=target.appendingPathComponent("manifest.json")
                var manifest=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:manifestPath)) as? [String:Any])
                manifest["sourceIdentity"]=["runID":source.runID,"requestIdentity":source.requestIdentity,"modelManifestSHA256":source.modelManifestSHA256,"exportManifestSHA256":WorkflowArchiveCodec.hash(sourceBytes)]
                let manifestBytes=try JSONSerialization.data(withJSONObject:manifest,options:[.sortedKeys]);try manifestBytes.write(to:manifestPath)
                for name in ["programExportStarted","programExportCompleted"] {
                    var event:[String:Any]=["event":name,"projectID":outer.projectID,"exportID":outer.operationID,"requestIdentity":outer.requestIdentity]
                    if name == "programExportCompleted" {event["manifestSHA256"]=WorkflowArchiveCodec.hash(manifestBytes)}
                    line(String(decoding:try JSONSerialization.data(withJSONObject:event),as:UTF8.self))
                }
            }
            pendingExit=exit;held.fulfill()
            return .init(cancel:{cancelCalls += 1})
        })
        let task=Task {()throws->WorkflowProgramExportPackage in
            defer {returned=true}
            return try await transport.capture(source:source,workspace:workspace,runtime:runtime,configuration:ProgramExportConfiguration(values:ProgramExportConfiguration.defaults),executionBudget:window.policy,budgetWindow:window)
        }
        await fulfillment(of:[held],timeout:15)
        let directory=try XCTUnwrap(owned)
        XCTAssertFalse(returned)
        XCTAssertThrowsError(try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.shared))
        if cancel {task.cancel();for _ in 0..<20 {await Task.yield()};XCTAssertFalse(returned);XCTAssertEqual(cancelCalls,1)}
        try XCTUnwrap(pendingExit)(0,"")
        if cancel {
            do {_ = try await task.value;XCTFail("Cancelled capture must refuse late output")} catch is CancellationError {} catch {XCTFail("Unexpected cancellation error: \(error)")}
            XCTAssertFalse(FileManager.default.fileExists(atPath:directory.path))
        } else {
            let package=try await task.value
            XCTAssertEqual(package.identity.runID,source.runID);XCTAssertEqual(package.fileCount,5)
            XCTAssertTrue(FileManager.default.fileExists(atPath:package.directory.path))
            package.remove();XCTAssertFalse(FileManager.default.fileExists(atPath:directory.path))
        }
    }

}
