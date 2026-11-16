import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class FileInputInspectionExecutionTests:XCTestCase {
    private func fixture(largeOptions:Bool=false)throws->(URL,RuntimeLocation,FileInputCollectionCapture,[String:JSONValue],ExecutionBudgetWindow) {
        let (workspace,_,_,_)=try WorkflowEvaluationRequestTests().fixture()
        addTeardownBlock{try? FileManager.default.removeItem(at:workspace)}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let store=WorkspaceStore(rootOverride:repository,storageOverride:workspace,detectCapabilities:false)
        let runtime=try XCTUnwrap(store.runtime,"Actual development runtime is required for captured-tool staging")
        let directory=workspace.appendingPathComponent("chosen-files");try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        for index in 0..<10 {try Data("bounded text \(index)".utf8).write(to:directory.appendingPathComponent(String(format:"%02d.txt",index)))}
        var config=FileInputConfiguration.newDefaults;config["sourceMode"] = .string("directory")
        if largeOptions {
            config["excludePatterns"] = .array((0..<64).map {.string("not-selected-\($0)-"+String(repeating:"x",count:80)+".txt")})
        }
        let capture=try FileInputCollectionCapture.capture(path:directory.path,configuration:config,workspace:workspace,sourcePermissions:workspace.appendingPathComponent("permissions"),storageLimitBytes:nil,previewLimit:8)
        addTeardownBlock{capture.remove()}
        let window=try ExecutionBudgetWindow(policy:.resolve(global:.factory,projectMaximumRunSeconds:nil),startedAtUptime:ProcessInfo.processInfo.systemUptime)
        return (workspace,runtime,capture,config,window)
    }
    func testCapturedInspectionAcceptsValidFileOptionsBetweenFourAndThirtyTwoKiB()async throws {
        let (workspace,runtime,capture,config,window)=try fixture(largeOptions:true)
        var launched=false,exit:((Int32,String)->Void)?
        let finished=expectation(description:"Large bounded File Input configuration is joined")
        let handle=try await SourceInspectionExecution.launch(workspace:workspace,projectID:UUID().uuidString,runtime:runtime,window:window,sourceURL:capture.source.url,sourceSHA256:capture.source.sha256,separator:",",configuration:config,collectionMembers:capture.source.collectionMembers,isFileInput:true,launcher:{_,args,_,_,onExit in
            launched=true;exit=onExit
            let index=try XCTUnwrap(args.firstIndex(of:"--kwip-operation-supervise")),directory=URL(fileURLWithPath:args[index+1])
            let request=try WorkflowToolRequest.decode(Data(contentsOf:directory.appendingPathComponent("request.json")))
            XCTAssertEqual(request.arguments[2],"--file-options")
            XCTAssertGreaterThan(request.arguments[3].utf8.count,4096)
            XCTAssertLessThanOrEqual(request.arguments[3].utf8.count,32768)
            try request.validate(directory:directory,verifyInputs:true)
            return .init(cancel:{})
        },onExit:{code,_,result in XCTAssertNotEqual(code,0);XCTAssertNil(result);finished.fulfill()})
        XCTAssertTrue(launched);withExtendedLifetime(handle){exit?(130,"")}
        await fulfillment(of:[finished],timeout:5)
    }
    func testCapturedInspectionStagesManifestAndEightMembersBeforeWorkerLaunch()async throws {
        let (workspace,runtime,capture,config,window)=try fixture()
        var launched=false,exit:((Int32,String)->Void)?
        let finished=expectation(description:"Captured collection inspection is joined")
        let handle=try await SourceInspectionExecution.launch(workspace:workspace,projectID:UUID().uuidString,runtime:runtime,window:window,sourceURL:capture.source.url,sourceSHA256:capture.source.sha256,separator:",",configuration:config,collectionMembers:capture.source.collectionMembers,isFileInput:true,launcher:{_,args,_,_,onExit in
            launched=true;exit=onExit
            let index=try XCTUnwrap(args.firstIndex(of:"--kwip-operation-supervise")),directory=URL(fileURLWithPath:args[index+1])
            let request=try WorkflowToolRequest.decode(Data(contentsOf:directory.appendingPathComponent("request.json")))
            XCTAssertEqual(request.inputFiles.count,9)
            XCTAssertEqual(request.arguments[2],"--file-options")
            XCTAssertTrue(request.inputFiles.allSatisfy {$0.path.hasPrefix("input/file-collections/"+capture.source.sha256+"/")})
            XCTAssertFalse(request.arguments.contains(workspace.appendingPathComponent("chosen-files").path))
            try request.validate(directory:directory,verifyInputs:true)
            return .init(cancel:{})
        },onExit:{code,_,result in XCTAssertNotEqual(code,0);XCTAssertNil(result);finished.fulfill()})
        XCTAssertTrue(launched);withExtendedLifetime(handle){exit?(130,"")}
        await fulfillment(of:[finished],timeout:5)
    }
    func testActualSupervisedFileInspectionReadsBoundedClosedCollection()async throws {
        let (workspace,runtime,capture,config,window)=try fixture()
        let finished=expectation(description:"Genuine File Input inspection")
        let handle=try await SourceInspectionExecution.launch(workspace:workspace,projectID:UUID().uuidString,runtime:runtime,window:window,sourceURL:capture.source.url,sourceSHA256:capture.source.sha256,separator:",",configuration:config,collectionMembers:capture.source.collectionMembers,isFileInput:true,onExit:{code,message,result in
            XCTAssertEqual(code,0,message)
            XCTAssertEqual(result?.rows.count,8);XCTAssertEqual(result?.sourceSHA256,capture.source.sha256)
            XCTAssertEqual(result?.collection?.selectedFiles,8);XCTAssertEqual(result?.collection?.complete,false)
            XCTAssertEqual(result?.collection?.imagePixelsIncluded,false)
            XCTAssertTrue(result?.columns.contains("text_content")==true)
            finished.fulfill()
        })
        await fulfillment(of:[finished],timeout:30);withExtendedLifetime(handle){}
    }
}
