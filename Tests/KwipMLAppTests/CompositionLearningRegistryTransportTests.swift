import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class CompositionLearningRegistryTransportTests: XCTestCase {
    private func fixture(captured:Bool) throws -> (URL,WorkflowToolRequest) {
        var root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        guard let canonical=realpath(root.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        root=URL(fileURLWithPath:String(cString:canonical));free(canonical)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let project=UUID().uuidString,operation=UUID().uuidString
        let directory=root.appendingPathComponent("workflow-operations/\(project)/\(operation)")
        try FileManager.default.createDirectory(at:directory.appendingPathComponent("input"),withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        let names=captured ? ["assessment.json","submission.json","model-registry.json"] : ["assessment.json","submission.json"]
        var files:[WorkflowArchiveFile]=[]
        for name in names {
            let bytes=Data("{}".utf8);try DiagnosticFileIO.writeImmutable(bytes,root:directory,parts:["input",name],limit:2_000_000)
            files.append(.init(path:"input/"+name,bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes)))
        }
        var args=["--learning-evidence-request",directory.appendingPathComponent("input/assessment.json").path,
                  "--learning-submission",directory.appendingPathComponent("input/submission.json").path,
                  "--learning-submission-sha256",files[1].sha256]
        if captured {args += ["--registry",directory.appendingPathComponent("input/model-registry.json").path]}
        return (directory,.init(operationID:operation,projectID:project,requestIdentity:String(repeating:"a",count:64),toolMode:.learningAssessment,
            arguments:args,inputFiles:files,executionBudget:try .resolve(global:.factory,projectMaximumRunSeconds:60),remainingSeconds:30))
    }
    func testCapturedThirdInputAndLegacyTwoInputsHaveExactDistinctShapes() throws {
        for selected in [false,true] {
            let (directory,request)=try fixture(captured:selected)
            XCTAssertEqual(request.arguments.count,selected ? 8:6)
            XCTAssertEqual(request.inputFiles.count,selected ? 3:2)
            XCTAssertNoThrow(try request.validate(directory:directory,verifyInputs:false))
            XCTAssertNoThrow(try request.validate(directory:directory,verifyInputs:true))
            XCTAssertEqual(try WorkflowToolRequest.decode(JSONEncoder().encode(request)),request)
        }
    }
    func testForeignMissingDuplicateAndTamperedRegistryRefuse() throws {
        let (directory,request)=try fixture(captured:true)
        let base=try JSONSerialization.jsonObject(with:JSONEncoder().encode(request)) as! [String:Any]
        for change in ["missingFile","missingFlag","foreignPath","duplicateFlag","runtimeMode"] {
            var value=base
            var arguments=request.arguments
            switch change {
            case "missingFile":value["inputFiles"]=Array(request.inputFiles.dropLast()).map { ["path":$0.path,"bytes":$0.bytes,"sha256":$0.sha256] as [String:Any] }
            case "missingFlag":arguments=Array(arguments.dropLast(2))
            case "foreignPath":arguments[7]=directory.appendingPathComponent("input/foreign.json").path
            case "duplicateFlag":arguments += Array(arguments.suffix(2))
            default:value["toolMode"]="learningRuntime"
            }
            value["arguments"]=arguments
            XCTAssertThrowsError(try WorkflowToolRequest.decode(JSONSerialization.data(withJSONObject:value)),change)
        }
        try Data("changed".utf8).write(to:directory.appendingPathComponent("input/model-registry.json"))
        XCTAssertThrowsError(try request.validate(directory:directory,verifyInputs:true))
    }
    @MainActor func testAdapterConfigurationIsBoundedResettableAndNeverAddedToDiscovery() async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:repository,allowDevelopment:true)
        let registry=try Data(contentsOf:runtime.registryURL),adapter=LearningAssessmentAdapter()
        try adapter.configureCompositionRegistry(registry)
        XCTAssertThrowsError(try adapter.configureCompositionRegistry(Data(repeating:32,count:2_000_001)))
        XCTAssertThrowsError(try adapter.configureCompositionRegistry(Data("null".utf8)))
        try adapter.configureCompositionRegistry(nil);try adapter.configureCompositionRegistry(registry)
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:root)}
        try adapter.configureExecution(workspace:root,projectID:UUID().uuidString,budget:ResolvedExecutionBudget.resolve(global:.factory,projectMaximumRunSeconds:60))
        let finished=expectation(description:"Discovery keeps no assessment registry input")
        adapter.launchTool={_,_,_,_,mode,inputs,_,_,line,exit in
            XCTAssertEqual(mode,.learningRuntime);XCTAssertTrue(inputs.isEmpty)
            line("{\"event\":\"learningRuntime\",\"versions\":{\"pytorch\":\"2.14.1\"}}")
            exit(0,"");finished.fulfill();return .init(cancel:{})
        }
        adapter.discoverRuntime(runtime)
        await fulfillment(of:[finished],timeout:5)
        XCTAssertNil(adapter.error)
    }
}
