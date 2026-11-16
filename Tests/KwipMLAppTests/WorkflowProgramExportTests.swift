import XCTest
import Darwin
import Combine
@testable import KwipMLCore
@testable import KwipMLApp

final class WorkflowProgramExportTests:XCTestCase {
    func testExactRequestRoundTripAndForeignControlsRefuse()throws {
        let root=try Self.temporaryRoot()
        let project=UUID().uuidString,export=UUID().uuidString,pin=String(repeating:"a",count:64)
        let source=root.appendingPathComponent("workflow-program-exports/\(project)/\(export)/source")
        try FileManager.default.createDirectory(at:source,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let request=WorkflowProgramExportRequest(projectID:project,exportID:export,requestIdentity:pin,sourceDirectory:source.path,sourceManifestSHA256:pin,configuration:ProgramExportConfiguration.defaults)
        let raw=try request.encoded()
        XCTAssertEqual(try WorkflowProgramExportRequest.decode(raw),request)
        var value=try JSONSerialization.jsonObject(with:raw) as! [String:Any]
        value["unexpected"]=true
        XCTAssertThrowsError(try WorkflowProgramExportRequest.decode(JSONSerialization.data(withJSONObject:value)))
        value.removeValue(forKey:"unexpected");value["configuration"]=["strict":true]
        XCTAssertThrowsError(try WorkflowProgramExportRequest.decode(JSONSerialization.data(withJSONObject:value)))
        var wrong=request;wrong.schemaVersion=3;XCTAssertThrowsError(try wrong.validate())
    }
    func testStrictPackageFramingAndExactPublication()throws {
        let (root,identity,configuration,pin,sourcePin)=try fixture()
        defer {try? FileManager.default.removeItem(at:root)}
        let package=try WorkflowProgramExportPackage.verify(directory:root,expected:identity,configuration:configuration,sourceManifestSHA256:sourcePin,manifestSHA256:pin,expectedTorchVersion:"2.14.1")
        XCTAssertEqual(package.fileCount,5);XCTAssertEqual(package.inputDtype,"float32")
        XCTAssertEqual(package.inputSampleShape,[2]);XCTAssertEqual(package.outputSampleShape,[3])
        let target=root.deletingLastPathComponent().appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:target)}
        try package.publish(to:target)
        XCTAssertEqual(try Data(contentsOf:target.appendingPathComponent("program.pt2")),Data("framing fixture, not a native executable proof".utf8))
        XCTAssertThrowsError(try package.publish(to:target))
        XCTAssertThrowsError(try WorkflowProgramExportPackage.verify(directory:root,expected:identity,configuration:configuration,sourceManifestSHA256:sourcePin,manifestSHA256:pin,expectedTorchVersion:"future"))
        try Data("unexpected".utf8).write(to:root.appendingPathComponent("private.csv"))
        XCTAssertThrowsError(try WorkflowProgramExportPackage.verify(directory:root,expected:identity,configuration:configuration,sourceManifestSHA256:sourcePin,manifestSHA256:pin,expectedTorchVersion:"2.14.1"))
    }
    func testChangedIdentityShapeAndSourceBindingCannotBecomeReview()throws {
        let (root,identity,configuration,pin,sourcePin)=try fixture()
        defer {try? FileManager.default.removeItem(at:root)}
        XCTAssertThrowsError(try WorkflowProgramExportPackage.verify(directory:root,expected:.init(runID:UUID().uuidString,requestIdentity:identity.requestIdentity,modelManifestSHA256:identity.modelManifestSHA256),configuration:configuration,sourceManifestSHA256:sourcePin,manifestSHA256:pin,expectedTorchVersion:"2.14.1"))
        let package=try WorkflowProgramExportPackage.verify(directory:root,expected:identity,configuration:configuration,sourceManifestSHA256:sourcePin,manifestSHA256:pin,expectedTorchVersion:"2.14.1")
        try Data("changed".utf8).write(to:root.appendingPathComponent("program.pt2"))
        let target=root.deletingLastPathComponent().appendingPathComponent(UUID().uuidString)
        XCTAssertThrowsError(try package.publish(to:target));XCTAssertFalse(FileManager.default.fileExists(atPath:target.path))
    }
    private static func temporaryRoot()throws->URL {
        guard let pointer=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(pointer)}
        return URL(fileURLWithPath:String(cString:pointer)).appendingPathComponent(UUID().uuidString)
    }
    static func fixture()throws->(URL,WorkflowProgramExportIdentity,ProgramExportConfiguration,String,String) {
        let root=try Self.temporaryRoot()
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
        let a=String(repeating:"a",count:64),b=String(repeating:"b",count:64)
        let identity=WorkflowProgramExportIdentity(runID:UUID().uuidString,requestIdentity:a,modelManifestSHA256:a)
        let config=try ProgramExportConfiguration(values:ProgramExportConfiguration.defaults)
        var files:[WorkflowArchiveFile]=[]
        for name in WorkflowProgramExportPackage.limits.keys {
            let bytes=Data((name == "program.pt2" ? "framing fixture, not a native executable proof":"fixture").utf8)
            try bytes.write(to:root.appendingPathComponent(name));files.append(.init(path:name,bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes)))
        }
        let manifest:JSONValue = .object(["schemaVersion":1,"kind":.string("kwipmlExportedProgram"),"runtime":.object(["torch":.string("2.14.1"),"device":.string("cpu")]),"configuration":.object(config.values),"input":.object(["dtype":.string("float32"),"sampleShape":.array([2])]),"output":.object(["dtype":.string("float32"),"sampleShape":.array([3])]),"sourceIdentity":.object(["runID":.string(identity.runID),"requestIdentity":.string(a),"modelManifestSHA256":.string(a),"exportManifestSHA256":.string(b)]),"shapeConstraints":.object(["mode":.string("static"),"leadingAxis":.string("batch"),"minimum":2,"maximum":2,"example":2]),"rangeConstraints":.array([]),"state":.array([]),"files":.array(files.map{.object(["path":.string($0.path),"bytes":.integer(Int64($0.bytes)),"sha256":.string($0.sha256)])})])
        let bytes=try WorkflowArchiveCodec.encode(manifest);try bytes.write(to:root.appendingPathComponent("manifest.json"))
        return (root,identity,config,WorkflowArchiveCodec.hash(bytes),b)
    }
    private func fixture()throws->(URL,WorkflowProgramExportIdentity,ProgramExportConfiguration,String,String) {try Self.fixture()}
}
