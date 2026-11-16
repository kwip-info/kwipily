import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowDevelopmentGuardEvidenceTests:XCTestCase {
    private func stage(_ runtime:RuntimeLocation,library:Data=Data("native fixture".utf8))throws {
        let root=runtime.workingDirectoryURL,native=root.appendingPathComponent("runner/native"),staged=root.appendingPathComponent(".local/native-watchdog")
        try FileManager.default.createDirectory(at:native,withIntermediateDirectories:true)
        try FileManager.default.createDirectory(at:staged,withIntermediateDirectories:true)
        let source=Data("canonical watchdog source".utf8)
        try source.write(to:native.appendingPathComponent("lifetime_watchdog.c"))
        try library.write(to:staged.appendingPathComponent("lifetime_watchdog.dylib"))
        let manifest:[String:Any] = ["schemaVersion":1,"sourceSHA256":WorkflowPhaseArtifacts.hash(source),"librarySHA256":WorkflowPhaseArtifacts.hash(library)]
        try JSONSerialization.data(withJSONObject:manifest,options:[.sortedKeys]).write(to:staged.appendingPathComponent("lifetime_watchdog.json"))
    }
    @MainActor func testConsistentGuardRebuildChangesRequestIdentityAndCurrentEvidence()throws {
        let helper=WorkflowExecutionRequestTests(),(root,owner,runtime)=try helper.setup()
        defer {try? FileManager.default.removeItem(at:root)}
        try stage(runtime)
        let first=try helper.capture(owner,runtime)
        let prefix=".local/native-watchdog/"
        XCTAssertNotNil(first.runtimeEvidence.workerFiles[prefix+"lifetime_watchdog.dylib"])
        XCTAssertNotNil(first.runtimeEvidence.workerFiles[prefix+"lifetime_watchdog.json"])
        XCTAssertTrue(try first.runtimeEvidence.matchesCurrent(runtime))
        try stage(runtime,library:Data("different valid build".utf8))
        let second=try helper.capture(owner,runtime)
        XCTAssertNotEqual(first.requestIdentity,second.requestIdentity)
        XCTAssertFalse(try first.runtimeEvidence.matchesCurrent(runtime))
        XCTAssertEqual(first.document,second.document)
    }
    @MainActor func testPartialInvalidAndSymlinkGuardEvidenceRejectsCapture()throws {
        for mutation in ["missingManifest","missingLibrary","libraryMismatch","sourceMismatch","symlink","oversizeManifest","extraField","booleanVersion"] {
            let helper=WorkflowExecutionRequestTests(),(root,owner,runtime)=try helper.setup()
            defer {try? FileManager.default.removeItem(at:root)}
            try stage(runtime)
            let staged=runtime.workingDirectoryURL.appendingPathComponent(".local/native-watchdog")
            let manifest=staged.appendingPathComponent("lifetime_watchdog.json"),library=staged.appendingPathComponent("lifetime_watchdog.dylib")
            switch mutation {
            case "missingManifest":try FileManager.default.removeItem(at:manifest)
            case "missingLibrary":try FileManager.default.removeItem(at:library)
            case "libraryMismatch":try Data("changed".utf8).write(to:library)
            case "sourceMismatch":try Data("changed".utf8).write(to:runtime.workingDirectoryURL.appendingPathComponent("runner/native/lifetime_watchdog.c"))
            case "symlink":
                let alternate=root.appendingPathComponent("alternate.dylib")
                try FileManager.default.moveItem(at:library,to:alternate)
                try FileManager.default.createSymbolicLink(at:library,withDestinationURL:alternate)
            case "oversizeManifest":try Data(repeating:32,count:16385).write(to:manifest)
            default:
                var object=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:manifest)) as? [String:Any])
                if mutation == "extraField" {object["unexpected"]=true} else {object["schemaVersion"]=true}
                try JSONSerialization.data(withJSONObject:object).write(to:manifest)
            }
            XCTAssertThrowsError(try helper.capture(owner,runtime),mutation)
        }
    }
    @MainActor func testOrdinaryAbsentGuardEvidenceRemainsCompatible()throws {
        let helper=WorkflowExecutionRequestTests(),(root,owner,runtime)=try helper.setup()
        defer {try? FileManager.default.removeItem(at:root)}
        let request=try helper.capture(owner,runtime)
        XCTAssertNil(request.runtimeEvidence.workerFiles[".local/native-watchdog/lifetime_watchdog.dylib"])
        let encoded=try JSONEncoder().encode(request.runtimeEvidence)
        XCTAssertEqual(try JSONDecoder().decode(WorkflowRuntimeEvidence.self,from:encoded),request.runtimeEvidence)
        XCTAssertTrue(try request.runtimeEvidence.matchesCurrent(runtime))
    }
    @MainActor func testGroupedExecutionWithoutGuardEvidenceRejectsBeforeReservation()throws {
        let helper=WorkflowExecutionRequestTests(),(root,owner,runtime)=try helper.setup()
        defer {try? FileManager.default.removeItem(at:root)}
        let spec=WorkflowExperimentSpecification(axes:[.init(field:"seed",values:[7,8])],budget:.init(maximumRuns:2,maximumTotalEpochs:6))
        XCTAssertTrue(owner.setExperimentSpecification(spec,for:.init(domain:.control,localID:"setup"),lease:owner.lease()))
        let base=try helper.capture(owner,runtime),file=root.appendingPathComponent("input.csv")
        try Data("a,b,label\n1,2,0\n".utf8).write(to:file)
        let staged=try WorkflowGroupOwnership.createGroup(workspace:root,projectID:owner.projectID)
        defer {staged.owner.close()}
        let inputs=try WorkflowExperimentInputs.captureBytes(addresses:base.plan.sourceAddresses,directory:staged.directory) {_ in try Data(contentsOf:file)}
        let request=try base.experimentVariant(ordinal:1,inputs:inputs,owner:owner)
        let session=WorkflowExecutionSession(request:request,owner:owner,sources:inputs.sources,directory:root.appendingPathComponent("unused"),reservation:.init(acquire:{XCTFail("Missing guard must not reserve");return nil},release:{_ in}),groupOwner:staged.owner,launch:{_,_,_,_,_ in XCTFail("Missing guard must not launch");return .init(cancel:{})},onTerminal:{_ in})
        XCTAssertFalse(session.start())
        XCTAssertTrue(session.statusMessage.lowercased().contains("guard"),session.statusMessage)
    }
}
