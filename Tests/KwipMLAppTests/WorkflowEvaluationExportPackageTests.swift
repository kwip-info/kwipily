import XCTest
import Foundation
import KwipMLCore
@testable import KwipMLApp

/// Artifact packaging checks use synthetic bytes; actual model validity belongs to
/// worker integration tests, not these inventory/publication fixtures.
final class WorkflowEvaluationExportPackageTests:XCTestCase {
    private let pin=String(repeating:"a",count:64)
    private func fixture(includeRows:Bool=false,version:Int=1,device:String="cpu",dependencies:[String:Data]=[:])throws->(URL,WorkflowEvaluationExportIdentity,String) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let expected=WorkflowEvaluationExportIdentity(evaluationID:UUID().uuidString,evaluationManifestSHA256:pin,checkpointSHA256:pin,stateDigest:pin,snapshotReceiptSHA256:pin)
        var data=Dictionary(uniqueKeysWithValues:WorkflowEvaluationExportPackage.requiredLimits.keys.map {($0,Data("fixture".utf8))})
        data.merge(dependencies) {_,new in new}
        let contract:[String:Any]=["schemaVersion":1,"kind":"portableEvaluationInput","columns":[["name":"first","dtype":"float64","nullable":false],["name":"second","dtype":"float64","nullable":false],["name":"label","dtype":"int64","nullable":false]],"featureColumns":["first","second"],"targetColumn":"label","dtype":"float32","sampleShape":[2],"fittedStateDigests":[],"appliedPrefix":[],"testEndpoint":["node":"split","port":"test"],"testCount":14,"membershipDigest":pin,"selectedDataVersion":pin]
        data["input-contract.json"]=try JSONSerialization.data(withJSONObject:contract)
        data["fitted-parameters.json"]=Data("[]".utf8)
        data["saved-summary.json"]=try JSONSerialization.data(withJSONObject:["schemaVersion":1,"device":device])
        if includeRows {for name in WorkflowEvaluationExportPackage.optionalLimits.keys {data[name]=Data("protected".utf8)}}
        let entries=data.map {WorkflowArchiveFile(path:$0.key,bytes:$0.value.count,sha256:WorkflowArchiveCodec.hash($0.value))}
        let manifest=WorkflowEvaluationExportManifest(schemaVersion:version,kind:"portableFrozenEvaluation",includesTestRows:includeRows,identities:expected,files:entries)
        let bytes=try WorkflowArchiveCodec.encode(manifest)
        for (name,bytes) in data {try bytes.write(to:root.appendingPathComponent(name))}
        try bytes.write(to:root.appendingPathComponent("manifest.json"))
        return (root,expected,WorkflowArchiveCodec.hash(bytes))
    }
    func testCorrectedPortableVersionsAndLegacyDeviceCompatibility()throws {
        for (version,device) in [(1,"cpu"),(2,"cpu"),(2,"mps")] {
            let(root,identity,pin)=try fixture(version:version,device:device)
            XCTAssertNoThrow(try WorkflowEvaluationExportPackage.verify(directory:root,expected:identity,includesTestRows:false,manifestSHA256:pin))
        }
        let(root,identity,pin)=try fixture(version:1,device:"mps")
        let original=try Data(contentsOf:root.appendingPathComponent("manifest.json"))
        XCTAssertThrowsError(try WorkflowEvaluationExportPackage.verify(directory:root,expected:identity,includesTestRows:false,manifestSHA256:pin)) {error in
            XCTAssertTrue(error.localizedDescription.contains("MPS"))
            XCTAssertTrue(error.localizedDescription.contains("may be incorrect"))
            XCTAssertTrue(error.localizedDescription.lowercased().contains("rerun"))
        }
        XCTAssertEqual(try Data(contentsOf:root.appendingPathComponent("manifest.json")),original)
    }
    func testVerifiedReviewShowsSelectedColumnsAndPublishesExactFiles()throws {
        for includeRows in [false,true] {
            let(root,identity,pin)=try fixture(includeRows:includeRows)
            let package=try WorkflowEvaluationExportPackage.verify(directory:root,expected:identity,includesTestRows:includeRows,manifestSHA256:pin)
            XCTAssertEqual(package.includesTestRows,includeRows)
            XCTAssertEqual(package.columns.map(\.name),["first","second","label"])
            XCTAssertEqual(package.featureColumns,["first","second"])
            XCTAssertEqual(package.testCount,14)
            let destination=root.deletingLastPathComponent().appendingPathComponent(UUID().uuidString)
            defer {try? FileManager.default.removeItem(at:destination)}
            try package.publish(to:destination)
            for file in package.files {XCTAssertEqual(try Data(contentsOf:root.appendingPathComponent(file.path)),try Data(contentsOf:destination.appendingPathComponent(file.path)))}
            XCTAssertEqual(try Data(contentsOf:root.appendingPathComponent("manifest.json")),try Data(contentsOf:destination.appendingPathComponent("manifest.json")))
            XCTAssertThrowsError(try package.publish(to:destination))
        }
    }
    func testChangedPrivacySelectionOrManifestCannotBecomeReview()throws {
        let(root,identity,pin)=try fixture()
        XCTAssertThrowsError(try WorkflowEvaluationExportPackage.verify(directory:root,expected:identity,includesTestRows:true,manifestSHA256:pin))
        XCTAssertThrowsError(try WorkflowEvaluationExportPackage.verify(directory:root,expected:identity,includesTestRows:false,manifestSHA256:String(repeating:"b",count:64)))
        try Data("private".utf8).write(to:root.appendingPathComponent("unlisted.csv"))
        XCTAssertThrowsError(try WorkflowEvaluationExportPackage.verify(directory:root,expected:identity,includesTestRows:false,manifestSHA256:pin))
    }
    func testChangedReviewedArtifactRejectsPublicationWithoutLeavingDestination()throws {
        let(root,identity,pin)=try fixture()
        let package=try WorkflowEvaluationExportPackage.verify(directory:root,expected:identity,includesTestRows:false,manifestSHA256:pin)
        try Data("different".utf8).write(to:root.appendingPathComponent("model.py"))
        let parent=root.deletingLastPathComponent().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:parent,withIntermediateDirectories:false)
        defer {try? FileManager.default.removeItem(at:parent)}
        XCTAssertThrowsError(try package.publish(to:parent.appendingPathComponent("export")))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:parent.path),[])
    }
    func testSymlinkArtifactNeverBecomesReviewedPackage()throws {
        let(root,identity,pin)=try fixture()
        let file=root.appendingPathComponent("weights.pt")
        try FileManager.default.removeItem(at:file)
        try FileManager.default.createSymbolicLink(at:file,withDestinationURL:root.appendingPathComponent("model.py"))
        XCTAssertThrowsError(try WorkflowEvaluationExportPackage.verify(directory:root,expected:identity,includesTestRows:false,manifestSHA256:pin))
    }
    func testCurrentWorkerDependencyInventoryAndLegacyPackagesRemainExactlyBounded()throws {
        // Independently listed current worker modules; do not derive from the native allowlist.
        let modules=["preprocess_parameters.py","dataflow_temporal.py","model_output_configuration.py"]
        for includeRows in [false,true] {
            for subset in [Array(modules.prefix(1)),modules] {
                let dependencies=Dictionary(uniqueKeysWithValues:subset.map {($0,Data("reviewed bundled runtime".utf8))})
                let(root,identity,pin)=try fixture(includeRows:includeRows,version:2,dependencies:dependencies)
                let package=try WorkflowEvaluationExportPackage.verify(directory:root,expected:identity,includesTestRows:includeRows,manifestSHA256:pin)
                XCTAssertEqual(package.fileCount,12+subset.count+(includeRows ? 2:0))
                XCTAssertTrue(Set(subset).isSubset(of:Set(package.files.map(\.path))))
            }
        }
    }
    func testUnknownMissingTamperedAndOversizedRuntimeDependenciesAreRefused()throws {
        let cases:[(String,Data)] = [("arbitrary_helper.py",Data("unreviewed".utf8)),("dataflow_temporal.py",Data(repeating:0,count:200_001))]
        for(name,bytes) in cases {
            let(root,identity,pin)=try fixture(version:2,dependencies:[name:bytes])
            XCTAssertThrowsError(try WorkflowEvaluationExportPackage.verify(directory:root,expected:identity,includesTestRows:false,manifestSHA256:pin))
        }
        for tamper in [false,true] {
            let(root,identity,pin)=try fixture(version:2,dependencies:["model_output_configuration.py":Data("captured runtime".utf8)])
            let file=root.appendingPathComponent("model_output_configuration.py")
            if tamper {try Data("modified runtime".utf8).write(to:file)} else {try FileManager.default.removeItem(at:file)}
            XCTAssertThrowsError(try WorkflowEvaluationExportPackage.verify(directory:root,expected:identity,includesTestRows:false,manifestSHA256:pin))
        }
    }

}
