import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowSaveModelPublicationTests:XCTestCase {
    private let bytes=Data("verified checkpoint bytes".utf8)
    func testLearnedPreparationPublishesExactPinnedFileWithoutBroadeningInventory()throws {
        let root=try canonicalBackendTemporaryDirectory();defer{try? FileManager.default.removeItem(at:root)}
        let source=root.appendingPathComponent("source"),destination=root.appendingPathComponent("package")
        try FileManager.default.createDirectory(at:source,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        try bytes.write(to:source.appendingPathComponent("preparation.json"))
        let file=WorkflowArchiveFile(path:"preparation.json",bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))
        let pending=try WorkflowPendingExportPublication(destination:destination,directory:true);defer{pending.close()}
        XCTAssertThrowsError(try pending.writeClassifierFile(.init(path:"custom.py",bytes:bytes.count,sha256:file.sha256),from:source))
        XCTAssertThrowsError(try pending.writeClassifierFile(.init(path:file.path,bytes:20_000_001,sha256:file.sha256),from:source))
        XCTAssertThrowsError(try pending.writeClassifierFile(.init(path:file.path,bytes:file.bytes,sha256:String(repeating:"0",count:64)),from:source))
        try pending.writeClassifierFile(file,from:source)
        XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path))
        try pending.commit()
        XCTAssertEqual(try Data(contentsOf:destination.appendingPathComponent(file.path)),bytes)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:destination.path),[file.path])
    }
    private func folder()throws->URL {
        let url=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at:url,withIntermediateDirectories:false,attributes:[.posixPermissions:0o700])
        addTeardownBlock {try? FileManager.default.removeItem(at:url)};return url.resolvingSymlinksInPath()
    }
    func testResumeGroupsRemainPrivateUntilCommitAndFailedCopyCleansEveryChild()throws {
        let root=try canonicalBackendTemporaryDirectory(),source=root.appendingPathComponent("source");defer{try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        try FileManager.default.createDirectory(at:source,withIntermediateDirectories:false)
        try bytes.write(to:source.appendingPathComponent("training-state.safetensors"))
        let file=WorkflowArchiveFile(path:"training-state.safetensors",bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))
        let target=root.appendingPathComponent("resume"),pending=try WorkflowPendingExportPublication(destination:target,directory:true)
        try pending.writeModelGroup("latest",files:[file],from:source)
        XCTAssertFalse(FileManager.default.fileExists(atPath:target.path))
        pending.close()
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),["source"])
        let broken=try WorkflowPendingExportPublication(destination:target,directory:true)
        XCTAssertThrowsError(try broken.writeModelGroup("latest",files:[.init(path:file.path,bytes:file.bytes,sha256:String(repeating:"a",count:64))],from:source))
        broken.close();XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),["source"])
        XCTAssertEqual(try Data(contentsOf:source.appendingPathComponent(file.path)),bytes)
    }
    func testResumeGroupsRefuseSymlinkAndNeverReplaceConcurrentDestination()throws {
        let root=try canonicalBackendTemporaryDirectory(),source=root.appendingPathComponent("source");defer{try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700]);try FileManager.default.createDirectory(at:source,withIntermediateDirectories:false)
        let outside=root.appendingPathComponent("keep");try bytes.write(to:outside)
        try FileManager.default.createSymbolicLink(at:source.appendingPathComponent("model.safetensors"),withDestinationURL:outside)
        let file=WorkflowArchiveFile(path:"model.safetensors",bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes)),target=root.appendingPathComponent("resume")
        let pending=try WorkflowPendingExportPublication(destination:target,directory:true)
        XCTAssertThrowsError(try pending.writeModelGroup("base",files:[file],from:source));pending.close()
        XCTAssertEqual(try Data(contentsOf:outside),bytes)
        try FileManager.default.removeItem(at:source.appendingPathComponent(file.path));try bytes.write(to:source.appendingPathComponent(file.path))
        let other=try WorkflowPendingExportPublication(destination:target,directory:true);try other.writeModelGroup("base",files:[file],from:source)
        try FileManager.default.createDirectory(at:target,withIntermediateDirectories:false)
        XCTAssertThrowsError(try other.commit());other.close();XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:target.path),[])
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath:root.path).contains{$0.hasPrefix(".kwipml-export-")})
    }
    func testCheckpointRemainsUnpublishedUntilExclusiveCommit()throws {
        let root=try folder(),destination=root.appendingPathComponent("model.pt")
        let pending=try WorkflowExportPublication.stageExclusiveCheckpoint(bytes,sha256:WorkflowArchiveCodec.hash(bytes),to:destination)
        XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path))
        try pending.commit();pending.close()
        XCTAssertEqual(try Data(contentsOf:destination),bytes)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),["model.pt"])
        XCTAssertThrowsError(try pending.commit())
    }
    func testExistingFileAndDanglingSymlinkAreNeverOverwritten()throws {
        let root=try folder(),file=root.appendingPathComponent("existing.pt"),link=root.appendingPathComponent("link.pt")
        let original=Data("keep me".utf8);try original.write(to:file)
        try FileManager.default.createSymbolicLink(at:link,withDestinationURL:root.appendingPathComponent("missing"))
        for destination in [file,link] {
            XCTAssertThrowsError(try WorkflowExportPublication.stageExclusiveCheckpoint(bytes,sha256:WorkflowArchiveCodec.hash(bytes),to:destination))
        }
        XCTAssertEqual(try Data(contentsOf:file),original)
        XCTAssertEqual(try FileManager.default.destinationOfSymbolicLink(atPath:link.path),root.appendingPathComponent("missing").path)
    }
    func testConcurrentDestinationCreationWinsWithoutPartialOverwrite()throws {
        let root=try folder(),destination=root.appendingPathComponent("new.pt")
        let pending=try WorkflowExportPublication.stageExclusiveCheckpoint(bytes,sha256:WorkflowArchiveCodec.hash(bytes),to:destination)
        let other=Data("created by somebody else".utf8);try other.write(to:destination)
        XCTAssertThrowsError(try pending.commit());pending.close()
        XCTAssertEqual(try Data(contentsOf:destination),other)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),["new.pt"])
    }
    func testInvalidHashExtensionAndSymlinkParentRefuseBeforeStaging()throws {
        let root=try folder(),link=root.appendingPathComponent("alias")
        try FileManager.default.createSymbolicLink(at:link,withDestinationURL:root)
        for (destination,hash) in [(root.appendingPathComponent("new.pt"),String(repeating:"a",count:64)),
                                   (root.appendingPathComponent("new.txt"),WorkflowArchiveCodec.hash(bytes)),
                                   (link.appendingPathComponent("new.pt"),WorkflowArchiveCodec.hash(bytes))] {
            XCTAssertThrowsError(try WorkflowExportPublication.stageExclusiveCheckpoint(bytes,sha256:hash,to:destination))
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),["alias"])
    }
    func testCancelledCommitCleansItsTemporaryFile()async throws {
        let root=try folder(),destination=root.appendingPathComponent("cancel.pt")
        let pending=try WorkflowExportPublication.stageExclusiveCheckpoint(bytes,sha256:WorkflowArchiveCodec.hash(bytes),to:destination)
        let task=Task {
            withUnsafeCurrentTask {$0?.cancel()}
            do {try pending.commit();return false} catch is CancellationError {return true} catch {return false}
        }
        let cancelled=await task.value;XCTAssertTrue(cancelled);pending.close()
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),[])
    }
    private func package(in root:URL)throws->WorkflowTrainedExportPackage {
        let source=root.appendingPathComponent("source")
        try FileManager.default.createDirectory(at:source,withIntermediateDirectories:false,attributes:[.posixPermissions:0o700])
        let manifest=Data("reviewed manifest".utf8)
        try bytes.write(to:source.appendingPathComponent("checkpoint.pt"));try manifest.write(to:source.appendingPathComponent("export-manifest.json"))
        return .init(reference:.init(projectID:UUID().uuidString,directoryID:UUID()),runID:UUID().uuidString,
                     requestIdentity:String(repeating:"a",count:64),modelManifestSHA256:String(repeating:"b",count:64),
                     directory:source,files:[.init(path:"checkpoint.pt",bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))],manifestBytes:manifest,
                     inputContract:try TrainedModelInputContract(graph:ExampleFlows.parallelClassifier,registry:.builtIn(),columns:["x","y"],target:"label"))
    }
    func testPackagePublishesExactInventoryAndManifest()throws {
        let root=try folder(),package=try package(in:root),destination=root.appendingPathComponent("model")
        let pending=try WorkflowExportPublication.stageExclusive(package,to:destination)
        XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path));try pending.commit();pending.close()
        XCTAssertEqual(try Data(contentsOf:destination.appendingPathComponent("checkpoint.pt")),bytes)
        XCTAssertEqual(try Data(contentsOf:destination.appendingPathComponent("export-manifest.json")),package.manifestBytes)
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath:destination.path)),["checkpoint.pt","export-manifest.json"])
    }
    func testTamperedSourcePackageRefusesAndCleansStage()throws {
        let root=try folder(),package=try package(in:root)
        try Data("tampered".utf8).write(to:package.directory.appendingPathComponent("checkpoint.pt"))
        XCTAssertThrowsError(try WorkflowExportPublication.stageExclusive(package,to:root.appendingPathComponent("model")))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),["source"])
    }
    func testClosedDestinationGrantCannotBeUsed()throws {
        let root=try folder(),access=try WorkflowModelDestinationAccess(destination:root.appendingPathComponent("new.pt"),accessURL:root,revision:7)
        XCTAssertEqual(access.revision,7);try access.check();access.close();XCTAssertThrowsError(try access.check())
        let other=try folder()
        XCTAssertThrowsError(try WorkflowModelDestinationAccess(destination:root.appendingPathComponent("new.pt"),accessURL:other,revision:0))
    }
}
