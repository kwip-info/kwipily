import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class OCRWorkflowTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    func testReferenceAuditRefusesLinkedNamespace()throws {
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        let peer=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:peer)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        try FileManager.default.createDirectory(at:peer,withIntermediateDirectories:true)
        try FileManager.default.createSymbolicLink(at:root.appendingPathComponent("projects"),withDestinationURL:peer)
        XCTAssertThrowsError(try OCRPackStore.referenceCount(workspace:root))
    }
    func testMissingPackIsActionableAndRetainedReferencesProtectRemoval()throws {
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        XCTAssertThrowsError(try OCRPackStore.inputs(workspace:root)) {error in
            XCTAssertTrue(error.localizedDescription.contains("Settings → Models & assets"))
        }
        XCTAssertEqual(try OCRPackStore.referenceCount(workspace:root),0)
        let projects=root.appendingPathComponent("projects")
        try FileManager.default.createDirectory(at:projects,withIntermediateDirectories:true)
        let project=projects.appendingPathComponent("retained.kwipml")
        try Data(OCRConfiguration.operatorID.utf8).write(to:project)
        XCTAssertEqual(try OCRPackStore.referenceCount(workspace:root),1)
        XCTAssertThrowsError(try OCRPackStore.removeUnreferenced(workspace:root))
        try Data(repeating:32,count:2_000_001).write(to:project)
        XCTAssertThrowsError(try OCRPackStore.referenceCount(workspace:root))
    }
    func testPracticeLaunchSourcesSharedLessonAndArtwork()throws {
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(OCRPracticeExample.exampleID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        let path=try XCTUnwrap(host.bindings.path(for:OCRPracticeExample.source))
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath:path)),Set(OCRPracticeExample.resourceNames))
        XCTAssertEqual(ComponentLessonDestination.entryID(type:OCRConfiguration.operatorID),OCRPracticeExample.exampleID)
        XCTAssertNotNil(LessonContentCatalog.content(entryID:OCRPracticeExample.exampleID))
        XCTAssertNotNil(ComponentArtworkCatalog.image(for:OCRConfiguration.operatorID))
    }
    func testVerifiedUnusedPackRemovalPreservesProjectDirectory()throws {
        guard let pack=ProcessInfo.processInfo.environment["KWIPML_TEST_OCR_PACK"] else {throw XCTSkip("Explicit reviewed OCR archives required.")}
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        let destination=OCRPackStore.directory(workspace:root)
        try FileManager.default.createDirectory(at:destination,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        let projects=root.appendingPathComponent("projects")
        try FileManager.default.createDirectory(at:projects,withIntermediateDirectories:true)
        for asset in OCRPackCatalog.assets {
            try FileManager.default.copyItem(at:URL(fileURLWithPath:pack).appendingPathComponent(asset.name.replacingOccurrences(of:".zip",with:".pth.zip")),to:destination.appendingPathComponent(asset.name))
        }
        XCTAssertTrue(OCRPackStore.installed(workspace:root))
        try OCRPackStore.removeUnreferenced(workspace:root)
        XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath:projects.path))
    }
    func testRealCapturedOCRWorkerProducesCompleteSharedTablesAndHistory()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_SKLEARN_TEST_PYTHON"],let pack=ProcessInfo.processInfo.environment["KWIPML_TEST_OCR_PACK"] else {throw XCTSkip("Explicit reviewed OCR environment required.")}
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(OCRPracticeExample.exampleID,device:"cpu"),workspace.statusMessage)
        let destination=OCRPackStore.directory(workspace:workspace.workspaceDirectory)
        try FileManager.default.createDirectory(at:destination,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        for asset in OCRPackCatalog.assets {let bytes=try Data(contentsOf:URL(fileURLWithPath:pack).appendingPathComponent(asset.name.replacingOccurrences(of:".zip",with:".pth.zip")));let target=destination.appendingPathComponent(asset.name);try bytes.write(to:target,options:.withoutOverwriting);try FileManager.default.setAttributes([.posixPermissions:0o600],ofItemAtPath:target.path)}
        _=try OCRPackStore.inputs(workspace:workspace.workspaceDirectory)
        let host=try XCTUnwrap(workspace.workflowSession)
        host.executionLaunch={_,arguments,directory,onLine,onExit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),arguments,directory,onLine,onExit)}
        XCTAssertTrue(host.run(),host.statusMessage)
        let deadline=Date().addingTimeInterval(100)
        while workspace.isBusy && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        let preparation=try XCTUnwrap(host.currentDataPreparation)
        XCTAssertEqual(preparation.phase,.completed,preparation.statusMessage+"\n"+preparation.messages.joined(separator:"\n"))
        let output=try XCTUnwrap(preparation.preparationOutputs["ocr"])
        XCTAssertNotNil(output["documents"]);XCTAssertNotNil(output["regions"])
        XCTAssertThrowsError(try OCRPackStore.removeUnreferenced(workspace:workspace.workspaceDirectory))
        XCTAssertTrue(FileManager.default.fileExists(atPath:destination.path))
    }
}
