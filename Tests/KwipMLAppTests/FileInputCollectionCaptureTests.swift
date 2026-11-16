import XCTest
import Darwin
@testable import KwipMLApp
import KwipMLCore

final class FileInputCollectionCaptureTests:XCTestCase {
    private func root()throws->URL {
        guard let resolved=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw WorkflowFileLeaseError.system(errno)}
        let parent=URL(fileURLWithPath:String(cString:resolved));free(resolved)
        let url=parent.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:url,withIntermediateDirectories:true)
        addTeardownBlock{try? FileManager.default.removeItem(at:url)}
        let checked=try WorkflowEvaluationOwnership.openDirectory(url);Darwin.close(checked)
        return url
    }
    private func configuration(_ overrides:[String:JSONValue]=[:])->[String:JSONValue] {FileInputConfiguration.newDefaults.merging(overrides){_,new in new}}
    private func capture(_ source:URL,root:URL,options:[String:JSONValue],preview:Int?=nil)throws->FileInputCollectionCapture {
        try FileInputCollectionCapture.capture(path:source.path,configuration:options,workspace:root,sourcePermissions:root.appendingPathComponent("permissions"),storageLimitBytes:nil,previewLimit:preview)
    }
    func testDirectoryCollectionUsesDeterministicHardExtensionAndGlobSelection()throws {
        let root=try root(),folder=root.appendingPathComponent("selected");try FileManager.default.createDirectory(at:folder.appendingPathComponent("sub"),withIntermediateDirectories:true)
        for name in ["b.txt","a.txt","unchecked.pdf",".hidden.txt","sub/c.txt"] {try Data(name.utf8).write(to:folder.appendingPathComponent(name))}
        let options=configuration(["sourceMode":.string("directory"),"extensions":.array([.string("txt")]),"includeSubfolders":.bool(true),"excludePatterns":.array([.string("b*")]),"readMode":.string("metadata")])
        let captured=try capture(folder,root:root,options:options);defer{captured.remove()}
        let manifest=try FileInputCollectionManifest.decode(Data(contentsOf:captured.source.url))
        XCTAssertEqual(manifest.members.map(\.relativePath),["a.txt","sub/c.txt"])
        XCTAssertTrue(manifest.complete);XCTAssertTrue(captured.source.collectionMembers.isEmpty)
        XCTAssertTrue(manifest.members.allSatisfy {$0.status=="metadata" && $0.capturePath==nil && $0.sha256==nil})
    }
    func testSingleExplicitFileBypassesDirectoryExtensionsAndCapturesExactContent()throws {
        let root=try root(),source=root.appendingPathComponent("source.txt"),raw=Data("hello café".utf8);try raw.write(to:source)
        let captured=try capture(source,root:root,options:configuration(["extensions":.array([.string("pdf")])]))
        defer{captured.remove()};let manifest=try FileInputCollectionManifest.decode(Data(contentsOf:captured.source.url))
        XCTAssertEqual(manifest.members.count,1);XCTAssertEqual(manifest.members[0].status,"content")
        let member=try XCTUnwrap(captured.source.collectionMembers.first)
        XCTAssertEqual(try Data(contentsOf:member.url),raw);XCTAssertEqual(member.sha256,WorkflowArchiveCodec.hash(raw))
        let staged=captured.source.stagedInputs();XCTAssertEqual(staged.files.count,2)
        XCTAssertEqual(staged.binding,"input/file-collections/"+captured.source.sha256+"/manifest.json")
    }
    func testMetadataOnlyLargeFileHasNoBodyCaptureAndPreviewDoesNotClaimCensus()throws {
        let root=try root(),folder=root.appendingPathComponent("files");try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
        let large=folder.appendingPathComponent("large.txt");try Data().write(to:large)
        let handle=try FileHandle(forWritingTo:large);try handle.truncate(atOffset:67_108_865);try handle.close()
        let captured=try capture(folder,root:root,options:configuration(["sourceMode":.string("directory"),"readMode":.string("metadata")]),preview:8)
        defer{captured.remove()};let manifest=try FileInputCollectionManifest.decode(Data(contentsOf:captured.source.url))
        XCTAssertEqual(manifest.members[0].bytes,67_108_865);XCTAssertTrue(captured.source.collectionMembers.isEmpty)
        XCTAssertFalse(manifest.complete)
    }
    func testSymbolicLinksFailOrSkipWithoutGrantingTargets()throws {
        let root=try root(),folder=root.appendingPathComponent("files");try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
        let secret=root.appendingPathComponent("outside.txt");try Data("outside".utf8).write(to:secret)
        try FileManager.default.createSymbolicLink(at:folder.appendingPathComponent("link.txt"),withDestinationURL:secret)
        XCTAssertThrowsError(try capture(folder,root:root,options:configuration(["sourceMode":.string("directory"),"symlinks":.string("error")])))
        let skipped=try capture(folder,root:root,options:configuration(["sourceMode":.string("directory"),"symlinks":.string("skip")]));defer{skipped.remove()}
        XCTAssertTrue(try FileInputCollectionManifest.decode(Data(contentsOf:skipped.source.url)).members.isEmpty)
        XCTAssertThrowsError(try capture(folder.appendingPathComponent("link.txt"),root:root,options:configuration(["symlinks":.string("skip")])))
    }
    func testContentLimitsRefuseOrRecordWithoutClaimingCopiedBytes()throws {
        let root=try root(),source=root.appendingPathComponent("source.txt");try Data("hello".utf8).write(to:source)
        XCTAssertThrowsError(try capture(source,root:root,options:configuration(["maxFileBytes":.integer(4)])))
        let recorded=try capture(source,root:root,options:configuration(["maxFileBytes":.integer(4),"errorPolicy":.string("record")]));defer{recorded.remove()}
        let member=try XCTUnwrap(FileInputCollectionManifest.decode(Data(contentsOf:recorded.source.url)).members.first)
        XCTAssertEqual(member.status,"error");XCTAssertNil(member.capturePath);XCTAssertNil(member.sha256)
        XCTAssertTrue(recorded.source.collectionMembers.isEmpty)
    }
    func testDirectoryCountDepthAndEffectiveStorageBudgetsFailClearly()throws {
        let root=try root(),folder=root.appendingPathComponent("files");try FileManager.default.createDirectory(at:folder.appendingPathComponent("nested"),withIntermediateDirectories:true)
        try Data("a".utf8).write(to:folder.appendingPathComponent("a.txt"));try Data("b".utf8).write(to:folder.appendingPathComponent("b.txt"))
        XCTAssertThrowsError(try capture(folder,root:root,options:configuration(["sourceMode":.string("directory"),"maxFiles":.integer(1)])))
        try FileManager.default.createDirectory(at:folder.appendingPathComponent("nested/deeper"),withIntermediateDirectories:true)
        XCTAssertThrowsError(try capture(folder,root:root,options:configuration(["sourceMode":.string("directory"),"includeSubfolders":.bool(true),"maxDepth":.integer(1)])))
        XCTAssertThrowsError(try FileInputCollectionCapture.capture(path:folder.path,configuration:configuration(["sourceMode":.string("directory")]),workspace:root,sourcePermissions:root.appendingPathComponent("permissions"),storageLimitBytes:1))
    }
    func testPreviewEightFilesAndFullCaptureRetainDifferentHonestCensus()throws {
        let root=try root(),folder=root.appendingPathComponent("files");try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
        for index in 0..<12 {try Data("record".utf8).write(to:folder.appendingPathComponent(String(format:"%02d.txt",index)))}
        let options=configuration(["sourceMode":.string("directory"),"readMode":.string("metadata")])
        let preview=try capture(folder,root:root,options:options,preview:8),full=try capture(folder,root:root,options:options)
        defer{preview.remove();full.remove()}
        let previewManifest=try FileInputCollectionManifest.decode(Data(contentsOf:preview.source.url)),fullManifest=try FileInputCollectionManifest.decode(Data(contentsOf:full.source.url))
        XCTAssertEqual(previewManifest.members.count,8);XCTAssertFalse(previewManifest.complete)
        XCTAssertEqual(fullManifest.members.count,12);XCTAssertTrue(fullManifest.complete)
        XCTAssertEqual(previewManifest.members.map(\.relativePath),Array(fullManifest.members.prefix(8).map(\.relativePath)))
        XCTAssertNotEqual(preview.source.sha256,full.source.sha256)
    }

    func testFileChangedAfterCopyBeforeCollectionPublicationIsRefused()throws {
        let root=try root(),folder=root.appendingPathComponent("files");try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
        let original=folder.appendingPathComponent("a.txt");try Data("first".utf8).write(to:original)
        try Data("later".utf8).write(to:folder.appendingPathComponent("b.txt"))
        var hookCalled=false
        XCTAssertThrowsError(try FileInputCollectionCapture.capture(path:folder.path,configuration:configuration(["sourceMode":.string("directory")]),workspace:root,sourcePermissions:root.appendingPathComponent("permissions"),storageLimitBytes:nil,beforeFinalVerification:{
            hookCalled=true
            // Write through the existing inode: directory identity/timestamps stay unchanged.
            let handle=try FileHandle(forWritingTo:original);try handle.write(contentsOf:Data("other".utf8));try handle.close()
        })) {error in XCTAssertTrue(error.localizedDescription.contains("Selected file changed before collection verification"))}
        XCTAssertTrue(hookCalled,"The mutation must occur after successful capture, not pass through an unrelated admission failure.")
    }

    func testImageInspectionDoesNotCaptureBodiesButFullRunDoes()throws {
        let root=try root(),source=root.appendingPathComponent("frame.png");try Data("image bytes are not decoded by the capture host".utf8).write(to:source)
        let preview=try capture(source,root:root,options:configuration(),preview:8),full=try capture(source,root:root,options:configuration())
        defer{preview.remove();full.remove()}
        let previewManifest=try FileInputCollectionManifest.decode(Data(contentsOf:preview.source.url)),fullManifest=try FileInputCollectionManifest.decode(Data(contentsOf:full.source.url))
        XCTAssertEqual(previewManifest.members.first?.status,"metadata");XCTAssertTrue(preview.source.collectionMembers.isEmpty)
        XCTAssertEqual(fullManifest.members.first?.status,"content");XCTAssertEqual(full.source.collectionMembers.count,1)
        XCTAssertEqual(previewManifest.selection,fullManifest.selection)
    }

}
