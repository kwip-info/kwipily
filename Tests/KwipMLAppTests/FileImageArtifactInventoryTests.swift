import XCTest
import KwipMLCore
@testable import KwipMLApp

final class FileImageArtifactInventoryTests:XCTestCase {
    // Golden bytes from the Python encode_pixels fixture; includes exact canonical header.
    private let pixels=Data(base64Encoded:"S1dJUElNRzEKAAAA4XsiYnl0ZU9yZGVyIjoibGl0dGxlIiwiY2hhbm5lbHMiOjMsImR0eXBlIjoidWludDgiLCJoZWlnaHQiOjEsImtpbmQiOiJpbWFnZVBpeGVscyIsImxheW91dCI6IkhXQyIsInBvbGljeSI6eyJhY3R1YWxNb2RlIjoiUkdCIiwiY29sb3JNb2RlIjoicHJlc2VydmUiLCJvcmllbnRhdGlvbiI6InN0b3JlZCIsInBhbGV0dGUiOiJleHBhbmRSR0JBIn0sInNjaGVtYVZlcnNpb24iOjEsIndpZHRoIjoyfQAAAAAAAA==")!
    private func fixture()throws->(URL,Data,[WorkflowArchiveFile],FileImageReference,[[String:Any]]) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let reference=try FileImageReference(sha256:"0ecb1b5f13725a3904def9eb16bf6fdc54ae61c898ae230061c16838217777b3",width:2,height:1,channels:3,dtype:"uint8",layout:"HWC")
        let object=try JSONSerialization.jsonObject(with:JSONEncoder().encode(reference))
        let name="image-"+reference.sha256+".kwipimg"
        try pixels.write(to:root.appendingPathComponent(name))
        let preview=try JSONSerialization.data(withJSONObject:["rows":[[object]]])
        try preview.write(to:root.appendingPathComponent("previews.json"))
        let entries:[[String:Any]]=[["path":name,"kind":"imagePixels","bytes":pixels.count,"sha256":reference.sha256,"image":object],
            ["path":"previews.json","kind":"previews","bytes":preview.count,"sha256":WorkflowArchiveCodec.hash(preview)]]
        let files=[WorkflowArchiveFile(path:name,bytes:pixels.count,sha256:reference.sha256),.init(path:"previews.json",bytes:preview.count,sha256:WorkflowArchiveCodec.hash(preview))]
        return (root,try JSONSerialization.data(withJSONObject:["artifacts":entries]),files,reference,entries)
    }
    func testCrossLanguageContainerAndPreviewResolveOnlyThroughPinnedOwnedInventory()throws {
        let(root,manifest,files,reference,_)=try fixture()
        XCTAssertEqual(try FileImageArtifactInventory.verify(manifest:manifest,files:files,prefix:"",directory:root),[reference.sha256:reference])
        let decoded=try FileImageReference.decode(container:pixels,expected:reference)
        XCTAssertEqual(decoded.pixels,Data(repeating:0,count:6))
    }
    func testMissingChangedAndDuplicateAssetsNeverGrantPreviewAuthority()throws {
        let(root,manifest,files,reference,entries)=try fixture()
        let missing=try JSONSerialization.data(withJSONObject:["artifacts":[entries[1]]])
        XCTAssertThrowsError(try FileImageArtifactInventory.verify(manifest:missing,files:files,prefix:"",directory:root))
        let duplicate=try JSONSerialization.data(withJSONObject:["artifacts":entries+[entries[0]]])
        XCTAssertThrowsError(try FileImageArtifactInventory.verify(manifest:duplicate,files:files,prefix:"",directory:root))
        var changed=pixels;changed[changed.count-1]=1
        try changed.write(to:root.appendingPathComponent("image-"+reference.sha256+".kwipimg"))
        XCTAssertThrowsError(try FileImageArtifactInventory.verify(manifest:manifest,files:files,prefix:"",directory:root))
    }
}
