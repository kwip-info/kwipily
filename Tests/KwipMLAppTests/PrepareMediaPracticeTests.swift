import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PrepareMediaPracticeTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    func testPracticeCompilesBothMediaPathsWithPrivateSourcesOutsideGraph()throws {
        let example=PreparedWorkflowExample.component(PrepareMediaPracticeExample.exampleID),document=try example.document(device:"cpu")
        let registry=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        let plan=try WorkflowDataPreparationPlan.compile(document,registry:registry)
        XCTAssertEqual(document.nodes.count,6);XCTAssertEqual(document.edges.count,4)
        XCTAssertEqual(Set(plan.nodes.map(\.operatorID)),["kwip.data.sourceFiles",PrepareMediaConfiguration.operatorID,"kwip.data.view"])
        XCTAssertTrue(example.sourceResources.isEmpty)
        let bytes=try WorkflowCodec.encode(document);XCTAssertFalse(String(decoding:bytes,as:UTF8.self).contains("/Users/"))
    }
    @MainActor func testPracticeLaunchOwnsSyntheticFilesAndRoutesSharedLessonAndArtwork()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(PrepareMediaPracticeExample.exampleID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        let a=try XCTUnwrap(host.bindings.path(for:PrepareMediaPracticeExample.imageSource)),b=try XCTUnwrap(host.bindings.path(for:PrepareMediaPracticeExample.audioSource))
        XCTAssertEqual(a,b);XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath:a)),Set(PrepareMediaPracticeExample.resourceNames))
        XCTAssertEqual(ComponentLessonDestination.entryID(type:PrepareMediaConfiguration.operatorID),PrepareMediaPracticeExample.exampleID)
        XCTAssertNotNil(LessonContentCatalog.content(entryID:PrepareMediaPracticeExample.exampleID))
        XCTAssertNotNil(ComponentArtworkCatalog.image(for:PrepareMediaConfiguration.operatorID))
    }
}


extension PrepareMediaPracticeTests {
    func testPreserved24BitPreparedAudioPreviewUsesOriginalPrecisionWithoutChangingSamples()throws {
        func reference(_ kind:String,_ dtype:String,_ metadata:[String:JSONValue])throws->MediaArtifactReference {
            let raw:JSONValue = .object(["schemaVersion":1,"kind":.string(kind),"sha256":.string(String(repeating:"a",count:64)),"shape":.array([1,1]),"axes":.array((kind=="audioSamplesRef" ? ["samples","channels"]:["channels","samples"]).map(JSONValue.string)),"dtype":.string(dtype),"metadata":.object(metadata)])
            return try MediaArtifactReference.decode(JSONEncoder().encode(raw))
        }
        let source=try reference("audioSamplesRef","int32",["sampleRate":8000,"sourceBits":24])
        let prepared=try reference("preparedTensorRef","int32",["sampleRate":8000])
        let samples=Data([0,0,64,0]) // +0.5 at preserved 24-bit precision in int32 storage.
        XCTAssertEqual(MediaArtifactPreviewView.amplitude(samples,index:0,ref:prepared,source:source),0.5)
        let float=try reference("preparedTensorRef","float32",["sampleRate":8000])
        XCTAssertEqual(MediaArtifactPreviewView.amplitude(Data([0,0,0,63]),index:0,ref:float,source:source),0.5)
        XCTAssertEqual(samples,Data([0,0,64,0]))
    }
}
