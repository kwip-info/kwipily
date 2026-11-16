import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DataLoadingRuntimeAdmissionTests:XCTestCase {
    func testIndependentExactReaderDescriptorsAreRequired()throws {
        let dataset=try WorkflowArchiveCodec.encode(BundledDataLoadingComponents.entries["dataset_source"]!.runtimeContract)
        let loader=try WorkflowArchiveCodec.encode(BundledDataLoadingComponents.entries["data_loader"]!.runtimeContract)
        let verified=DataLoadingRuntimeCapabilities(dataset:dataset,loader:loader)
        XCTAssertTrue(verified.contains("dataset_source"));XCTAssertTrue(verified.contains("data_loader"))
        XCTAssertFalse(verified.contains("linear"));XCTAssertFalse(DataLoadingRuntimeCapabilities().contains("dataset_source"))
        XCTAssertFalse(DataLoadingRuntimeCapabilities(dataset:loader,loader:dataset).contains("dataset_source"))
        var raw=try XCTUnwrap(JSONSerialization.jsonObject(with:loader) as? [String:Any]);raw["inOrder"]=false
        let changed=try JSONSerialization.data(withJSONObject:raw)
        XCTAssertFalse(DataLoadingRuntimeCapabilities(loader:changed).contains("data_loader"))
    }
    func testProgramExportHasSeparateExactArtifactAdmission()throws {
        let bytes=try WorkflowArchiveCodec.encode(BundledProgramExportCapabilities.entry.runtimeContract)
        XCTAssertTrue(ProgramExportRuntimeCapability(descriptor:bytes).isAvailable)
        XCTAssertFalse(ProgramExportRuntimeCapability().isAvailable)
        let weights=try WorkflowArchiveCodec.encode(BundledLifecycleCapabilities.entry.runtimeContract)
        XCTAssertFalse(ProgramExportRuntimeCapability(descriptor:weights).isAvailable)
        var raw=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any]);raw["unsafeLoadIsSandboxed"]=true
        XCTAssertFalse(ProgramExportRuntimeCapability(descriptor:try JSONSerialization.data(withJSONObject:raw)).isAvailable)
    }
}
