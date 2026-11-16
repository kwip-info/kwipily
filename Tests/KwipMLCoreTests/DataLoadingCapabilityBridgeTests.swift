import XCTest
@testable import KwipMLCore

final class DataLoadingCapabilityBridgeTests:XCTestCase {
    func testReaderRecipesAreStateControlsOutsideForward()throws {
        for id in ["dataset_source","data_loader"] {
            let definition=try XCTUnwrap(BundledDataLoadingComponents.entries[id]?.definition)
            let descriptor=try DataLoadingCapabilityBridge.reviewedDescriptor(definition)
            XCTAssertTrue(CapabilityContractValidator.validate(descriptor).isValid)
            XCTAssertEqual(descriptor.ports.count,1)
            XCTAssertEqual(descriptor.ports.first?.schema.kind,.state)
            XCTAssertEqual(descriptor.ports.first?.schema.state?.key,id == "dataset_source" ? "datasetRecipe":"loaderRecipe")
            XCTAssertEqual(descriptor.fields,definition.fields)
            XCTAssertFalse(descriptor.lifecycle.phases.contains(.forward))
        }
    }
    func testAlteredPortsCannotAcquireReviewedCapability()throws {
        var definition=try XCTUnwrap(BundledDataLoadingComponents.entries["dataset_source"]?.definition)
        definition.outputs=["tensor"]
        XCTAssertThrowsError(try DataLoadingCapabilityBridge.reviewedDescriptor(definition))
    }
}
