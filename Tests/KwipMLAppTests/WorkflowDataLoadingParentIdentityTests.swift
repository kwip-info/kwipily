import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowDataLoadingParentIdentityTests:XCTestCase {
    private func object(_ value:JSONValue?)throws->[String:JSONValue] {guard case .object(let fields)=value else {throw WorkflowPhaseError("Invalid actual parent fixture.")};return fields}
    func testActualPinnedParentRecordedPopulationAndLegacyAbsence()throws {
        let fixture=DataLoadingPopulationArtifactTests()
        for (name,retained,m) in [("subset",5,3),("drop",4,2),("stream",5,1)] {
            let (manifest,_)=try fixture.fixture(name),identity=try XCTUnwrap(WorkflowDataLoadingParentIdentity.validate(manifest))
            XCTAssertEqual(identity.originalRows,12);XCTAssertEqual(identity.retainedRows,retained);XCTAssertEqual(identity.batchesPerEpoch,m)
        }
        let (legacy,_)=try fixture.fixture("legacy");XCTAssertNil(try WorkflowDataLoadingParentIdentity.validate(legacy))
    }
    func testParentRecordedRowsAndRecipeCannotContradictItsCopies()throws {
        let (manifest,_)=try DataLoadingPopulationArtifactTests().fixture()
        var changed=manifest,provenance=try object(manifest["provenance"])
        if case .array(let ids)=provenance["trainingRowIDs"] {provenance["trainingRowIDs"] = .array(Array(ids.reversed()))}
        changed["provenance"] = .object(provenance)
        XCTAssertThrowsError(try WorkflowDataLoadingParentIdentity.validate(changed))
        var settings=try object(manifest["settings"]);settings.removeValue(forKey:"datasetRecipe");changed=manifest;changed["settings"] = .object(settings)
        XCTAssertThrowsError(try WorkflowDataLoadingParentIdentity.validate(changed))
    }
}
