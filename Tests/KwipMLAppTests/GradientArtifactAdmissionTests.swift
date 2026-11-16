import XCTest
import KwipMLCore
@testable import KwipMLApp

/// Reserved selected-policy metadata must not silently alter historical omitted-policy meaning.
final class GradientArtifactAdmissionTests:XCTestCase {
    private var legacy:[String:JSONValue] {
        ["settings":.object(["epochs":2,"batchSize":5,"batchesPerEpoch":3]),
         "runtime":.object(["device":.string("cpu"),"torch":.string("2.14.1")]),
         "provenance":.object([:]),"checkpoint":.object(["contract":.object([:])])]
    }
    func testLegacyOmittedPolicyDoesNotRequireNewMetadataOrPermission()throws {
        XCTAssertNoThrow(try WorkflowPhaseArtifacts.validateUnselectedGradientMetadata(legacy,derivativeCapture:nil))
        XCTAssertNoThrow(try WorkflowPhaseArtifacts.validateUnselectedGradientMetadata([:],derivativeCapture:nil))
        // Existing gradient summaries are a separate older feature, not derivative-policy selection.
        var old=legacy;old["gradientCapture"] = .object(["legacy":.bool(true)])
        XCTAssertNoThrow(try WorkflowPhaseArtifacts.validateUnselectedGradientMetadata(old,derivativeCapture:nil))
    }
    func testThreeReservedCadenceAndSummaryFieldsCannotAppearWithoutPolicy()throws {
        for (section,key,value) in [("settings","microbatchesPerEpoch",JSONValue.integer(3)),("settings","optimizerUpdatesPerEpoch",JSONValue.integer(2)),("runtime","gradientSummaryContext",JSONValue.object(["schemaVersion":1,"phase":.string("afterBackwardBeforeStep"),"values":.string("accumulatedWindowSoFar")]))] {
            var bad=legacy
            guard case .object(var fields)=bad[section] else {return XCTFail("Fixture section missing")}
            fields[key]=value;bad[section] = .object(fields)
            XCTAssertThrowsError(try WorkflowPhaseArtifacts.validateUnselectedGradientMetadata(bad,derivativeCapture:nil),key)
        }
    }
    func testForeignRecipesCadenceDerivativeArtifactAndPermissionAreRefused()throws {
        for section in ["settings","provenance","contract"] {for name in ["gradientPolicy","trainability"] {
            var bad=legacy
            let key=section == "settings" ? name+"Recipe":name
            if section == "contract" {bad["checkpoint"] = .object(["contract":.object([key:.object([:])])])}
            else {bad[section] = .object([key:.object([:])])}
            XCTAssertThrowsError(try WorkflowPhaseArtifacts.validateUnselectedGradientMetadata(bad,derivativeCapture:nil),section+"."+key)
        }}
        for field in ["derivativeCapture","derivativeResults"] {var bad=legacy;bad[field] = .object([:]);XCTAssertThrowsError(try WorkflowPhaseArtifacts.validateUnselectedGradientMetadata(bad,derivativeCapture:nil),field)}
        var cadence=legacy;cadence["runtime"] = .object(["updateCadence":.object([:])]);XCTAssertThrowsError(try WorkflowPhaseArtifacts.validateUnselectedGradientMetadata(cadence,derivativeCapture:nil))
        for values in [false,true] {XCTAssertThrowsError(try WorkflowPhaseArtifacts.validateUnselectedGradientMetadata(legacy,derivativeCapture:.init(values:values)))}
    }
}
