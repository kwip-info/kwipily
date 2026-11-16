import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TrainDataAccessDraftTests:XCTestCase {
    func testOmissionPreservesDefaultPathAndOtherTrainChoices() throws {
        let original:[String:JSONValue]=["trainingMode":.string("fresh")]
        var draft=try TrainDataAccessDraft(configuration:original)
        XCTAssertFalse(draft.accessEnabled);XCTAssertFalse(draft.batchingEnabled)
        XCTAssertEqual(try draft.configuration(preserving:original),original)
        draft.accessEnabled=true;draft.batchingEnabled=true;draft.batchSize="7"
        let config=try draft.configuration(preserving:original)
        XCTAssertEqual(config["trainingMode"],original["trainingMode"])
        XCTAssertEqual(try TrainDataAccessDraft(configuration:config).configuration(preserving:config),config)
        draft.accessEnabled=false;draft.batchingEnabled=false
        XCTAssertEqual(try draft.configuration(preserving:config),original)
    }
    func testTypedPositionsAndReaderTransportConstraints() throws {
        var draft=try TrainDataAccessDraft(configuration:[:]);draft.accessEnabled=true;draft.kind="subset";draft.indices="4, 1\n0"
        let parsed=try XCTUnwrap(draft.configuration(preserving:[:])["datasetAccess"])
        guard case .object(let fields)=parsed else {return XCTFail("Missing configuration")}
        XCTAssertEqual(fields["indices"],.array([4,1,0]))
        draft.indices="1,1";XCTAssertThrowsError(try draft.configuration(preserving:[:]))
        draft.kind="boundedIterable";draft.maximumRows="5";draft.batchingEnabled=true;draft.shuffle=true
        XCTAssertThrowsError(try draft.configuration(preserving:[:]))
        draft.shuffle=false;draft.workers="1";draft.timeout="0"
        XCTAssertThrowsError(try draft.configuration(preserving:[:]))
        draft.timeout="2";draft.prefetch="2"
        XCTAssertNoThrow(try draft.configuration(preserving:[:]))
        draft.batchSize="7.5";XCTAssertThrowsError(try draft.configuration(preserving:[:]))
    }
    func testSharedOriginsRequireExplicitCapturedTrainRoles() throws {
        let train=WorkflowExamples.trainAddress,foreign=WorkflowNodeAddress(domain:.control,localID:"foreign")
        XCTAssertNoThrow(try WorkflowDataLoadingVerifier.validateSources(dataset:train,loader:train,trainOrigin:train,inlineRoles:["dataset","loader"]))
        XCTAssertThrowsError(try WorkflowDataLoadingVerifier.validateSources(dataset:train,loader:train,trainOrigin:train,inlineRoles:[]))
        XCTAssertThrowsError(try WorkflowDataLoadingVerifier.validateSources(dataset:foreign,loader:foreign,trainOrigin:train,inlineRoles:["dataset","loader"]))
        XCTAssertThrowsError(try WorkflowDataLoadingVerifier.validateSources(dataset:nil,loader:train,trainOrigin:train,inlineRoles:["dataset","loader"]))
        XCTAssertThrowsError(try WorkflowDataLoadingVerifier.validateSources(dataset:train,loader:nil,trainOrigin:train,inlineRoles:["unknown"]))
    }
}
