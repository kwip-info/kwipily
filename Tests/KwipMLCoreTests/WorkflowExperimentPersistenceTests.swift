import XCTest
@testable import KwipMLCore

final class WorkflowExperimentPersistenceTests:XCTestCase {
    func testExplicitMigrationRoundTripAndSingleRunPreservation()throws {
        let original=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        var document=try WorkflowCodec.migrateToVersion4(original)
        let index=try XCTUnwrap(document.nodes.firstIndex {if case .runSetup = $0.payload{return true};return false})
        guard case .runSetup(var config)=document.nodes[index].payload else {return XCTFail()}
        config["experiment"] = .object(["schemaVersion":1,"axes":.array([.object(["field":.string("seed"),"values":.array([7,8])])]),"budget":.object(["maximumRuns":2,"maximumTotalEpochs":6])])
        document.nodes[index].payload = .runSetup(config:config)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
        XCTAssertEqual(try WorkflowCodec.migrateToVersion3(document),document,"Lifecycle editing must not downgrade experiment documents")
        let spec=try XCTUnwrap(WorkflowExperimentSpecification.decode(config["experiment"]!))
        XCTAssertEqual(spec.axes,[.init(field:"seed",values:[7,8])])
        let settings=try WorkflowControlContracts.runSettings(config,address:document.nodes[index].address)
        XCTAssertEqual(settings.seed,42,"Single Run uses configured seed, never the first experiment value")
        for version in [1,2,3] {var old=document;old.schemaVersion=version;XCTAssertThrowsError(try WorkflowCodec.encode(old))}
        XCTAssertNotEqual(original.schemaVersion,4)
    }
    func testStrictSpecificationRejectsUnknownAndDuplicateFields()throws {
        let valid:JSONValue = .object(["schemaVersion":1,"axes":.array([.object(["field":.string("seed"),"values":.array([7,8])])]),"budget":.object(["maximumRuns":2,"maximumTotalEpochs":6])])
        XCTAssertNoThrow(try WorkflowExperimentSpecification.decode(valid))
        guard case .object(var fields)=valid else {return XCTFail()}
        fields["autoLaunch"] = .bool(true)
        XCTAssertThrowsError(try WorkflowExperimentSpecification.decode(.object(fields)))
        fields.removeValue(forKey:"autoLaunch")
        fields["axes"] = .array([.object(["field":.string("seed"),"values":.array([7,7])])])
        XCTAssertThrowsError(try WorkflowExperimentSpecification.decode(.object(fields)))
    }
}
