import XCTest
@testable import KwipMLCore

final class FineTuningConfigurationTests:XCTestCase {
    func testNumericControlsRemainStableAfterSaveAndReload()throws {
        for supplied:[String:JSONValue] in [["alpha":.number(16),"dropout":.number(0)],["alpha":.number(1.5),"dropout":.number(0.1)]] {
            let original=try FineTuningConfiguration(supplied)
            let decoded=try JSONDecoder().decode([String:JSONValue].self,from:JSONEncoder().encode(original.values))
            XCTAssertEqual(original.values,decoded)
            XCTAssertEqual(try FineTuningConfiguration(decoded),original)
        }
    }
    func testLegacyTrainKeepsOriginalContract()throws {
        XCTAssertNil(try FineTuningConfiguration.decode(["runSetup":.object([:])]))
    }
    func testDefaultsAndStrictBoundaries()throws {
        let c=try FineTuningConfiguration([:])
        XCTAssertEqual(c.values["mode"],.string("full"))
        XCTAssertEqual(c.values["maximumTokens"],128)
        XCTAssertThrowsError(try FineTuningConfiguration(["remoteCode":true]))
        XCTAssertThrowsError(try FineTuningConfiguration(["maximumTokens":8,"stride":8]))
        XCTAssertThrowsError(try FineTuningConfiguration(["rank":true]))
        XCTAssertThrowsError(try FineTuningConfiguration(["dropout":1]))
        XCTAssertThrowsError(try FineTuningConfiguration(["groups":.array([.string("block:0"),.string("block:0")])]))
    }
    func testNewSchemaRoundTripAndSourceSpecificPorts()throws {
        let document=try FineTuningPracticeExample.document()
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
        let train=try XCTUnwrap(document.nodes.first{$0.address==FineTuningPracticeExample.train})
        let models=ComponentRegistry(),data=DataFlowOperatorRegistry(schemaVersion:1,packageID:"fixture",operators:[])
        XCTAssertEqual(try WorkflowPortContracts.resolve(train,document:document,modelRegistry:models,dataRegistry:data).inputs.first?.kind,.frozenModel)
        var authored=train
        guard case .train(var c)=authored.payload else {return XCTFail("Missing Train")}
        c.removeValue(forKey:"fineTuning");authored.payload = .train(config:c)
        XCTAssertEqual(try WorkflowPortContracts.resolve(authored,document:document,modelRegistry:models,dataRegistry:data).inputs.first?.kind,.modelDefinition)
        var old=document;old.schemaVersion=20
        XCTAssertThrowsError(try WorkflowCodec.encode(old),"Old schemas cannot silently reinterpret new task controls")
    }
    func testSaveContentsRemainStrictAndLegacyDefaultsRemain()throws {
        XCTAssertEqual(try WorkflowSaveModelConfiguration.decode([:]).content,.full)
        for content in WorkflowSaveModelConfiguration.Content.allCases {
            let c=WorkflowSaveModelConfiguration(format:.huggingFace,content:content)
            XCTAssertEqual(try WorkflowSaveModelConfiguration.decode(c.values),c)
        }
        XCTAssertThrowsError(try WorkflowSaveModelConfiguration.decode(["format":.string("package"),"content":.string("adapter")]))
        XCTAssertThrowsError(try WorkflowSaveModelConfiguration.decode(["format":.string("huggingFace"),"content":.string("arbitrary")]))
    }

}

extension FineTuningConfigurationTests {
    func testExpansionIsOptionalAndRoundTripsWithoutRewritingLegacyDefaults()throws {
        let original=try FineTuningConfiguration([:])
        XCTAssertNil(original.values["exampleFormat"])
        XCTAssertNil(original.values["checkpointCount"])
        var values=FineTuningConfiguration.expansionDefaults
        values["exampleFormat"] = .string("promptResponse")
        values["startMode"] = .string("resume");values["resumeRecord"] = .string(UUID().uuidString)
        let config=try FineTuningConfiguration(values)
        XCTAssertEqual(try FineTuningConfiguration(JSONDecoder().decode([String:JSONValue].self,from:JSONEncoder().encode(config.values))),config)
    }
    func testExpansionRejectsInvalidMasksResumeAndUnboundedRetention() {
        for supplied:[String:JSONValue] in [["exampleFormat":.string("arbitraryCode")],["promptColumn":.string("same"),"responseColumn":.string("same"),"exampleFormat":.string("promptResponse")],["resumeRecord":.string("/tmp/model.pt")],["checkpointCount":11],["checkpointBytes":0],["activationCheckpointing":1],["separator":.string(String(repeating:"x",count:257))]] {
            XCTAssertThrowsError(try FineTuningConfiguration(supplied))
        }
    }
}
