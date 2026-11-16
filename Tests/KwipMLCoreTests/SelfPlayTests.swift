import XCTest
@testable import KwipMLCore
final class SelfPlayTests:XCTestCase {
    func registries()throws->(ComponentRegistry,DataFlowOperatorRegistry){(try ComponentRegistry.builtIn(),try DataFlowOperatorRegistry.load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json")))}
    func testDifferentRuleDefinitionsCompileExplicitPairedHeadsAndRoundTrip()throws {
        let (m,d)=try registries()
        for example in GameRuleConfiguration.examples {
            let doc=try SelfPlayPractice.document(example:example);XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(doc)),doc)
            let plan=try SelfPlayPlan.compile(doc,address:SelfPlayPractice.train,modelRegistry:m,dataRegistry:d)
            XCTAssertEqual(plan.model.graph.nodes.filter{$0.type=="output"}.first?.config,["task":.string("gamePolicyValue"),"taskVersion":1])
            let node=try XCTUnwrap(doc.nodes.first{$0.address==SelfPlayPractice.train});let ports=try WorkflowPortContracts.resolve(node,document:doc,modelRegistry:m,dataRegistry:d)
            XCTAssertEqual(ports.inputs.map(\.id),["model","definition"]);XCTAssertEqual(ports.outputs.map(\.id),["model","iterations","episodes","moves","evaluation"])
        }
    }
    func testLegacyOutputAndOldWorkflowKeepTheirMeaning()throws {
        XCTAssertEqual(try ModelOutputConfiguration([:]).values,ModelOutputConfiguration.defaults)
        var doc=try SelfPlayPractice.document();doc.schemaVersion=22;XCTAssertThrowsError(try WorkflowCodec.validate(doc))
        for c:[String:JSONValue] in [["selfPlay":.object(["games":3])],["selfPlay":.object(["device":.string("cuda")])],["selfPlay":.object(["iterations":99,"updates":4096])],["selfPlay":.object([:]),"dataMapping":.object([:])]]{XCTAssertThrowsError(try SelfPlayConfiguration(c))}
    }
    func testCampaignBudgetsArePerSliceAndAdvancedProfilesRemainExplicit()throws {
        var fields=SelfPlayConfiguration.defaults
        fields["campaign"]=SelfPlayConfiguration.advancedDefaults["campaign"]
        if case .object(var c)=fields["campaign"] {c["enabled"]=true;fields["campaign"] = .object(c)}
        fields["iterations"]=1000
        let plan=try SelfPlayConfiguration(["selfPlay":.object(fields)])
        XCTAssertEqual(plan.values["iterations"],1000)
        fields["device"] = .string("mps");fields["compute"]=SelfPlayConfiguration.advancedDefaults["compute"]
        XCTAssertNoThrow(try SelfPlayConfiguration(["selfPlay":.object(fields)]))
        if case .object(var c)=fields["campaign"] {c["sliceIterations"]=99;fields["campaign"] = .object(c)}
        XCTAssertThrowsError(try SelfPlayConfiguration(["selfPlay":.object(fields)]))
        XCTAssertNil(try SelfPlayConfiguration(["selfPlay":.object([:])]).values["campaign"])
    }
}
