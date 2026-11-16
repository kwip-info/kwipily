import XCTest
@testable import KwipMLCore
final class GameSessionTests:XCTestCase {
    func registries()throws->(ComponentRegistry,DataFlowOperatorRegistry){(try ComponentRegistry.builtIn(),try DataFlowOperatorRegistry.load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json")))}
    func testConfigurationRejectsInvalidAndDeclaresTypedInputs()throws {
        let c=try GameSessionConfiguration([:]);XCTAssertEqual(c.inputPorts.map(\.kind),[.table,.modelDefinition,.modelDefinition])
        let random=try GameSessionConfiguration(["policies":.object(["A":.object(["source":.string("random")]),"B":.object(["source":.string("random")])])]);XCTAssertEqual(random.inputPorts.count,1)
        for raw:[String:JSONValue] in [["episodes":3],["episodes":true],["unexpected":1],["maxMoves":4097],["device":.string("cuda")],["policies":.object(["A":.object(["source":.string("sameA")])])]] {XCTAssertThrowsError(try GameSessionConfiguration(raw))}
    }
    func testDifferentGamesCompileAndRoundTripWithoutTraining()throws {
        let (models,data)=try registries()
        for example in GameRuleConfiguration.examples {
            let doc=try GameEpisodePractice.document(example:example)
            let round=try WorkflowCodec.decode(WorkflowCodec.encode(doc));XCTAssertEqual(doc,round)
            let plan=try GameSessionPlan.compile(doc,address:GameEpisodePractice.session,modelRegistry:models,dataRegistry:data)
            XCTAssertEqual(plan.canvasPolicies.count,1);XCTAssertEqual(plan.configuration.source("B"),"random")
            XCTAssertEqual(plan.canvasPolicies["A"]?.graph.nodes.count,3)
            XCTAssertFalse(doc.nodes.contains {if case .train=$0.payload{return true};return false})
        }
    }
    func testOrdinaryTableCannotImpersonateDefinitionAndSourceChangePreservesWires()throws {
        let (models,data)=try registries();var doc=try GameEpisodePractice.document()
        let index=try XCTUnwrap(doc.nodes.firstIndex(where:{$0.address==GameEpisodePractice.session}))
        doc.nodes[index].payload = .gameSession(config:["policies":.object(["A":.object(["source":.string("random")]),"B":.object(["source":.string("random")])])])
        XCTAssertThrowsError(try GameSessionPlan.compile(doc,address:GameEpisodePractice.session,modelRegistry:models,dataRegistry:data))
        doc=try GameEpisodePractice.document();doc.edges[0].from.port="summary"
        XCTAssertThrowsError(try GameSessionPlan.compile(doc,address:GameEpisodePractice.session,modelRegistry:models,dataRegistry:data))
    }
    func testDownstreamPortsStayDistinctAndCapturedPreparationOmitsSession()throws {
        let doc=try GameEpisodePractice.document();let expected=Set(["episodes","moves"])
        let sources=doc.edges.filter{$0.from.address==GameEpisodePractice.session};XCTAssertEqual(Set(sources.map(\.from.port)),expected)
        let pins=Dictionary(uniqueKeysWithValues:sources.map{($0.from,String(repeating:"a",count:64))})
        let derived=try WorkflowPredictionPreparation(document:doc,pins:pins)
        XCTAssertEqual(derived.sources.count,2);XCTAssertEqual(Set(derived.sources.keys).count,2)
        XCTAssertFalse(derived.document.nodes.contains{if case .gameSession=$0.payload{return true};return false})
    }
}
extension GameSessionTests {
    func testPracticeIsSourceFreeAndDoesNotBorrowTrainingCompiler()throws {
        let example=PreparedWorkflowExample.component(GameEpisodePractice.lessonID)
        XCTAssertTrue(example.sourceResources.isEmpty)
        XCTAssertEqual(try example.document().schemaVersion,22)
        XCTAssertNoThrow(try WorkflowTrainRunSetup.migrate(example.document()))
    }
    func testChangingSourceRefusesWiredPortAndVersion22PreservesTrainSettings()throws {
        let (models,data)=try registries();let document=try GameEpisodePractice.document()
        let fields:[String:JSONValue]=["policies":.object(["A":.object(["source":.string("saved")]),"B":.object(["source":.string("random")])])]
        XCTAssertThrowsError(try WorkflowEditing.apply(.configure(address:GameEpisodePractice.session,config:fields),to:document,modelRegistry:models,dataRegistry:data))
        var training=try FineTuningPracticeExample.document();training.schemaVersion=22
        XCTAssertNoThrow(try WorkflowCodec.validate(training))
        XCTAssertNoThrow(try WorkflowTrainRunSetup.setting([:],in:["runSetup":.object([:])],schemaVersion:22))
    }
}
